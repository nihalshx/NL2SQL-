"""
data_preprocessor.py
────────────────────
Dataset loading, preprocessing, and schema-aware serialization
for training Text-to-SQL Transformer models.

Supports:
  • WikiSQL  – single-table, simple queries
  • Spider   – multi-table, complex queries (JOINs, nested, etc.)
  • Custom   – any CSV of (question, sql, schema) triples

Schema serialization format (fed as prefix to the encoder):
  "table: employees | columns: id, name, age, department, salary"
"""

import json
import csv
import os
import random
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────
@dataclass
class TableSchema:
    """Represents a single database table schema."""
    table_name: str
    columns: List[str]
    column_types: List[str] = field(default_factory=list)
    primary_key: Optional[str] = None
    foreign_keys: List[Dict] = field(default_factory=list)

    def serialize(self) -> str:
        """Convert schema to a flat string for model input."""
        cols = ", ".join(self.columns)
        return f"table: {self.table_name} | columns: {cols}"

    def serialize_typed(self) -> str:
        """Serialize with column types for richer context."""
        if not self.column_types:
            return self.serialize()
        typed_cols = ", ".join(
            f"{c} ({t})" for c, t in zip(self.columns, self.column_types)
        )
        return f"table: {self.table_name} | columns: {typed_cols}"


@dataclass
class DatabaseSchema:
    """Represents a full database schema (multiple tables)."""
    db_id: str
    tables: List[TableSchema]

    def serialize(self) -> str:
        return " | ".join(t.serialize() for t in self.tables)

    def serialize_typed(self) -> str:
        return " | ".join(t.serialize_typed() for t in self.tables)


@dataclass
class NL2SQLSample:
    """A single (question, SQL, schema) training sample."""
    question: str
    sql: str
    schema: str              # serialized schema string
    db_id: str = ""
    difficulty: str = ""     # easy / medium / hard / extra


# ─────────────────────────────────────────────
# WikiSQL Preprocessor
# ─────────────────────────────────────────────
class WikiSQLPreprocessor:
    """
    Loads and preprocesses the WikiSQL dataset.

    Expected file structure:
        data/
        ├── train.jsonl
        ├── dev.jsonl
        ├── test.jsonl
        └── train.tables.jsonl
    """

    SQL_OPS = {0: "", 1: "MAX", 2: "MIN", 3: "COUNT", 4: "SUM", 5: "AVG"}
    COND_OPS = {0: "=", 1: ">", 2: "<"}
    LOGIC_OPS = {0: "AND", 1: "OR"}

    def __init__(self, data_dir: str = "data/wikisql"):
        self.data_dir = data_dir
        self.tables: Dict[str, dict] = {}

    def load_tables(self, tables_file: str) -> None:
        """Load table schemas from WikiSQL .tables.jsonl file."""
        path = os.path.join(self.data_dir, tables_file)
        if not os.path.exists(path):
            logger.warning(f"Tables file not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                table = json.loads(line.strip())
                self.tables[table["id"]] = table
        logger.info(f"Loaded {len(self.tables)} tables from {tables_file}")

    def _build_sql(self, sql_dict: dict, table: dict) -> str:
        """Reconstruct SQL string from WikiSQL's structured representation."""
        columns = table["header"]
        table_name = table.get("name", table.get("id", "table"))
        # Clean table name
        table_name = table_name.replace("-", "_").replace(" ", "_")

        # SELECT clause
        agg = self.SQL_OPS.get(sql_dict["agg"], "")
        sel_col = columns[sql_dict["sel"]]

        if agg:
            select = f"SELECT {agg}({sel_col})"
        else:
            select = f"SELECT {sel_col}"

        # FROM clause
        from_clause = f"FROM {table_name}"

        # WHERE clause
        conditions = sql_dict.get("conds", {})
        cond_list = conditions if isinstance(conditions, list) else conditions.get("conditions", [])

        where_parts = []
        for cond in cond_list:
            col_idx, op_idx, value = cond
            col_name = columns[col_idx]
            op = self.COND_OPS.get(op_idx, "=")
            if isinstance(value, str):
                where_parts.append(f'{col_name} {op} "{value}"')
            else:
                where_parts.append(f"{col_name} {op} {value}")

        sql = f"{select} {from_clause}"
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

        return sql

    def load_split(self, split_file: str, tables_file: str = None) -> List[NL2SQLSample]:
        """Load a data split (train/dev/test) and return NL2SQLSample list."""
        if tables_file:
            self.load_tables(tables_file)

        path = os.path.join(self.data_dir, split_file)
        if not os.path.exists(path):
            logger.error(f"Split file not found: {path}")
            return []

        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip())
                table_id = row["table_id"]
                table = self.tables.get(table_id)
                if not table:
                    continue

                question = row["question"]
                sql = self._build_sql(row["sql"], table)

                # Build schema string
                schema = TableSchema(
                    table_name=table.get("name", table_id).replace("-", "_").replace(" ", "_"),
                    columns=table["header"],
                    column_types=table.get("types", [])
                )

                samples.append(NL2SQLSample(
                    question=question,
                    sql=sql,
                    schema=schema.serialize(),
                    db_id=table_id,
                    difficulty="simple"
                ))

        logger.info(f"Loaded {len(samples)} samples from {split_file}")
        return samples

    def load_all(self) -> Dict[str, List[NL2SQLSample]]:
        """Load all splits."""
        self.load_tables("train.tables.jsonl")
        return {
            "train": self.load_split("train.jsonl"),
            "dev": self.load_split("dev.jsonl"),
            "test": self.load_split("test.jsonl"),
        }


# ─────────────────────────────────────────────
# Spider Preprocessor
# ─────────────────────────────────────────────
class SpiderPreprocessor:
    """
    Loads and preprocesses the Spider dataset.

    Expected file structure:
        data/spider/
        ├── train_spider.json
        ├── dev.json
        └── tables.json
    """

    def __init__(self, data_dir: str = "data/spider"):
        self.data_dir = data_dir
        self.schemas: Dict[str, DatabaseSchema] = {}

    def load_tables(self, tables_file: str = "tables.json") -> None:
        """Load database schemas from Spider's tables.json."""
        path = os.path.join(self.data_dir, tables_file)
        if not os.path.exists(path):
            logger.warning(f"Tables file not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            databases = json.load(f)

        for db in databases:
            db_id = db["db_id"]
            table_names = db["table_names_original"]
            col_info = db["column_names_original"]  # [(table_idx, col_name), ...]
            col_types = db.get("column_types", [])

            # Group columns by table
            tables = {}
            for i, (table_idx, col_name) in enumerate(col_info):
                if table_idx < 0:
                    continue  # skip the * wildcard
                tname = table_names[table_idx]
                if tname not in tables:
                    tables[tname] = {"columns": [], "types": []}
                tables[tname]["columns"].append(col_name)
                if i < len(col_types):
                    tables[tname]["types"].append(col_types[i])

            # Build foreign keys
            fk_pairs = db.get("foreign_keys", [])
            foreign_keys = []
            for fk in fk_pairs:
                if len(fk) == 2:
                    src_col = col_info[fk[0]]
                    tgt_col = col_info[fk[1]]
                    foreign_keys.append({
                        "source_table": table_names[src_col[0]] if src_col[0] >= 0 else "",
                        "source_col": src_col[1],
                        "target_table": table_names[tgt_col[0]] if tgt_col[0] >= 0 else "",
                        "target_col": tgt_col[1],
                    })

            table_schemas = []
            primary_keys = db.get("primary_keys", [])
            for tname, tdata in tables.items():
                table_schemas.append(TableSchema(
                    table_name=tname,
                    columns=tdata["columns"],
                    column_types=tdata["types"],
                    foreign_keys=[fk for fk in foreign_keys if fk["source_table"] == tname]
                ))

            self.schemas[db_id] = DatabaseSchema(
                db_id=db_id,
                tables=table_schemas
            )

        logger.info(f"Loaded schemas for {len(self.schemas)} databases")

    def _classify_difficulty(self, sql: str) -> str:
        """Heuristic difficulty classification."""
        sql_upper = sql.upper()
        score = 0
        if "JOIN" in sql_upper:
            score += 2
        if "SUBQUERY" in sql_upper or sql_upper.count("SELECT") > 1:
            score += 3
        if "GROUP BY" in sql_upper:
            score += 1
        if "HAVING" in sql_upper:
            score += 1
        if "ORDER BY" in sql_upper:
            score += 1
        if "UNION" in sql_upper or "INTERSECT" in sql_upper or "EXCEPT" in sql_upper:
            score += 3

        if score <= 1:
            return "easy"
        elif score <= 3:
            return "medium"
        elif score <= 5:
            return "hard"
        return "extra"

    def load_split(self, split_file: str) -> List[NL2SQLSample]:
        """Load a data split."""
        if not self.schemas:
            self.load_tables()

        path = os.path.join(self.data_dir, split_file)
        if not os.path.exists(path):
            logger.error(f"Split file not found: {path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for entry in data:
            db_id = entry["db_id"]
            schema = self.schemas.get(db_id)
            if not schema:
                continue

            question = entry["question"]
            sql = entry.get("query", entry.get("sql", ""))

            samples.append(NL2SQLSample(
                question=question,
                sql=sql,
                schema=schema.serialize(),
                db_id=db_id,
                difficulty=self._classify_difficulty(sql)
            ))

        logger.info(f"Loaded {len(samples)} samples from {split_file}")
        return samples

    def load_all(self) -> Dict[str, List[NL2SQLSample]]:
        """Load all splits."""
        self.load_tables()
        return {
            "train": self.load_split("train_spider.json"),
            "dev": self.load_split("dev.json"),
        }


# ─────────────────────────────────────────────
# Custom CSV Preprocessor
# ─────────────────────────────────────────────
class CustomCSVPreprocessor:
    """
    Load custom training data from a CSV file.

    Expected CSV columns:
        question, sql, table_name, columns
    
    Example row:
        "Show all employees", "SELECT * FROM employees", "employees", "id,name,age,salary"
    """

    def load(self, csv_path: str) -> List[NL2SQLSample]:
        """Load samples from CSV."""
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return []

        samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row.get("question", "").strip()
                sql = row.get("sql", "").strip()
                table_name = row.get("table_name", "table").strip()
                columns = row.get("columns", "").strip()

                if not question or not sql:
                    continue

                schema_str = f"table: {table_name} | columns: {columns}"
                samples.append(NL2SQLSample(
                    question=question,
                    sql=sql,
                    schema=schema_str,
                    db_id=table_name
                ))

        logger.info(f"Loaded {len(samples)} samples from {csv_path}")
        return samples


# ─────────────────────────────────────────────
# Input Formatter (Model-Ready)
# ─────────────────────────────────────────────
class InputFormatter:
    """
    Formats NL2SQL samples into model-ready input/target strings.

    Input format:
        "translate to SQL: {question} | schema: {serialized_schema}"

    Target format:
        "{sql_query}"
    """

    PREFIX = "translate to SQL"

    @staticmethod
    def format_input(sample: NL2SQLSample, include_schema: bool = True) -> str:
        """Format a sample as model input string."""
        parts = [f"{InputFormatter.PREFIX}: {sample.question}"]
        if include_schema and sample.schema:
            parts.append(f"schema: {sample.schema}")
        return " | ".join(parts)

    @staticmethod
    def format_target(sample: NL2SQLSample) -> str:
        """Format the target SQL query."""
        return sample.sql.strip()

    @staticmethod
    def format_batch(
        samples: List[NL2SQLSample],
        include_schema: bool = True
    ) -> List[Dict[str, str]]:
        """Format a batch of samples into input/target pairs."""
        return [
            {
                "input": InputFormatter.format_input(s, include_schema),
                "target": InputFormatter.format_target(s),
            }
            for s in samples
        ]

    @staticmethod
    def create_inference_input(
        question: str,
        table_name: str,
        columns: List[str]
    ) -> str:
        """Create a formatted input string for inference."""
        schema_str = f"table: {table_name} | columns: {', '.join(columns)}"
        return f"{InputFormatter.PREFIX}: {question} | schema: {schema_str}"


# ─────────────────────────────────────────────
# Data Augmentation
# ─────────────────────────────────────────────
class DataAugmenter:
    """
    Simple data augmentation strategies for NL2SQL:
      • Synonym replacement for question words
      • Column name shuffling in schema
      • Question paraphrasing templates
    """

    QUESTION_SYNONYMS = {
        "show": ["display", "list", "get", "find", "retrieve", "fetch"],
        "all": ["every", "each", "the complete list of"],
        "how many": ["count", "total number of", "number of"],
        "average": ["mean", "avg"],
        "highest": ["maximum", "max", "greatest", "top", "largest"],
        "lowest": ["minimum", "min", "least", "smallest"],
        "greater than": ["more than", "above", "over", "exceeding"],
        "less than": ["below", "under", "fewer than", "lower than"],
        "sort by": ["order by", "sorted by", "arranged by", "ranked by"],
    }

    @staticmethod
    def synonym_replace(question: str, n_replacements: int = 1) -> str:
        """Replace random phrases with synonyms."""
        result = question.lower()
        replacements_made = 0

        phrases = list(DataAugmenter.QUESTION_SYNONYMS.keys())
        random.shuffle(phrases)

        for phrase in phrases:
            if replacements_made >= n_replacements:
                break
            if phrase in result:
                synonym = random.choice(DataAugmenter.QUESTION_SYNONYMS[phrase])
                result = result.replace(phrase, synonym, 1)
                replacements_made += 1

        return result

    @staticmethod
    def augment_samples(
        samples: List[NL2SQLSample],
        augment_ratio: float = 0.3
    ) -> List[NL2SQLSample]:
        """Augment a dataset with synonym-replaced variants."""
        n_augment = int(len(samples) * augment_ratio)
        selected = random.sample(samples, min(n_augment, len(samples)))

        augmented = []
        for sample in selected:
            new_question = DataAugmenter.synonym_replace(sample.question)
            if new_question != sample.question.lower():
                augmented.append(NL2SQLSample(
                    question=new_question,
                    sql=sample.sql,
                    schema=sample.schema,
                    db_id=sample.db_id,
                    difficulty=sample.difficulty,
                ))

        logger.info(f"Augmented {len(augmented)} samples from {len(samples)} originals")
        return samples + augmented


# ─────────────────────────────────────────────
# Demo/Synthetic Data Generator
# ─────────────────────────────────────────────
class SyntheticDataGenerator:
    """
    Generates synthetic NL-SQL pairs for demo/testing when
    real datasets (WikiSQL/Spider) are not available.
    """

    TEMPLATES = [
        # Basic SELECT
        ("Show all {table}", "SELECT * FROM {table}"),
        ("List all {table}", "SELECT * FROM {table}"),
        ("Get all {table}", "SELECT * FROM {table}"),
        ("Display all {table}", "SELECT * FROM {table}"),

        # Column select
        ("Show {col1} and {col2} of {table}", "SELECT {col1}, {col2} FROM {table}"),
        ("List {col1} from {table}", "SELECT {col1} FROM {table}"),

        # WHERE with string
        ("Show {table} where {cat_col} is {value}", "SELECT * FROM {table} WHERE {cat_col} = '{value}'"),
        ("Find {table} with {cat_col} equal to {value}", "SELECT * FROM {table} WHERE {cat_col} = '{value}'"),
        ("Get all {table} in {value} {cat_col}", "SELECT * FROM {table} WHERE {cat_col} = '{value}'"),

        # WHERE with number
        ("Show {table} where {num_col} is greater than {num}", "SELECT * FROM {table} WHERE {num_col} > {num}"),
        ("Find {table} with {num_col} above {num}", "SELECT * FROM {table} WHERE {num_col} > {num}"),
        ("List {table} where {num_col} is less than {num}", "SELECT * FROM {table} WHERE {num_col} < {num}"),

        # COUNT
        ("How many {table} are there", "SELECT COUNT(*) FROM {table}"),
        ("Count all {table}", "SELECT COUNT(*) FROM {table}"),
        ("How many {table} have {cat_col} equal to {value}", "SELECT COUNT(*) FROM {table} WHERE {cat_col} = '{value}'"),

        # AVG
        ("What is the average {num_col} of {table}", "SELECT AVG({num_col}) FROM {table}"),
        ("Average {num_col} for {table}", "SELECT AVG({num_col}) FROM {table}"),
        ("What is the average {num_col} by {cat_col}", "SELECT {cat_col}, AVG({num_col}) FROM {table} GROUP BY {cat_col}"),

        # MAX/MIN
        ("What is the maximum {num_col} of {table}", "SELECT MAX({num_col}) FROM {table}"),
        ("Find the highest {num_col} in {table}", "SELECT MAX({num_col}) FROM {table}"),
        ("What is the minimum {num_col} of {table}", "SELECT MIN({num_col}) FROM {table}"),
        ("Find the lowest {num_col} in {table}", "SELECT MIN({num_col}) FROM {table}"),

        # SUM
        ("What is the total {num_col} of {table}", "SELECT SUM({num_col}) FROM {table}"),

        # GROUP BY
        ("Count {table} by {cat_col}", "SELECT {cat_col}, COUNT(*) FROM {table} GROUP BY {cat_col}"),
        ("How many {table} per {cat_col}", "SELECT {cat_col}, COUNT(*) FROM {table} GROUP BY {cat_col}"),

        # ORDER BY
        ("Show {table} ordered by {num_col} descending", "SELECT * FROM {table} ORDER BY {num_col} DESC"),
        ("List {table} sorted by {num_col}", "SELECT * FROM {table} ORDER BY {num_col} ASC"),

        # LIMIT
        ("Show top 5 {table} by {num_col}", "SELECT * FROM {table} ORDER BY {num_col} DESC LIMIT 5"),
        ("Show top 10 {table} by {num_col}", "SELECT * FROM {table} ORDER BY {num_col} DESC LIMIT 10"),
        ("Show first 3 {table}", "SELECT * FROM {table} LIMIT 3"),
    ]

    DEMO_TABLES = {
        "employees": {
            "columns": ["id", "name", "age", "department", "salary", "hire_date", "city"],
            "num_cols": ["age", "salary"],
            "cat_cols": ["department", "city"],
            "cat_values": {"department": ["Engineering", "Marketing", "Sales", "HR", "Finance"],
                          "city": ["New York", "London", "Tokyo", "Berlin", "Paris"]},
            "num_range": (20000, 150000),
        },
        "products": {
            "columns": ["id", "name", "category", "price", "stock", "rating", "brand"],
            "num_cols": ["price", "stock", "rating"],
            "cat_cols": ["category", "brand"],
            "cat_values": {"category": ["Electronics", "Clothing", "Books", "Food", "Sports"],
                          "brand": ["Apple", "Samsung", "Nike", "Sony", "Dell"]},
            "num_range": (10, 5000),
        },
        "orders": {
            "columns": ["id", "customer_name", "product_id", "quantity", "total_amount", "status"],
            "num_cols": ["quantity", "total_amount"],
            "cat_cols": ["status"],
            "cat_values": {"status": ["Pending", "Shipped", "Delivered", "Cancelled", "Returned"]},
            "num_range": (10, 10000),
        },
        "students": {
            "columns": ["id", "name", "age", "gpa", "major", "enrollment_date"],
            "num_cols": ["age", "gpa"],
            "cat_cols": ["major"],
            "cat_values": {"major": ["Computer Science", "Mathematics", "Physics", "Biology", "English"]},
            "num_range": (1, 100),
        },
    }

    @staticmethod
    def generate(n_samples: int = 1000) -> List[NL2SQLSample]:
        """Generate synthetic NL-SQL training pairs."""
        samples = []

        for _ in range(n_samples):
            table_name = random.choice(list(SyntheticDataGenerator.DEMO_TABLES.keys()))
            table = SyntheticDataGenerator.DEMO_TABLES[table_name]
            template_nl, template_sql = random.choice(SyntheticDataGenerator.TEMPLATES)

            num_col = random.choice(table["num_cols"])
            cat_col = random.choice(table["cat_cols"])
            value = random.choice(table["cat_values"][cat_col])
            num = random.randint(*table["num_range"])
            col1 = random.choice(table["columns"])
            col2 = random.choice([c for c in table["columns"] if c != col1])

            replacements = {
                "table": table_name,
                "num_col": num_col,
                "cat_col": cat_col,
                "value": value,
                "num": str(num),
                "col1": col1,
                "col2": col2,
            }

            try:
                nl = template_nl.format(**replacements)
                sql = template_sql.format(**replacements)
            except (KeyError, IndexError):
                continue

            schema = TableSchema(
                table_name=table_name,
                columns=table["columns"]
            )

            samples.append(NL2SQLSample(
                question=nl,
                sql=sql,
                schema=schema.serialize(),
                db_id=table_name,
                difficulty="synthetic"
            ))

        logger.info(f"Generated {len(samples)} synthetic samples")
        return samples


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NL2SQL Data Preprocessor")
    parser.add_argument("--dataset", choices=["wikisql", "spider", "synthetic", "custom"],
                       default="synthetic", help="Dataset to preprocess")
    parser.add_argument("--data_dir", default="data/", help="Data directory path")
    parser.add_argument("--csv_path", default=None, help="Path to custom CSV file")
    parser.add_argument("--output", default="data/processed_train.json", help="Output file")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--n_synthetic", type=int, default=2000, help="Number of synthetic samples")

    args = parser.parse_args()

    if args.dataset == "wikisql":
        preprocessor = WikiSQLPreprocessor(os.path.join(args.data_dir, "wikisql"))
        splits = preprocessor.load_all()
        samples = splits.get("train", [])
    elif args.dataset == "spider":
        preprocessor = SpiderPreprocessor(os.path.join(args.data_dir, "spider"))
        splits = preprocessor.load_all()
        samples = splits.get("train", [])
    elif args.dataset == "custom":
        preprocessor = CustomCSVPreprocessor()
        samples = preprocessor.load(args.csv_path)
    else:
        samples = SyntheticDataGenerator.generate(args.n_synthetic)

    if args.augment:
        samples = DataAugmenter.augment_samples(samples)

    # Format for model training
    formatted = InputFormatter.format_batch(samples)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(formatted)} formatted samples to {args.output}")