"""
NL2SQL - Natural Language to SQL Query Generator
Flask-based web application that converts natural language questions
into syntactically correct SQL queries.
"""

from flask import Flask, render_template, request, jsonify
import re
import json

app = Flask(__name__)

# ─────────────────────────────────────────────
# Sample Database Schema for Demo
# ─────────────────────────────────────────────
SAMPLE_SCHEMAS = {
    "employees": {
        "table": "employees",
        "columns": ["id", "name", "age", "department", "salary", "hire_date", "manager_id", "email", "city"],
        "description": "Employee records"
    },
    "products": {
        "table": "products",
        "columns": ["id", "name", "category", "price", "stock", "rating", "brand", "created_date"],
        "description": "Product catalog"
    },
    "orders": {
        "table": "orders",
        "columns": ["id", "customer_name", "product_id", "quantity", "total_amount", "order_date", "status", "shipping_city"],
        "description": "Customer orders"
    },
    "students": {
        "table": "students",
        "columns": ["id", "name", "age", "grade", "gpa", "major", "enrollment_date", "email"],
        "description": "Student records"
    }
}

# ─────────────────────────────────────────────
# NL2SQL Engine (Rule-based + Pattern Matching)
# ─────────────────────────────────────────────
class NL2SQLEngine:
    """
    A rule-based Natural Language to SQL converter.
    In production, this would be replaced with a fine-tuned
    Transformer model (T5/BART on WikiSQL/Spider datasets).
    """

    def __init__(self):
        self.aggregate_patterns = {
            r'\b(how many|count|number of|total number)\b': 'COUNT',
            r'\b(average|avg|mean)\b': 'AVG',
            r'\b(sum of|total of)\b': 'SUM',
            r'\b(maximum|max|highest|most|greatest)\b': 'MAX',
            r'\b(minimum|min|lowest|least|smallest)\b': 'MIN',
        }

        self.condition_patterns = {
            r'\b(greater than|more than|above|over|exceeds?|higher than)\s+([\d.]+)': '> {}',
            r'\b(less than|below|under|fewer than|lower than)\s+([\d.]+)': '< {}',
            r'\b(equal to|equals?|exactly)\s+([\d.]+)': '= {}',
            r'\b(at least|no less than|minimum of)\s+([\d.]+)': '>= {}',
            r'\b(at most|no more than|maximum of)\s+([\d.]+)': '<= {}',
            r'\b(between)\s+([\d.]+)\s+(?:and)\s+([\d.]+)': 'BETWEEN {} AND {}',
        }

        self.order_patterns = {
            r'\b(order by|sort by|sorted by|ordered by|arrange by)\b': True,
            r'\b(ascending|asc|lowest first|smallest first|a-z)\b': 'ASC',
            r'\b(descending|desc|highest first|largest first|z-a|top)\b': 'DESC',
        }

    def detect_table(self, text):
        """Detect which table the query refers to."""
        text_lower = text.lower()
        for key, schema in SAMPLE_SCHEMAS.items():
            if key in text_lower or key.rstrip('s') in text_lower:
                return schema
        # Default to employees
        return SAMPLE_SCHEMAS["employees"]

    def detect_columns(self, text, schema):
        """Detect which columns are referenced."""
        text_lower = text.lower()
        found = []
        for col in schema["columns"]:
            col_variants = [col, col.replace('_', ' ')]
            for variant in col_variants:
                if variant in text_lower:
                    found.append(col)
                    break
        return found

    def detect_aggregate(self, text):
        """Detect aggregate functions."""
        text_lower = text.lower()
        # "top N" is for LIMIT, not MAX
        if re.search(r'\btop\s+\d+\b', text_lower):
            # Don't treat "top N" as MAX unless there's also a max keyword
            if not re.search(r'\b(maximum|max|highest|most|greatest)\b', text_lower):
                return None
        for pattern, func in self.aggregate_patterns.items():
            if re.search(pattern, text_lower):
                # Exclude "top" when followed by a number
                if func == 'MAX' and re.search(r'\btop\s+\d+\b', text_lower):
                    continue
                return func
        return None

    def detect_conditions(self, text, schema):
        """Detect WHERE conditions."""
        text_lower = text.lower()
        conditions = []

        # Numeric conditions
        for col in schema["columns"]:
            col_variants = [col, col.replace('_', ' ')]
            matched = False
            for variant in col_variants:
                if matched:
                    break
                for pattern, template in self.condition_patterns.items():
                    match = re.search(f'{variant}\\s+(?:is\\s+)?{pattern}', text_lower)
                    if not match:
                        match = re.search(f'{pattern}\\s+(?:in\\s+)?{variant}', text_lower)
                    if match:
                        groups = match.groups()
                        nums = [g for g in groups if g and re.match(r'^[\d.]+$', g)]
                        if nums:
                            cond = template.format(*nums)
                            new_cond = f"{col} {cond}"
                            if new_cond not in conditions:
                                conditions.append(new_cond)
                            matched = True
                            break

        # String equality patterns - more specific matching
        # "in X department", "from X city", "in X category"
        context_patterns = [
            (r"in\s+(?:the\s+)?['\"]?(\w[\w\s]*?)\s+department", "department"),
            (r"(?:from|in)\s+(?:the\s+)?['\"]?(\w[\w\s]*?)\s+city", "city"),
            (r"in\s+(?:the\s+)?['\"]?(\w[\w\s]*?)\s+(?:category|categories)", "category"),
            (r"(?:named?|called)\s+['\"]?(\w[\w\s]+?)['\"]?(?:\s|$|,|\.)", "name"),
            (r"department\s+(?:is|=|:)\s+['\"]?(\w[\w\s]*?)['\"]?(?:\s|$|,|\.)", "department"),
            (r"(?:category|status|major|brand)\s+(?:is|=|:)\s+['\"]?(\w[\w\s]*?)['\"]?(?:\s|$|,|\.)", None),
            (r"with\s+status\s+['\"]?(\w[\w\s]*?)['\"]?(?:\s|$|,|\.)", "status"),
            (r"in\s+(?:the\s+)?['\"]?(\w[\w\s]*?)['\"]?\s+major", "major"),
            (r"(?:department|category|brand|status)\s+['\"]?(\w[\w\s]*?)['\"]?(?:\s|$|,|\.)", None),
        ]

        # Stopwords to skip
        stop_words = {'the', 'a', 'an', 'each', 'every', 'all', 'is', 'are', 'in',
                      'from', 'with', 'by', 'for', 'of', 'to', 'and', 'or', 'their',
                      'this', 'that', 'what', 'how', 'many', 'much', 'show', 'find',
                      'get', 'list', 'display', 'give', 'me', 'there'}

        for pattern, target_col in context_patterns:
            match = re.search(pattern, text_lower)
            if match:
                value = match.group(1).strip()
                if value.lower() in stop_words or len(value) < 2:
                    continue
                if target_col and target_col in schema["columns"]:
                    conditions.append(f"{target_col} = '{value.title()}'")
                elif target_col is None:
                    # Try to match the keyword to a column
                    keyword_match = re.match(r'(\w+)', pattern.replace('(?:', '').replace(')', ''))
                    for col in ['category', 'status', 'major', 'brand', 'department']:
                        if col in schema["columns"] and col in pattern:
                            conditions.append(f"{col} = '{value.title()}'")
                            break

        # Deduplicate conditions
        seen = set()
        unique_conditions = []
        for cond in conditions:
            if cond not in seen:
                seen.add(cond)
                unique_conditions.append(cond)

        return unique_conditions

    def detect_ordering(self, text, schema):
        """Detect ORDER BY clause."""
        text_lower = text.lower()
        has_order = False
        direction = 'ASC'

        for pattern in self.order_patterns:
            if re.search(pattern, text_lower):
                result = self.order_patterns[pattern]
                if result is True:
                    has_order = True
                elif result in ('ASC', 'DESC'):
                    has_order = True
                    direction = result

        # "top N" pattern implies ORDER BY DESC LIMIT
        if re.search(r'\btop\s+\d+', text_lower):
            has_order = True
            direction = 'DESC'

        if has_order:
            # Find the column to order by
            cols = self.detect_columns(text, schema)
            numeric_cols = ['salary', 'price', 'age', 'stock', 'rating', 'gpa',
                          'total_amount', 'quantity']
            order_col = None
            for c in cols:
                if c in numeric_cols:
                    order_col = c
                    break
            if not order_col and cols:
                order_col = cols[-1]
            if order_col:
                return f"ORDER BY {order_col} {direction}"

        return None

    def detect_limit(self, text):
        """Detect LIMIT clause."""
        text_lower = text.lower()
        match = re.search(r'\btop\s+(\d+)', text_lower)
        if match:
            return f"LIMIT {match.group(1)}"
        match = re.search(r'\bfirst\s+(\d+)', text_lower)
        if match:
            return f"LIMIT {match.group(1)}"
        match = re.search(r'\blimit\s+(\d+)', text_lower)
        if match:
            return f"LIMIT {match.group(1)}"
        return None

    def detect_group_by(self, text, schema):
        """Detect GROUP BY clause."""
        text_lower = text.lower()

        # Only group if there's an aggregate function
        agg = self.detect_aggregate(text)
        if not agg:
            return None

        group_triggers = [
            r'\b(?:by|per|each|every|for each|group by|grouped by)\s+(\w+)',
            r'\b(\w+)[\s-]wise\b',
        ]
        categorical = ['department', 'category', 'status', 'city',
                       'shipping_city', 'brand', 'major', 'grade']

        for pattern in group_triggers:
            for match in re.finditer(pattern, text_lower):
                word = match.group(1) if match.lastindex else None
                if word:
                    for col in schema["columns"]:
                        if (word in col or col in word) and col in categorical:
                            return f"GROUP BY {col}"

        # If aggregate and categorical column mentioned
        cols = self.detect_columns(text, schema)
        for c in cols:
            if c in categorical:
                return f"GROUP BY {c}"

        return None

    def generate_sql(self, natural_language):
        """Main method: Convert natural language to SQL."""
        text = natural_language.strip()
        if not text:
            return {"sql": "", "explanation": "Please enter a question.", "confidence": 0}

        schema = self.detect_table(text)
        table = schema["table"]
        columns = self.detect_columns(text, schema)
        aggregate = self.detect_aggregate(text)
        conditions = self.detect_conditions(text, schema)
        ordering = self.detect_ordering(text, schema)
        limit = self.detect_limit(text)
        group_by = self.detect_group_by(text, schema)

        # Build SELECT clause
        # Determine if user wants all columns
        text_lower = text.lower()
        wants_all = bool(re.search(r'\b(show|list|display|find|get)\s+(all|every)\b', text_lower))
        wants_all = wants_all or (not columns and not aggregate)

        # Separate condition columns from display columns
        condition_cols = set()
        for cond in conditions:
            col_match = re.match(r'(\w+)\s+', cond)
            if col_match:
                condition_cols.add(col_match.group(1))

        display_columns = [c for c in columns if c not in condition_cols]

        if aggregate:
            if columns:
                target_col = columns[-1]
                numeric_cols = ['salary', 'price', 'age', 'stock', 'rating',
                              'gpa', 'total_amount', 'quantity', 'id']
                if aggregate in ('COUNT',):
                    select_clause = f"{aggregate}(*)"
                elif target_col in numeric_cols:
                    select_clause = f"{aggregate}({target_col})"
                else:
                    select_clause = f"{aggregate}(*)"
            else:
                select_clause = f"{aggregate}(*)"

            if group_by:
                group_col = group_by.replace("GROUP BY ", "")
                select_clause = f"{group_col}, {select_clause}"
        elif display_columns and not wants_all:
            select_clause = ", ".join(display_columns)
        else:
            select_clause = "*"

        # Build the query
        sql_parts = [f"SELECT {select_clause}", f"FROM {table}"]

        if conditions:
            sql_parts.append(f"WHERE {' AND '.join(conditions)}")

        if group_by:
            sql_parts.append(group_by)

        if ordering:
            sql_parts.append(ordering)

        if limit:
            sql_parts.append(limit)

        sql = "\n".join(sql_parts) + ";"

        # Calculate confidence
        confidence = 0.5
        if columns or aggregate:
            confidence += 0.15
        if conditions:
            confidence += 0.15
        if group_by or ordering:
            confidence += 0.1
        confidence = min(confidence, 0.95)

        # Build explanation
        explanation_parts = []
        explanation_parts.append(f"Detected table: <strong>{table}</strong>")
        if aggregate:
            explanation_parts.append(f"Aggregate function: <strong>{aggregate}</strong>")
        if columns:
            explanation_parts.append(f"Columns: <strong>{', '.join(columns)}</strong>")
        if conditions:
            explanation_parts.append(f"Conditions: <strong>{', '.join(conditions)}</strong>")
        if group_by:
            explanation_parts.append(f"Grouping: <strong>{group_by.replace('GROUP BY ', '')}</strong>")
        if ordering:
            explanation_parts.append(f"Ordering: <strong>{ordering.replace('ORDER BY ', '')}</strong>")

        return {
            "sql": sql,
            "explanation": " · ".join(explanation_parts),
            "confidence": round(confidence * 100),
            "schema": {
                "table": table,
                "columns": schema["columns"]
            }
        }


# Initialize engine
engine = NL2SQLEngine()

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', schemas=SAMPLE_SCHEMAS)


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    question = data.get('question', '')
    result = engine.generate_sql(question)
    return jsonify(result)


@app.route('/examples')
def examples():
    return jsonify([
        {"question": "Show all employees in the Engineering department", "category": "Basic Select"},
        {"question": "How many products are there in each category?", "category": "Aggregation"},
        {"question": "Find employees with salary greater than 80000", "category": "Filtering"},
        {"question": "Show top 5 products by price descending", "category": "Sorting"},
        {"question": "What is the average salary by department?", "category": "Group By"},
        {"question": "Count total orders with status Shipped", "category": "Count"},
        {"question": "Show students with gpa greater than 3.5 in Computer Science major", "category": "Multi-filter"},
        {"question": "Find the maximum price of products in Electronics category", "category": "Max"},
        {"question": "List all orders with total amount above 500 sorted by order date descending", "category": "Complex"},
        {"question": "Show the minimum age of employees by department", "category": "Min + Group"},
    ])


if __name__ == '__main__':
    app.run(debug=True, port=5000)
