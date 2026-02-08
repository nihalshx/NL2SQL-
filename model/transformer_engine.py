"""
transformer_engine.py
─────────────────────
Transformer-based NL2SQL inference engine using HuggingFace T5/BART.

Supports:
  • Loading pretrained or fine-tuned seq2seq models
  • Schema-aware input encoding
  • Beam search decoding with SQL validation
  • Confidence scoring via sequence probabilities
  • Batch inference for throughput
"""

import os
import re
import math
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Lazy imports — only loaded when Transformer
# engine is actually instantiated
# ─────────────────────────────────────────────
torch = None
T5ForConditionalGeneration = None
T5Tokenizer = None
AutoModelForSeq2SeqLM = None
AutoTokenizer = None


def _import_dependencies():
    """Import heavy ML libraries only when needed."""
    global torch, T5ForConditionalGeneration, T5Tokenizer
    global AutoModelForSeq2SeqLM, AutoTokenizer

    import torch as _torch
    from transformers import (
        T5ForConditionalGeneration as _T5Model,
        T5Tokenizer as _T5Tok,
        AutoModelForSeq2SeqLM as _AutoModel,
        AutoTokenizer as _AutoTok,
    )

    torch = _torch
    T5ForConditionalGeneration = _T5Model
    T5Tokenizer = _T5Tok
    AutoModelForSeq2SeqLM = _AutoModel
    AutoTokenizer = _AutoTok


# ─────────────────────────────────────────────
# SQL Post-Processor
# ─────────────────────────────────────────────
class SQLPostProcessor:
    """
    Cleans and validates generated SQL queries.
    Fixes common generation errors like:
      • Missing semicolons
      • Unmatched quotes or parentheses
      • Duplicate keywords
      • Case normalization
    """

    SQL_KEYWORDS = {
        "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN",
        "BETWEEN", "LIKE", "IS", "NULL", "JOIN", "INNER", "LEFT",
        "RIGHT", "OUTER", "ON", "GROUP", "BY", "HAVING", "ORDER",
        "ASC", "DESC", "LIMIT", "OFFSET", "UNION", "INTERSECT",
        "EXCEPT", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
        "DELETE", "CREATE", "DROP", "ALTER", "TABLE", "INDEX",
        "COUNT", "SUM", "AVG", "MAX", "MIN", "DISTINCT", "AS",
        "EXISTS", "CASE", "WHEN", "THEN", "ELSE", "END",
    }

    @staticmethod
    def normalize_keywords(sql: str) -> str:
        """Uppercase SQL keywords while preserving values."""
        tokens = sql.split()
        result = []
        in_string = False
        quote_char = None

        for token in tokens:
            # Track string literals
            for ch in token:
                if ch in ("'", '"') and not in_string:
                    in_string = True
                    quote_char = ch
                elif ch == quote_char and in_string:
                    in_string = False

            if not in_string and token.upper() in SQLPostProcessor.SQL_KEYWORDS:
                result.append(token.upper())
            else:
                result.append(token)

        return " ".join(result)

    @staticmethod
    def fix_parentheses(sql: str) -> str:
        """Balance unmatched parentheses."""
        open_count = sql.count("(")
        close_count = sql.count(")")
        if open_count > close_count:
            sql += ")" * (open_count - close_count)
        elif close_count > open_count:
            sql = "(" * (close_count - open_count) + sql
        return sql

    @staticmethod
    def fix_quotes(sql: str) -> str:
        """Balance unmatched single quotes."""
        single = sql.count("'")
        if single % 2 != 0:
            # Find last unmatched quote and close it
            sql += "'"
        return sql

    @staticmethod
    def remove_duplicate_keywords(sql: str) -> str:
        """Remove consecutive duplicate SQL keywords."""
        tokens = sql.split()
        cleaned = [tokens[0]] if tokens else []
        for i in range(1, len(tokens)):
            if tokens[i].upper() == tokens[i - 1].upper() and tokens[i].upper() in SQLPostProcessor.SQL_KEYWORDS:
                continue
            cleaned.append(tokens[i])
        return " ".join(cleaned)

    @staticmethod
    def ensure_semicolon(sql: str) -> str:
        """Ensure query ends with semicolon."""
        sql = sql.strip()
        if sql and not sql.endswith(";"):
            sql += ";"
        return sql

    @staticmethod
    def clean(sql: str) -> str:
        """Apply all post-processing steps."""
        if not sql or not sql.strip():
            return ""
        sql = sql.strip()
        sql = SQLPostProcessor.normalize_keywords(sql)
        sql = SQLPostProcessor.fix_quotes(sql)
        sql = SQLPostProcessor.fix_parentheses(sql)
        sql = SQLPostProcessor.remove_duplicate_keywords(sql)
        sql = SQLPostProcessor.ensure_semicolon(sql)
        return sql

    @staticmethod
    def validate(sql: str) -> Dict[str, bool]:
        """Basic SQL validation checks."""
        sql_upper = sql.upper().strip().rstrip(";")
        return {
            "has_select": sql_upper.startswith("SELECT"),
            "has_from": "FROM" in sql_upper,
            "balanced_parens": sql.count("(") == sql.count(")"),
            "balanced_quotes": sql.count("'") % 2 == 0,
            "not_empty": len(sql.strip()) > 0,
        }


# ─────────────────────────────────────────────
# Transformer NL2SQL Engine
# ─────────────────────────────────────────────
class TransformerNL2SQLEngine:
    """
    Transformer-based Text-to-SQL engine.

    Uses a T5 or BART seq2seq model to generate SQL from
    natural language + schema input.

    Usage:
        engine = TransformerNL2SQLEngine("path/to/finetuned-model")
        result = engine.generate_sql(
            question="Show employees with salary above 80000",
            table_name="employees",
            columns=["id", "name", "salary", "department"]
        )
    """

    # Supported base models for fine-tuning
    SUPPORTED_MODELS = {
        "t5-small": "google-t5/t5-small",
        "t5-base": "google-t5/t5-base",
        "t5-large": "google-t5/t5-large",
        "t5-3b": "google-t5/t5-3b",
        "bart-base": "facebook/bart-base",
        "bart-large": "facebook/bart-large",
        "flan-t5-base": "google/flan-t5-base",
        "flan-t5-large": "google/flan-t5-large",
        "codeT5-base": "Salesforce/codet5-base",
        "codeT5-large": "Salesforce/codet5-large",
    }

    def __init__(
        self,
        model_path: str = "t5-base",
        device: str = "auto",
        max_input_length: int = 512,
        max_output_length: int = 256,
        num_beams: int = 4,
        temperature: float = 1.0,
    ):
        """
        Initialize the Transformer engine.

        Args:
            model_path: Path to fine-tuned model or HuggingFace model name.
            device: "cpu", "cuda", or "auto" (auto-detect).
            max_input_length: Maximum input sequence length.
            max_output_length: Maximum generated SQL length.
            num_beams: Beam search width.
            temperature: Sampling temperature.
        """
        _import_dependencies()

        self.model_path = model_path
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams
        self.temperature = temperature
        self.post_processor = SQLPostProcessor()

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load model and tokenizer from path or HuggingFace hub."""
        resolved_path = self.SUPPORTED_MODELS.get(model_path, model_path)

        logger.info(f"Loading model from: {resolved_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(resolved_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully ({self._count_params()} parameters)")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _count_params(self) -> str:
        """Count model parameters in human-readable format."""
        n = sum(p.numel() for p in self.model.parameters())
        if n >= 1e9:
            return f"{n / 1e9:.1f}B"
        elif n >= 1e6:
            return f"{n / 1e6:.0f}M"
        return f"{n / 1e3:.0f}K"

    def _format_input(
        self,
        question: str,
        table_name: str,
        columns: List[str],
        column_types: Optional[List[str]] = None,
    ) -> str:
        """Format input with schema context."""
        cols_str = ", ".join(columns)
        if column_types:
            cols_str = ", ".join(f"{c} ({t})" for c, t in zip(columns, column_types))

        return f"translate to SQL: {question} | schema: table: {table_name} | columns: {cols_str}"

    def _compute_confidence(
        self,
        generated_ids: "torch.Tensor",
        scores: Optional[tuple] = None,
    ) -> float:
        """
        Compute confidence score from generation probabilities.

        Uses the mean log-probability of generated tokens, normalized
        by sequence length (length-normalized log-likelihood).
        """
        if scores is None:
            return 0.7  # default if scores not available

        try:
            log_probs = []
            for step_scores in scores:
                # step_scores shape: (batch_size * num_beams, vocab_size)
                probs = torch.softmax(step_scores, dim=-1)
                max_prob = probs.max(dim=-1).values[0]  # top beam
                log_probs.append(math.log(max_prob.item() + 1e-10))

            if not log_probs:
                return 0.7

            # Mean log probability → convert to 0-1 scale
            mean_log_prob = sum(log_probs) / len(log_probs)
            # log probs are negative; closer to 0 = more confident
            confidence = math.exp(mean_log_prob)
            return max(0.1, min(0.99, confidence))

        except Exception:
            return 0.7

    def generate_sql(
        self,
        question: str,
        table_name: str = "table",
        columns: Optional[List[str]] = None,
        column_types: Optional[List[str]] = None,
        return_multiple: bool = False,
        num_return: int = 3,
    ) -> Dict:
        """
        Generate SQL from a natural language question.

        Args:
            question: Natural language question.
            table_name: Target table name.
            columns: List of column names.
            column_types: Optional list of column types.
            return_multiple: If True, return top-N candidates.
            num_return: Number of candidates to return.

        Returns:
            Dict with keys: sql, confidence, explanation, candidates (optional)
        """
        if columns is None:
            columns = ["*"]

        # Format input
        input_text = self._format_input(question, table_name, columns, column_types)

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_output_length,
                num_beams=self.num_beams,
                num_return_sequences=num_return if return_multiple else 1,
                temperature=self.temperature,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
            )

        # Decode
        generated_ids = outputs.sequences
        scores = outputs.scores if hasattr(outputs, "scores") else None

        candidates = []
        for i in range(generated_ids.shape[0]):
            raw_sql = self.tokenizer.decode(
                generated_ids[i], skip_special_tokens=True
            )
            cleaned_sql = self.post_processor.clean(raw_sql)
            validation = self.post_processor.validate(cleaned_sql)
            candidates.append({
                "sql": cleaned_sql,
                "valid": all(validation.values()),
                "validation": validation,
            })

        # Pick best valid candidate
        best = None
        for c in candidates:
            if c["valid"]:
                best = c
                break
        if best is None:
            best = candidates[0] if candidates else {"sql": "", "valid": False}

        # Confidence
        confidence = self._compute_confidence(generated_ids, scores)
        if not best.get("valid", False):
            confidence *= 0.6  # penalize invalid SQL

        # Build explanation
        explanation_parts = [
            f"Model: <strong>{self.model_path}</strong>",
            f"Table: <strong>{table_name}</strong>",
        ]
        if columns and columns != ["*"]:
            explanation_parts.append(f"Columns: <strong>{', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}</strong>")

        result = {
            "sql": best["sql"],
            "confidence": round(confidence * 100),
            "explanation": " · ".join(explanation_parts),
            "schema": {
                "table": table_name,
                "columns": columns,
            },
            "valid": best.get("valid", False),
            "engine": "transformer",
        }

        if return_multiple:
            result["candidates"] = [
                {"sql": c["sql"], "valid": c["valid"]}
                for c in candidates
            ]

        return result

    def generate_batch(
        self,
        questions: List[str],
        table_name: str = "table",
        columns: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> List[Dict]:
        """
        Batch inference for multiple questions.

        Args:
            questions: List of natural language questions.
            table_name: Target table name (shared across batch).
            columns: Column names.
            batch_size: Processing batch size.

        Returns:
            List of result dicts.
        """
        if columns is None:
            columns = ["*"]

        results = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            input_texts = [
                self._format_input(q, table_name, columns) for q in batch
            ]

            inputs = self.tokenizer(
                input_texts,
                max_length=self.max_input_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_output_length,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )

            for j, gen_ids in enumerate(outputs):
                raw_sql = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                cleaned_sql = self.post_processor.clean(raw_sql)
                results.append({
                    "question": batch[j],
                    "sql": cleaned_sql,
                    "valid": all(self.post_processor.validate(cleaned_sql).values()),
                })

            logger.info(f"Processed batch {i // batch_size + 1} ({len(results)}/{len(questions)})")

        return results


# ─────────────────────────────────────────────
# Engine Factory
# ─────────────────────────────────────────────
class EngineFactory:
    """
    Factory for creating NL2SQL engines.
    Supports both rule-based and transformer-based engines
    with automatic fallback.
    """

    @staticmethod
    def create(
        engine_type: str = "auto",
        model_path: str = "t5-base",
        **kwargs,
    ):
        """
        Create an NL2SQL engine.

        Args:
            engine_type: "rule", "transformer", or "auto"
            model_path: Path to model (for transformer engine)

        Returns:
            Engine instance
        """
        if engine_type == "transformer":
            return TransformerNL2SQLEngine(model_path, **kwargs)

        if engine_type == "auto":
            try:
                _import_dependencies()
                if os.path.exists(model_path) or model_path in TransformerNL2SQLEngine.SUPPORTED_MODELS:
                    logger.info("Auto-detected: using Transformer engine")
                    return TransformerNL2SQLEngine(model_path, **kwargs)
            except ImportError:
                logger.info("PyTorch/Transformers not available, falling back to rule-based engine")
            except Exception as e:
                logger.warning(f"Failed to load Transformer engine: {e}. Using rule-based.")

        # Fall back to rule-based (imported from main app)
        logger.info("Using rule-based NL2SQL engine")
        from app import NL2SQLEngine
        return NL2SQLEngine()