"""
NL2SQL Model Package
────────────────────
Transformer-based Text-to-SQL generation pipeline.

Components:
  • data_preprocessor  – Dataset loading (WikiSQL, Spider, Synthetic)
  • transformer_engine – T5/BART inference with SQL post-processing
  • train              – Fine-tuning pipeline with evaluation
"""

__version__ = "1.0.0"