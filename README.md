# NL2SQL — Natural Language to SQL Generator

A Flask-based web application that converts natural language questions into SQL queries. Features a dual-engine architecture: a fast rule-based engine for instant results, and an optional Transformer-based deep learning engine (T5/CodeT5/BART) for higher accuracy.

---

## Features

| Feature | Rule-Based | Transformer |
|---|---|---|
| Zero dependencies (beyond Flask) | ✅ | ❌ |
| Instant response (<10ms) | ✅ | ❌ (~200ms) |
| Schema-aware generation | ✅ | ✅ |
| Complex JOINs and subqueries | ❌ | ✅ |
| Handles unseen patterns | ❌ | ✅ |
| Confidence scoring | ✅ | ✅ (probability-based) |
| Beam search candidates | ❌ | ✅ |

---

## Quick Start

### Rule-Based Engine Only (No GPU Required)

```bash
pip install flask
python app.py
# Open http://localhost:5000
```

### Full Setup with Transformer Engine

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Train on synthetic data (quick demo, ~5 min on GPU)
python -m model.train --dataset synthetic --epochs 5 --model t5-small

# 3. Run with transformer engine
NL2SQL_MODEL_PATH=checkpoints/best_model python app.py
```

---

## Project Structure

```
nl2sql/
├── app.py                          # Flask app + rule-based engine
├── requirements.txt                # Dependencies
├── README.md
├── templates/
│   └── index.html                  # Web UI (dark terminal aesthetic)
├── model/
│   ├── __init__.py
│   ├── data_preprocessor.py        # Dataset loading & preprocessing
│   ├── transformer_engine.py       # T5/BART inference engine
│   └── train.py                    # Fine-tuning pipeline
├── data/                           # Datasets (download separately)
│   ├── wikisql/
│   └── spider/
└── checkpoints/                    # Saved models (after training)
    ├── best_model/
    └── training_history.json
```

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                     Flask Web App (port 5000)              │
├────────────────────────┬──────────────────────────────────┤
│  Rule-Based Engine     │  Transformer Engine               │
│                        │  (model/transformer_engine.py)    │
│  • Table detection     │  • T5 / CodeT5 / BART            │
│  • Column detection    │  • Beam search decoding           │
│  • Aggregations        │  • SQL post-processing            │
│  • WHERE / GROUP BY    │  • Confidence via probabilities   │
│  • ORDER BY / LIMIT    │  • Batch inference                │
│  (always available)    │  (requires trained model)         │
└────────────────────────┴──────────────────────────────────┘
                         │
                  ┌──────▼──────┐
                  │   Frontend   │
                  │ • NL Input   │
                  │ • Engine     │
                  │   Toggle     │
                  │ • SQL Output │
                  │ • Syntax HL  │
                  │ • Confidence │
                  └─────────────┘
```

---

## Training Pipeline

### Supported Datasets

| Dataset | Type | Size | Complexity |
|---|---|---|---|
| Synthetic | Auto-generated | Configurable | Simple queries |
| WikiSQL | Real | 80K+ pairs | Single-table |
| Spider | Real | 10K+ pairs | Multi-table, JOINs |
| Custom CSV | User-provided | Any | Any |

### Supported Base Models

| Model | Size | Recommended For |
|---|---|---|
| `t5-small` | 60M | Quick prototyping |
| `t5-base` | 220M | Good balance |
| `t5-large` | 770M | High accuracy |
| `flan-t5-base` | 250M | Instruction-tuned |
| `codeT5-base` | 220M | Code-specialized (recommended) |
| `codeT5-large` | 770M | Best accuracy (recommended) |
| `bart-base` | 140M | Alternative architecture |

### Training Commands

```bash
# Quick demo (synthetic data, CPU ok)
python -m model.train \
    --dataset synthetic \
    --n_synthetic 2000 \
    --model t5-small \
    --epochs 5 \
    --batch_size 16

# WikiSQL training (GPU recommended)
# First download: https://github.com/salesforce/WikiSQL
python -m model.train \
    --dataset wikisql \
    --data_dir data/ \
    --model codeT5-base \
    --epochs 10 \
    --batch_size 8 \
    --lr 3e-4 \
    --fp16 \
    --augment

# Spider training (GPU required)
# First download: https://yale-lily.github.io/spider
python -m model.train \
    --dataset spider \
    --data_dir data/ \
    --model codeT5-base \
    --epochs 15 \
    --batch_size 4 \
    --lr 1e-4 \
    --grad_accum 4 \
    --fp16

# Custom dataset
python -m model.train \
    --dataset custom \
    --csv_path data/my_training_data.csv \
    --model t5-base \
    --epochs 10

# Resume from checkpoint
python -m model.train \
    --resume checkpoints/checkpoint_epoch_5 \
    --dataset spider \
    --epochs 5
```

### Custom CSV Format

```csv
question,sql,table_name,columns
"Show all employees","SELECT * FROM employees","employees","id,name,age,salary,department"
"Count products by category","SELECT category, COUNT(*) FROM products GROUP BY category","products","id,name,category,price"
```

### Data Preprocessing

```bash
# Generate synthetic training data
python -m model.data_preprocessor \
    --dataset synthetic \
    --n_synthetic 5000 \
    --augment \
    --output data/processed_train.json

# Preprocess WikiSQL
python -m model.data_preprocessor \
    --dataset wikisql \
    --data_dir data/ \
    --output data/wikisql_processed.json
```

### Input Format (What the Model Sees)

```
Input:  "translate to SQL: Show employees with salary above 80000 | schema: table: employees | columns: id, name, age, salary, department"
Target: "SELECT * FROM employees WHERE salary > 80000"
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Exact Match | Predicted SQL == Reference SQL (normalized) |
| Component Match | Per-clause accuracy (SELECT, WHERE, GROUP BY, ORDER BY) |
| Training Loss | Cross-entropy loss on token prediction |
| Validation Loss | Loss on held-out validation set |

Expected results (approximate):

| Model | WikiSQL EM | Spider EM |
|---|---|---|
| t5-small | ~75% | ~40% |
| t5-base | ~82% | ~55% |
| codeT5-base | ~85% | ~60% |
| codeT5-large | ~88% | ~65% |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web interface |
| `/generate` | POST | Generate SQL from natural language |
| `/examples` | GET | Get example queries |
| `/status` | GET | Engine availability status |

### POST /generate

```json
// Request
{
    "question": "Show employees with salary above 80000",
    "engine": "rule"
}

// Response
{
    "sql": "SELECT * FROM employees WHERE salary > 80000;",
    "confidence": 80,
    "explanation": "Detected table: employees · Conditions: salary > 80000",
    "engine": "rule",
    "schema": {
        "table": "employees",
        "columns": ["id", "name", "age", "department", "salary"]
    }
}
```

---

## Deployment

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NL2SQL_MODEL_PATH` | `checkpoints/best_model` | Path to trained model |

### Production (Gunicorn)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## Tech Stack

- **Backend**: Python 3.9+, Flask 3.0
- **ML**: PyTorch 2.0+, HuggingFace Transformers
- **Models**: T5, CodeT5, BART, Flan-T5
- **Frontend**: HTML5, CSS3, Vanilla JS
- **Datasets**: WikiSQL, Spider, Synthetic

## References

- WikiSQL — Zhong et al., 2017
- Spider — Yu et al., 2018
- T5: Exploring Transfer Learning — Raffel et al., 2020
- CodeT5 — Wang et al., 2021