# NL2SQL — Natural Language to SQL Generator

A Flask-based web application that converts natural language questions into SQL queries using intelligent pattern recognition and NLP techniques.

## Features

- **Natural Language Input** — Type questions in plain English
- **SQL Generation** — Automatic conversion to syntactically correct SQL
- **Syntax Highlighting** — Color-coded SQL output for readability
- **Confidence Scoring** — Shows how confident the engine is in the generated query
- **Schema Awareness** — Understands 4 sample tables: employees, products, orders, students
- **Copy to Clipboard** — One-click SQL copying
- **Example Queries** — 10 built-in examples to get started

## Supported Query Patterns

| Pattern | Example |
|---|---|
| Basic Select | "Show all employees in Engineering department" |
| Aggregation | "How many products are in each category?" |
| Filtering | "Find employees with salary greater than 80000" |
| Sorting | "Show top 5 products by price descending" |
| Group By | "What is the average salary by department?" |
| Count | "Count total orders with status Shipped" |
| Multi-filter | "Students with gpa above 3.5 in Computer Science" |
| Min/Max | "Find maximum price of products in Electronics" |

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python app.py

# 3. Open in browser
# http://localhost:5000
```

## Project Structure

```
nl2sql/
├── app.py                 # Flask app + NL2SQL engine
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── templates/
    └── index.html        # Frontend (HTML/CSS/JS)
```

## Architecture

```
User Input (Natural Language)
        │
        ▼
┌─────────────────────┐
│   Flask Web Server   │
│   (app.py)          │
├─────────────────────┤
│   NL2SQL Engine     │
│   ├─ Table Detection │
│   ├─ Column Detection│
│   ├─ Aggregate Func  │
│   ├─ WHERE Conditions│
│   ├─ GROUP BY        │
│   ├─ ORDER BY        │
│   └─ LIMIT           │
├─────────────────────┤
│   SQL Builder        │
│   (Query Assembly)   │
└─────────────────────┘
        │
        ▼
  Generated SQL Query
```

## Extending with Transformer Models

The current rule-based engine can be replaced with a fine-tuned Transformer model:

1. **Dataset**: Use WikiSQL or Spider dataset for training
2. **Model**: Fine-tune T5-base or BART on text-to-SQL pairs
3. **Integration**: Replace `NL2SQLEngine.generate_sql()` with model inference
4. **Example** using HuggingFace Transformers:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("your-finetuned-model")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def generate_sql(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)
- **Fonts**: Playfair Display, DM Sans, JetBrains Mono
- **NLP Engine**: Rule-based pattern matching (upgradeable to Transformer)
