"""
train.py
────────
Fine-tune a T5/CodeT5/BART model on Text-to-SQL data.

Usage:
    # Train on synthetic data (quick demo)
    python -m model.train --dataset synthetic --epochs 5

    # Train on WikiSQL
    python -m model.train --dataset wikisql --data_dir data/wikisql --epochs 10

    # Train on Spider
    python -m model.train --dataset spider --data_dir data/spider --epochs 15

    # Resume from checkpoint
    python -m model.train --resume checkpoints/best_model --epochs 5
"""

import os
import re
import json
import time
import logging
import argparse
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Lazy imports
# ─────────────────────────────────────────────
def import_training_deps():
    """Import heavy dependencies only when training."""
    global torch, Dataset, DataLoader
    global T5ForConditionalGeneration, T5Tokenizer
    global AutoModelForSeq2SeqLM, AutoTokenizer
    global AdamW, get_linear_schedule_with_warmup

    import torch as _torch
    from torch.utils.data import Dataset as _Dataset, DataLoader as _DataLoader
    from transformers import (
        T5ForConditionalGeneration as _T5,
        T5Tokenizer as _T5Tok,
        AutoModelForSeq2SeqLM as _Auto,
        AutoTokenizer as _AutoTok,
        AdamW as _AdamW,
        get_linear_schedule_with_warmup as _sched,
    )

    torch = _torch
    Dataset = _Dataset
    DataLoader = _DataLoader
    T5ForConditionalGeneration = _T5
    T5Tokenizer = _T5Tok
    AutoModelForSeq2SeqLM = _Auto
    AutoTokenizer = _AutoTok
    AdamW = _AdamW
    get_linear_schedule_with_warmup = _sched

    return torch


# ─────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────
class NL2SQLDataset:
    """PyTorch Dataset for NL2SQL training."""

    def __init__(self, data: List[Dict], tokenizer, max_input_len: int = 512, max_target_len: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize input
        input_encoding = self.tokenizer(
            item["input"],
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            item["target"],
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()

        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
class NL2SQLTrainer:
    """
    Training pipeline for fine-tuning seq2seq models on NL2SQL.

    Features:
      • Mixed-precision training (FP16)
      • Gradient accumulation
      • Learning rate warmup + linear decay
      • Checkpoint saving (best + periodic)
      • Validation with exact-match accuracy
      • Training history logging
    """

    def __init__(
        self,
        model_name: str = "t5-base",
        learning_rate: float = 3e-4,
        batch_size: int = 8,
        max_input_len: int = 512,
        max_target_len: int = 256,
        gradient_accumulation_steps: int = 2,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        fp16: bool = True,
        checkpoint_dir: str = "checkpoints",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.fp16 = fp16
        self.checkpoint_dir = checkpoint_dir
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # FP16 only on CUDA
        if self.device.type != "cuda":
            self.fp16 = False

        logger.info(f"Device: {self.device} | FP16: {self.fp16}")

        # Load model & tokenizer
        self._load_model()

    def _load_model(self):
        """Load pretrained model and tokenizer."""
        from model.transformer_engine import TransformerNL2SQLEngine
        resolved = TransformerNL2SQLEngine.SUPPORTED_MODELS.get(
            self.model_name, self.model_name
        )

        logger.info(f"Loading base model: {resolved}")
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(resolved)
        self.model.to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Parameters: {n_params:,} total | {trainable:,} trainable")

    def _create_optimizer(self, num_training_steps: int):
        """Create optimizer with weight decay and warmup scheduler."""
        # Separate parameters that should/shouldn't have weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        param_groups = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(param_groups, lr=self.learning_rate)

        warmup_steps = int(num_training_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        logger.info(f"Optimizer: AdamW | LR: {self.learning_rate} | Warmup: {warmup_steps} steps")

    def train(
        self,
        train_data: List[Dict],
        val_data: Optional[List[Dict]] = None,
        epochs: int = 10,
        save_every: int = 2,
        eval_every: int = 1,
    ):
        """
        Fine-tune the model.

        Args:
            train_data: List of {"input": ..., "target": ...} dicts
            val_data: Optional validation data (same format)
            epochs: Number of training epochs
            save_every: Save checkpoint every N epochs
            eval_every: Run validation every N epochs
        """
        logger.info(f"Training on {len(train_data)} samples for {epochs} epochs")

        # Create datasets
        train_dataset = NL2SQLDataset(
            train_data, self.tokenizer, self.max_input_len, self.max_target_len
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        val_loader = None
        if val_data:
            val_dataset = NL2SQLDataset(
                val_data, self.tokenizer, self.max_input_len, self.max_target_len
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )

        # Setup optimizer
        total_steps = (len(train_loader) // self.gradient_accumulation_steps) * epochs
        self._create_optimizer(total_steps)

        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # Training loop
        best_val_loss = float("inf")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            step_count = 0
            start_time = time.time()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass (with optional FP16)
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.gradient_accumulation_steps

                # Backward pass
                scaler.scale(loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                step_count += 1

                # Progress logging
                if (step + 1) % max(1, len(train_loader) // 5) == 0:
                    avg_loss = epoch_loss / step_count
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"  Epoch {epoch}/{epochs} | Step {step + 1}/{len(train_loader)} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )

            # Epoch summary
            avg_train_loss = epoch_loss / max(step_count, 1)
            elapsed = time.time() - start_time
            self.history["train_loss"].append(avg_train_loss)

            logger.info(
                f"Epoch {epoch}/{epochs} complete | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Validation
            if val_loader and epoch % eval_every == 0:
                val_loss, val_acc = self._evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)

                logger.info(
                    f"  Validation | Loss: {val_loss:.4f} | "
                    f"Exact Match: {val_acc:.2%}"
                )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(os.path.join(self.checkpoint_dir, "best_model"))
                    logger.info("  ✓ New best model saved!")

            # Periodic checkpoints
            if epoch % save_every == 0:
                self._save_checkpoint(
                    os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}")
                )

        # Save final model
        self._save_checkpoint(os.path.join(self.checkpoint_dir, "final_model"))
        self._save_history()

        logger.info("Training complete!")
        return self.history

    def _evaluate(self, val_loader) -> tuple:
        """Run validation and compute loss + exact match accuracy."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Loss
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()

                # Generate predictions for exact match
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_target_len,
                    num_beams=4,
                    early_stopping=True,
                )

                # Decode
                preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

                # Decode labels (replace -100 with pad token)
                labels_clean = labels.clone()
                labels_clean[labels_clean == -100] = self.tokenizer.pad_token_id
                refs = self.tokenizer.batch_decode(labels_clean, skip_special_tokens=True)

                for pred, ref in zip(preds, refs):
                    pred_norm = pred.strip().lower().rstrip(";")
                    ref_norm = ref.strip().lower().rstrip(";")
                    if pred_norm == ref_norm:
                        total_correct += 1
                    total_samples += 1

        avg_loss = total_loss / max(len(val_loader), 1)
        accuracy = total_correct / max(total_samples, 1)

        self.model.train()
        return avg_loss, accuracy

    def _save_checkpoint(self, path: str):
        """Save model and tokenizer to directory."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Checkpoint saved to: {path}")

    def _save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to: {history_path}")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        logger.info(f"Loading checkpoint from: {path}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)


# ─────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────
class NL2SQLEvaluator:
    """
    Evaluation metrics for Text-to-SQL models.

    Metrics:
      • Exact Match Accuracy (string-level)
      • Component Match (SELECT, WHERE, GROUP BY, ORDER BY)
      • Execution Accuracy (requires database connection)
    """

    @staticmethod
    def exact_match(predicted: str, reference: str) -> bool:
        """Check if predicted SQL exactly matches reference (normalized)."""
        def normalize(sql):
            sql = sql.strip().lower().rstrip(";")
            sql = " ".join(sql.split())  # normalize whitespace
            return sql
        return normalize(predicted) == normalize(reference)

    @staticmethod
    def component_match(predicted: str, reference: str) -> Dict[str, bool]:
        """Check component-level match."""
        def extract_components(sql):
            sql = sql.upper().strip().rstrip(";")
            components = {}

            # SELECT
            sel_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql, re.DOTALL)
            components["select"] = sel_match.group(1).strip() if sel_match else ""

            # FROM
            from_match = re.search(r"FROM\s+(.*?)(?:\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT|$)", sql)
            components["from"] = from_match.group(1).strip() if from_match else ""

            # WHERE
            where_match = re.search(r"WHERE\s+(.*?)(?:\s+GROUP|\s+ORDER|\s+LIMIT|$)", sql)
            components["where"] = where_match.group(1).strip() if where_match else ""

            # GROUP BY
            group_match = re.search(r"GROUP\s+BY\s+(.*?)(?:\s+HAVING|\s+ORDER|\s+LIMIT|$)", sql)
            components["group_by"] = group_match.group(1).strip() if group_match else ""

            # ORDER BY
            order_match = re.search(r"ORDER\s+BY\s+(.*?)(?:\s+LIMIT|$)", sql)
            components["order_by"] = order_match.group(1).strip() if order_match else ""

            return components

        pred_comp = extract_components(predicted)
        ref_comp = extract_components(reference)

        return {
            key: pred_comp.get(key, "") == ref_comp.get(key, "")
            for key in ["select", "from", "where", "group_by", "order_by"]
        }

    @staticmethod
    def evaluate_batch(
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Evaluate a batch and return aggregate metrics."""
        n = len(predictions)
        exact_matches = sum(
            NL2SQLEvaluator.exact_match(p, r)
            for p, r in zip(predictions, references)
        )

        component_scores = {"select": 0, "from": 0, "where": 0, "group_by": 0, "order_by": 0}
        for p, r in zip(predictions, references):
            comp = NL2SQLEvaluator.component_match(p, r)
            for key, match in comp.items():
                if match:
                    component_scores[key] += 1

        return {
            "exact_match_accuracy": exact_matches / max(n, 1),
            "component_accuracy": {
                k: v / max(n, 1) for k, v in component_scores.items()
            },
            "total_samples": n,
            "exact_matches": exact_matches,
        }


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NL2SQL Model Training")

    # Data arguments
    parser.add_argument("--dataset", choices=["wikisql", "spider", "synthetic", "custom"],
                       default="synthetic", help="Training dataset")
    parser.add_argument("--data_dir", default="data/", help="Dataset directory")
    parser.add_argument("--csv_path", default=None, help="Path to custom CSV")
    parser.add_argument("--n_synthetic", type=int, default=2000, help="Synthetic sample count")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")

    # Model arguments
    parser.add_argument("--model", default="t5-small",
                       help="Base model (t5-small, t5-base, codeT5-base, etc.)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_input_len", type=int, default=512, help="Max input length")
    parser.add_argument("--max_target_len", type=int, default=256, help="Max target length")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Checkpoint directory")

    # Misc
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Import dependencies
    torch = import_training_deps()
    torch.manual_seed(args.seed)

    # ── Load Data ──
    from model.data_preprocessor import (
        WikiSQLPreprocessor, SpiderPreprocessor,
        SyntheticDataGenerator, CustomCSVPreprocessor,
        InputFormatter, DataAugmenter,
    )

    logger.info(f"Loading dataset: {args.dataset}")

    if args.dataset == "wikisql":
        preprocessor = WikiSQLPreprocessor(os.path.join(args.data_dir, "wikisql"))
        splits = preprocessor.load_all()
        train_samples = splits.get("train", [])
        val_samples = splits.get("dev", [])
    elif args.dataset == "spider":
        preprocessor = SpiderPreprocessor(os.path.join(args.data_dir, "spider"))
        splits = preprocessor.load_all()
        train_samples = splits.get("train", [])
        val_samples = splits.get("dev", [])
    elif args.dataset == "custom":
        preprocessor = CustomCSVPreprocessor()
        all_samples = preprocessor.load(args.csv_path)
        split_idx = int(len(all_samples) * (1 - args.val_split))
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
    else:  # synthetic
        all_samples = SyntheticDataGenerator.generate(args.n_synthetic)
        split_idx = int(len(all_samples) * (1 - args.val_split))
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]

    # Augment
    if args.augment:
        train_samples = DataAugmenter.augment_samples(train_samples)

    # Format for model
    train_data = InputFormatter.format_batch(train_samples)
    val_data = InputFormatter.format_batch(val_samples) if val_samples else None

    logger.info(f"Train: {len(train_data)} samples | Val: {len(val_data) if val_data else 0} samples")

    # ── Train ──
    trainer = NL2SQLTrainer(
        model_name=args.resume or args.model,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
        gradient_accumulation_steps=args.grad_accum,
        fp16=args.fp16,
        checkpoint_dir=args.checkpoint_dir,
    )

    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
    )

    # ── Evaluate ──
    if val_data:
        logger.info("Running final evaluation...")
        from model.transformer_engine import TransformerNL2SQLEngine

        engine = TransformerNL2SQLEngine(
            os.path.join(args.checkpoint_dir, "best_model")
        )

        predictions = []
        references = []
        for sample in val_data[:200]:  # evaluate first 200
            result = engine.generate_sql(
                question=sample["input"].split(": ", 1)[-1].split(" | schema")[0],
                table_name="table",
                columns=["*"],
            )
            predictions.append(result["sql"])
            references.append(sample["target"])

        metrics = NL2SQLEvaluator.evaluate_batch(predictions, references)
        logger.info(f"Final Metrics:")
        logger.info(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
        for comp, acc in metrics['component_accuracy'].items():
            logger.info(f"  {comp:>10s} accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()