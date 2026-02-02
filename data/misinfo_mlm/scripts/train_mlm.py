import os
import glob
from datasets import load_dataset, DatasetDict
from transformers import (
    RobertaTokenizer, RobertaForMaskedLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
import torch

# --- Configuration ---
# Path to the unified schema directory, relative to the script location
script_dir = os.path.dirname(__file__)
UNIFIED_SCHEMA_PATH = os.path.abspath(os.path.join(script_dir, "../../unified_schema/"))
OUTPUT_DIR = "../checkpoints/roberta-misinfo-mlm"
LOGS_DIR = "../logs"
MAX_LENGTH = 256
SPECIAL_TOKENS = ["<CLAIM>", "</CLAIM>", "<ARTICLE>", "</ARTICLE>"]

def main():
    """Main function to run the MLM pre-training."""
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training might be slow and bf16 may not be effective.")

    # 1. Load Datasets based on train/validation splits
    print("Identifying train and validation splits from filenames...")
    train_files = glob.glob(os.path.join(UNIFIED_SCHEMA_PATH, "**/*train*.parquet"), recursive=True)
    val_files = glob.glob(os.path.join(UNIFIED_SCHEMA_PATH, "**/*val*.parquet"), recursive=True)

    if not train_files:
        print(f"ERROR: No training Parquet files found in '{UNIFIED_SCHEMA_PATH}' matching '*train*.parquet'")
        return

    print(f"Found {len(train_files)} training files: {train_files}")
    train_dataset = load_dataset("parquet", data_files=train_files, split="train")
    print(f"Training dataset loaded with {len(train_dataset)} examples.")

    if not val_files:
        print(f"WARNING: No validation files found in '{UNIFIED_SCHEMA_PATH}'. Evaluation during training will be skipped.")
        eval_dataset = None
    else:
        print(f"Found {len(val_files)} validation files: {val_files}")
        eval_dataset = load_dataset("parquet", data_files=val_files, split="train")
        print(f"Validation dataset loaded with {len(eval_dataset)} examples.")

    # 2. Setup Tokenizer & Custom Tokens
    print("Initializing tokenizer and adding special tokens...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    tokenizer.add_tokens(SPECIAL_TOKENS)
    print(f"Tokenizer ready. New vocabulary size: {len(tokenizer)}")

    # 3. Preprocess and Tokenize Datasets
    def preprocess_function(examples):
        texts = []
        claims = examples.get('claim_text', [None] * len(next(iter(examples.values()))))
        articles = examples.get('article_text', [None] * len(next(iter(examples.values()))))
        num_examples = len(claims)

        for i in range(num_examples):
            claim_content = claims[i] if claims and claims[i] else ""
            article_content = articles[i] if articles and articles[i] else ""
            texts.append(f"<CLAIM> {claim_content} </CLAIM> <ARTICLE> {article_content} </ARTICLE>")
        
        return tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LENGTH)

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    print("Training dataset tokenized.")
    
    tokenized_eval_dataset = None
    if eval_dataset:
        tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)
        print("Validation dataset tokenized.")

    # 4. Model setup (Resize is CRITICAL here)
    print("Loading RoBERTa model and resizing token embeddings...")
    model = RobertaForMaskedLM.from_pretrained("roberta-base")
    model.resize_token_embeddings(len(tokenizer))
    print("Model ready and embeddings resized.")

    # 5. Data Collator for MLM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    print("Data collator for MLM ready.")

    # 6. Training Arguments
    print("Configuring TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        bf16=True,
        max_grad_norm=1.0,
        logging_dir=LOGS_DIR,
        logging_steps=100,
        report_to="tensorboard",
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
    )
    print("TrainingArguments configured.")

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )
    print("Trainer initialized.")

    # 8. Run Training
    print("Starting MLM pre-training...")
    trainer.train()
    print("Training complete.")

    # 9. Save Final Model and Tokenizer
    final_model_output_dir = f"{OUTPUT_DIR}-final"
    trainer.save_model(final_model_output_dir)
    tokenizer.save_pretrained(final_model_output_dir)
    print(f"Final model and tokenizer saved to '{final_model_output_dir}'")

if __name__ == "__main__":
    main()
