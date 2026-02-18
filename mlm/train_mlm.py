import os
from pathlib import Path
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer

MODEL_NAME = "roberta-base"
DATASET_NAME = "ioverho/misinfo-general"

BASE_OUT = Path("./models/mlm")
os.makedirs(BASE_OUT, exist_ok=True)
MLM_OUTPUT = str(BASE_OUT / "roberta-misinfo-mlm")

MAX_LEN = 256
MLM_PROB = 0.15
TEST_SIZE = 0.01
BATCH_SIZE = 32
NUM_EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
USE_FP16 = True

HF_CACHE = './cache'

def main():
    #load all year splits
    print("loading misinfo-general from huggingface")
    if HF_CACHE:
        raw = load_dataset(DATASET_NAME, cache_dir=HF_CACHE)
    else:
        raw = load_dataset(DATASET_NAME)

    print("available splits and sizes:")
    for k, ds in raw.items():
        print(f"   {k}: {len(ds)} examples")

    year_keys = sorted(raw.keys())
    year_dsets = [raw[k] for k in year_keys]
    misinfo = concatenate_datasets(year_dsets)

    print("\nCombined dataset size:", len(misinfo))
    print("Columns: ", misinfo.column_names)

    # build canonical text field with special tokens
    def clean(example):
        """
        Build a single 'text' field for MLM
        Treat title as the claim and content as the article
        Wrap them in <CLAIM> and <ARTICLE> markers.
        """
        title = example.get("title") or ""
        content = example.get('content') or ""

        title = title.strip()
        content = content.strip()

        if title and content:
            txt = f"<CLAIM> {title} </CLAIM> <ARTICLE> {content} </ARTICLE>"
        else:
            # if only one side is present, wrap it as CLAIM
            base  = (title or content).strip()
            txt = f"<CLAIM> {base} </CLAIM>"
        return {'text':txt.strip()}

    print("\nBuilding canonical text field with <CLAIM>/<ARTICLE> tokens ...")
    misinfo = misinfo.map(
        clean, desc = 'Constructing text', keep_in_memory = False
    )

    print("filtering very short texts (len <= 10)...")
    misinfo = misinfo.filter(
        lambda x: len(x['text']) > 10,
        desc = 'Filtering'
    )

    print("After cleaning and filtering:", len(misinfo), "examples")

    # train /val split

    print("\nsplitting into train and validation...")
    splits = misinfo.train_test_split(test_size=TEST_SIZE, seed=42)
    train_ds = splits['train']
    val_ds = splits['test']

    print("train size:", len(train_ds))
    print("val size:", len(val_ds))

    # tokenizer and model
    print("\nLoading tokenizer and model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_tokens = ['<CLAIM>','</CLAIM>','<ARTICLE>','</ARTICLE>']
    tokenizer.add_special_tokens({"additional_special_tokens":special_tokens})

    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    # tokenization
    def tokenize_mlm(batch):
        return tokenizer(
            batch['text'],
            truncation=True,
            padding="max_length",
            max_length= MAX_LEN
        )

    print("\nTokenizing train set")
    train_tok = train_ds.map(tokenize_mlm,batched=True,remove_columns=train_ds.column_names, desc='Tokenizing train',keep_in_memory=False)


    print("\nTokenizing val set")
    val_tok = val_ds.map(tokenize_mlm, batched=True, remove_columns=val_ds.column_names, desc='Tokenizing val',keep_in_memory=False)

    train_tok.set_format(type='torch')
    val_tok.set_format(type='torch')

    print("\nTokenized train features:", train_tok.column_names)
    print("Tokenized val features:", val_tok.column_names)

    # data collector for mlm
    data_collector = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROB
    )

    # training args

    print("\nSetting up training args...")
    training_args = TrainingArguments(
        output_dir=MLM_OUTPUT,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        fp16=USE_FP16,
        logging_steps=500,
        eval_strategy='steps',
        eval_steps=2000,
        save_steps=2000,
        save_total_limit=2,
        report_to="none"
    )

    # trainer

    trainer = Trainer(
        model=model,
        args = training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collector
    )

    num_train_examples = len(train_tok)
    num_eval_examples = len(val_tok)
    approx_train_tokens = num_train_examples * MAX_LEN

    print("\nTraining statistics (approx):")
    print("  Train examples:", num_train_examples)
    print("  Val examples:", num_eval_examples)
    print("  Max sequence length:", MAX_LEN)
    print("  Approx train tokens seen per epoch:", approx_train_tokens)

    # train and evaluate
    print("\nstarting MLM training... \n")
    trainer.train()

    print("\nEvaluating on validation set...")
    eval_metrics = trainer.evaluate()
    print("\nValidation metrics:")
    for k,v in eval_metrics.items():
        print(f"  {k}: {v}")

    print("\nLast 5 log entries from training history:")
    for log in trainer.state.log_history[-5:]:
        print(log)

    # save model and tokenizer
    print(f"\nSaving model and tokenizer to {MLM_OUTPUT} ...")
    trainer.save_model(MLM_OUTPUT)
    tokenizer.save_pretrained(MLM_OUTPUT)

    print("\nDone.")
    print("Use this checkpoint as the encoder backbone for your +MLM classifier, for example:")
    print('  backbone_name = "./models/roberta-misinfo-mlm"')

if __name__ == "__main__":
    main()



