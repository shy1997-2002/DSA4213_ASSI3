import argparse

import evaluate
import yaml
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from data_prep import load_and_preprocess


def compute_metrics(eval_pred):
    import numpy as np
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = evaluate.load("accuracy").compute(predictions=preds, references=labels)
    f1 = evaluate.load("f1").compute(predictions=preds, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset, tokenizer = load_and_preprocess(cfg["task"], cfg["model_name"], cfg["max_length"])
    method = cfg["method"].lower()

    # ====== 模型构造 ======
    if method == "full":
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"], num_labels=cfg["num_labels"]
        )

    elif method == "lora":
        base_model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"], num_labels=cfg["num_labels"]
        )
        lora_cfg = LoraConfig(
            r=cfg.get("lora_r", 8),
            lora_alpha=cfg.get("lora_alpha", 16),
            target_modules=cfg.get("target_modules", ["query", "value"]),
            lora_dropout=cfg.get("lora_dropout", 0.1),
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(base_model, lora_cfg)

    elif method == "bitfit":
        # BitFit：只训练偏置参数
        base_model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"], num_labels=cfg["num_labels"]
        )
        for name, param in base_model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        model = base_model

    else:
        raise ValueError(f"Unknown method: {cfg['method']}")

    # ====== 训练参数 ======
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        evaluation_strategy=cfg.get("evaluation_strategy", "epoch"),
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=int(cfg["eval_batch_size"]),
        num_train_epochs=int(cfg["num_train_epochs"]),
        logging_steps=int(cfg.get("logging_steps", 100)),
        save_total_limit=2,
        seed=int(cfg.get("seed", 42)),
        fp16=bool(cfg.get("fp16", False)),
    )

    # ====== Trainer ======
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"].shuffle(seed=cfg["seed"]).select(range(20000)),
        eval_dataset=dataset["test"].select(range(5000)),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ====== 训练 ======
    trainer.train()

    # ====== 保存模型 ======
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
