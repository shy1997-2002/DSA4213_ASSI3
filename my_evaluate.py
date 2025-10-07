import yaml
import argparse
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import PeftModel, PeftConfig
import evaluate
from data_prep import load_and_preprocess
import json

def compute_metrics(eval_pred):
    import numpy as np
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = evaluate.load("accuracy").compute(predictions=preds, references=labels)
    f1 = evaluate.load("f1").compute(predictions=preds, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

def main(config_path: str):
    # ---- 读取配置 ----
    with open(config_path, "r" , encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- 加载数据 ----
    dataset, tokenizer = load_and_preprocess(cfg["task"], cfg["model_name"], cfg["max_length"])

    # ---- 加载模型 ----
    # 如果是 peft 模型（LoRA/Adapter/Prompt），需要用 PeftModel
    try:
        peft_cfg = PeftConfig.from_pretrained(cfg["output_dir"])
        base_model = AutoModelForSequenceClassification.from_pretrained(peft_cfg.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, cfg["output_dir"])
        print(f"Loaded PEFT model ({peft_cfg.peft_type}) from {cfg['output_dir']}")
    except Exception as e:
        print("Not a PEFT model, loading standard model.")
        model = AutoModelForSequenceClassification.from_pretrained(cfg["output_dir"])

    # ---- 评估配置 ----
    training_args = TrainingArguments(
        output_dir="logs/eval",
        per_device_eval_batch_size=cfg.get("eval_batch_size", 32),
    )

    # ---- 构建 Trainer 并评估 ----
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset["test"].select(range(min(5000, len(dataset["test"])))),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate()
    print("Evaluation results:", results)

    # ---- 保存结果到 JSON ----
    save_path = f"{cfg['output_dir']}/eval_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
