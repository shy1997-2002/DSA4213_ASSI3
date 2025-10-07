from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_preprocess(task: str, model_name: str, max_length: int = 256):
    if task == "imdb":
        dataset = load_dataset("imdb")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def preprocess(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

        dataset = dataset.map(preprocess, batched=True)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset, tokenizer
    else:
        raise ValueError(f"Unsupported task: {task}")
