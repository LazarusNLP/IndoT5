from dataclasses import dataclass
from datargs import parse

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)


@dataclass
class Args:
    model_checkpoint: str = "LazarusNLP/IndoNanoT5-base"
    dataset_name: str = "LazarusNLP/indonlg"
    dataset_config: str = "xpersona"
    context_column_name: str = "context"
    input_column_name: str = "input"
    target_column_name: str = "target"
    use_persona: bool = False
    input_max_length: int = 512
    target_max_length: int = 512
    num_beams: int = 5
    output_dir: str = "outputs/indo-nanot5-xpersona"
    num_train_epochs: int = 50
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0
    optim: str = "adamw_torch_fused"
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    hub_model_id: str = "LazarusNLP/IndoNanoT5-base-XPersona"


def main(args: Args):
    # load dataset, tokenizer, model
    dataset = load_dataset(args.dataset_name, args.dataset_config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

    def preprocess_function(examples):
        def generate_history(history):
            # take turns between user and system
            history = [f"U: {text}" if i % 2 == 0 else f"S: {text}" for i, text in enumerate(history)]
            # prompt for next system response
            history.append("S: ")
            return history

        def generate_input(history, persona=None) -> str:
            # try removing conversations from the start until we are within max token length
            while history:
                combined_input = f"{persona} | {' | '.join(history)}" if persona else " | ".join(history)
                tokenized_input = tokenizer(combined_input, truncation=False)
                # return input once it is within sequence length
                if len(tokenized_input["input_ids"]) <= args.input_max_length:
                    return combined_input
                # otherwise, remove oldest chat history
                history.pop(0)

            raise NotImplementedError

        persona = [f"P: {' '.join(ex)}" if args.use_persona else None for ex in examples[args.context_column_name]]
        history = [generate_history(ex) for ex in examples[args.input_column_name]]
        inputs = [generate_input(h, p) for h, p in zip(history, persona)]
        targets = examples[args.target_column_name]

        model_inputs = tokenizer(inputs, max_length=args.input_max_length, truncation=False)
        labels = tokenizer(text_target=targets, max_length=args.target_max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=args.model_checkpoint, label_pad_token_id=tokenizer.pad_token_id
    )

    bleu = evaluate.load("bleu")
    sacrebleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)["bleu"]
        sacrebleu_score = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)["score"]

        return {"bleu": round(bleu_score * 100, 4), "sacrebleu": round(sacrebleu_score, 4)}

    callbacks = [EarlyStoppingCallback(args.early_stopping_patience, args.early_stopping_threshold)]

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        optim=args.optim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=5,
        predict_with_generate=True,
        metric_for_best_model="sacrebleu",
        load_best_model_at_end=True,
        bf16=True,
        report_to="tensorboard",
        push_to_hub=True,
        hub_model_id=args.hub_model_id,
        hub_private_repo=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    trainer.evaluate(tokenized_dataset["test"], max_length=args.target_max_length, num_beams=args.num_beams)

    trainer.push_to_hub()


if __name__ == "__main__":
    args = parse(Args)
    main(args)
