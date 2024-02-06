from dataclasses import dataclass
from datargs import parse

from datasets import load_dataset
from transformers import T5Config, AutoTokenizer

from t5_tokenizer_model import SentencePieceUnigramTokenizer


@dataclass
class Args:
    vocab_size: int = 32_000
    batch_length: int = 1000
    dataset_name: str = "uonlp/CulturaX"
    dataset_config: str = "id"
    dataset_split: str = "train"
    output_dir: str = "outputs/indonesian-t5-base/"
    base_model_config: str = "google/t5-v1_1-base"
    hf_repo_id: str = "LazarusNLP/IndoNanoT5-base"


def main(args: Args):
    # Initialize a dataset
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split, streaming=True)

    # Build an iterator over this dataset
    def batch_iterator():
        batch = []
        for example in dataset:
            batch.append(example["text"])
            if len(batch) == args.batch_length:
                yield batch
                batch = []
        if batch:  # yield last batch
            yield batch

    tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")
    tokenizer.train_from_iterator(
        iterator=batch_iterator(),
        vocab_size=args.vocab_size,
        show_progress=True,
    )
    tokenizer.save(f"{args.output_dir}/tokenizer.json")

    # Create HF T5 Tokenizer and push to HF Hub
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    tokenizer.push_to_hub(args.hf_repo_id)

    # Create model config based on T5v1.1 and push to HF Hub
    config = T5Config.from_pretrained(args.base_model_config)
    config.push_to_hub(args.hf_repo_id)


if __name__ == "__main__":
    args = parse(Args)
    main(args)
