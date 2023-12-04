from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AutoTokenizer,
    default_data_collator,
    Seq2SeqTrainer,
)
import evaluate
import logging
import argparse

from trocr_handwritten.utils.trocr_model import (
    load_and_process_data,
    set_model_params,
    set_training_args,
    compute_metrics,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a handwritten OCR model.")
    parser.add_argument("--PATH_CONFIG", type=str, help="Path to the config files")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to the models",
        default="agomberto/FrenchCensus-handwritten-texts",
    )
    parser.add_argument(
        "--processor",
        type=str,
        help="Path to the processor",
        default="microsoft/trocr-large-handwritten",
    )
    parser.add_argument(
        "--trocr_model",
        type=str,
        help="Path to the pre-trained trocr model",
        default="microsoft/trocr-large-handwritten",
    )

    args = parser.parse_args()

    logging.info("Loading model & processor...")

    processor = TrOCRProcessor.from_pretrained(args.processor)
    model = VisionEncoderDecoderModel.from_pretrained(args.trocr_model)
    tokenizer = AutoTokenizer.from_pretrained(args.trocr_model)

    logging.info("Loading data...")

    model = set_model_params(model, processor, args.PATH_CONFIG)
    training_args = set_training_args(args.PATH_CONFIG)

    train_dataset, eval_dataset, test_dataset = load_and_process_data(
        args.dataset, tokenizer, processor
    )

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics(cer_metric, wer_metric, tokenizer, processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    logging.info("Trainer set-up")

    result = trainer.train()
