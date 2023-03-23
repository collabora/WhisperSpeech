import torch
import evaluate
import argparse

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from huggingface_hub import login
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperProcessor


# add token to push results to huggingface hub
login(token="ADD YOUR HUGGING FACE TOKEN TO PUSH TO HUB")


def load_data(dataset_name, split, subset=''):
    dataset = DatasetDict()
    dataset["train"] = load_dataset(
        dataset_name, subset, split=split["train"], use_auth_token=True)
    dataset["test"] = load_dataset(
        dataset_name, subset, split=split["eval"], use_auth_token=True)
        
    # libri = libri.cast_column("audio", Audio(sampling_rate=16000))
    dataset_sampling_rate = next(iter(dataset.values())).features["audio"].sampling_rate
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        dataset = dataset.cast_column(
            "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
        )
    return dataset


def prepare_dataset(batch):
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def get_length_of_dataset(dataset):
    """
    Utility method to get length of dataset in hours
    """
    length = 0
    for item in dataset:
        # length in seconds
        length += (len(item["audio"]["array"]) / item["audio"]["sampling_rate"])
    # length in hours
    return length//3600


def train(opt):
    # load dataset
    libri = load_data(
        opt.dataset_name,
        {"train": opt.train_name, "eval": opt.val_name},
        subset=opt.subset)

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-tiny.en", language="English", task="transcribe")

    libri = libri.map(
        prepare_dataset,
        remove_columns=libri.column_names["train"],
        num_proc=8)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").cuda()

    if opt.freeze_decoder:      # freeze decoder and retrain encoder from scratch
        for param in model.model.decoder.parameters():
            param.requires_grad = False
        model.model.decoder._requires_grad = False
        model.model.decoder.gradient_checkpointing = False
        # re-init encoder params
        model.model.encoder.init_weights()
    elif opt.freeze_encoder:   # freeze encoder and retrain decoder from scratch
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

        # re-init decoder params
        model.model.decoder.init_weights()
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-tiny-decoder-libriasr-clean-test",  # change to a repo name of your choice
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=opt.num_train_steps,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )


    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=libri["train"],
        eval_dataset=libri["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-name', 
        type=str, 
        default='librispeech_asr', 
        help="Name of the dataset on huggingface hub.")

    parser.add_argument(
        '--subset',
        type=str, 
        default='clean', 
        help='Name of the subset of the dataset.')

    parser.add_argument(
        '--train_name',
        type=str, 
        default='train.360', 
        help='Name of the training subset of the dataset.')
    
    parser.add_argument(
        '--val_name',
        type=str, 
        default='validation', 
        help='Name of the validation subset of the dataset.')

    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help="The workers to load data.")

    parser.add_argument(
        "--num_train_steps",
        default=8000,
        type=int,
        help="Total number of training steps to perform.")

    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
        help="random seed for initialization")

    parser.add_argument(
        "--train_batch_size",
        default=16,
        type=int,
        help="Total batch size for training.")

    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        help="Freeze encoder and only train decoder.")
    
    parser.add_argument(
        '--freeze_decoder',
        action='store_true',
        help="Freeze decoder and only train encoder.")
    
    parser.add_argument(
        '--whisper_size',
        type=str,
        default="tiny",
        help="Whisper model size."
    )
    opt =  parser.parse_args()
    if opt.freeze_encoder and opt.freeze_decoder:
        raise ValueError("Cannot freeze both encoder and decoder.")
    # load whisper tiny en feature extractor, tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        f"openai/whisper-{opt.whisper_size}.en")
    tokenizer = WhisperTokenizer.from_pretrained(
        f"openai/whisper-{opt.whisper_size}.en", language="English", task="transcribe")
    # Word Error Rate metric
    metric = evaluate.load("wer")
    train(opt)
