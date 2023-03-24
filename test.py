from datasets import load_dataset, load_metric
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer, Wav2Vec2Config
import transformers
import re
import json
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torchaudio
import random
import torch.nn.utils.prune as prune
# from nn_pruning.nn_pruning.inference_model_patcher import optimize_model, InferenceModelPatcher, optimize_model_directly
from nn_pruning.nn_pruning.model_structure import struct_from_config
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
transformers.utils.move_cache()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

timit = load_dataset("timit_asr", data_dir="/mnt/nvm1/timit")
timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).upper()
    return batch

timit = timit.map(remove_special_characters)

# def extract_all_chars(batch):
#   all_text = " ".join(batch["text"])
#   vocab = list(set(all_text))
#   return {"vocab": [vocab], "all_text": [all_text]}

# vocabs = timit.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=timit.column_names["train"])
# vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
# vocab_dict = {v: k for k, v in enumerate(vocab_list)}

# vocab_dict["|"] = vocab_dict[" "]
# del vocab_dict[" "]
# vocab_dict["[UNK]"] = len(vocab_dict)
# vocab_dict["[PAD]"] = len(vocab_dict)
# with open('vocab.json', 'w') as vocab_file:
#     json.dump(vocab_dict, vocab_file)


# tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
# feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch
timit = timit.map(prepare_dataset, remove_columns=timit.column_names["train"], num_proc=4)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base", 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
)

model.freeze_feature_extractor()

def pattern_match(patterns, module_name):
    for pattern in patterns:
        if re.match(pattern, module_name):
            return True
    return False

model_structure = struct_from_config(model.config_class)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"pre params: {total_params}")

def optimize_model_directly(model):

    model_structure = struct_from_config(model.config_class)

    # Further prune
    params = {}
    for name, parameter in model.named_parameters():
        params[name] = parameter
        if name.endswith("weight"):
            if model_structure.is_ffn(name):
                pos = model_structure.get_position_ffn(name)
                if pos == 0:
                    output_mask = params[name].abs().sum(1) == 0
                    n0 = name
                else:
                    input_mask = params[name].abs().sum(0) == 0
                    with torch.no_grad():
                        params[n0][input_mask] = 0
                        params[name][:, output_mask] = 0
                    print(f"in sum: {output_mask.sum()}, out: {input_mask.sum()}")

    def pattern_match(module_name, patterns):
        for pattern in patterns:
            if re.match(pattern, module_name):
                return True
        return False
    def get_sparsity(w, dim, prune=True):
        if prune:
            l = (w != 0).sum(dim)
            nnz = (l != 0).sum()
            idx = l.nonzero(as_tuple=False).squeeze(-1)
            # print(idx.shape, idx.shape[0])
            # TEMPORARY : NON EMPTY MATRICE
            if idx.shape[0] == 0:
                idx = torch.tensor([0], device=idx.device)
                nnz = 1
            return 1.0 - (float(nnz) / l.numel()), idx
        else:
            return 0.0, None
    
    pattern_prefix = model_structure.PATTERN_PREFIX
    patterns = [(pattern_prefix + model_structure.LAYER_PATTERNS[pattern]).replace(".", "\\.") for pattern in model_structure.FFN_LAYERS]
    for k, v in model.named_modules():
        if pattern_match(k, patterns):
            c_sparsity, c = get_sparsity(v.weight, 0)
            r_sparsity, r = get_sparsity(v.weight, 1)
            print(f"{k}, sparsity = {c_sparsity:0.2f}, {r_sparsity:0.2f}")
            if r is not None:
                v.weight.data = v.weight[r, :]
                v.bias.data = v.bias[r]
            if c is not None:
                v.weight.data = v.weight[:, c]

    return model

pattern_prefix = model_structure.PATTERN_PREFIX
pattern_name = [(pattern_prefix + model_structure.LAYER_PATTERNS[pattern]).replace(".", "\\.") for pattern in model_structure.FFN_LAYERS]
for name, module in model.named_modules():
    if pattern_match(pattern_name, name):
        if hasattr(module, 'weight') and len(module.weight.shape)>1:
            pos = model_structure.get_position_ffn(name)
            prune.ln_structured(module, 'weight', 0.5, 1, pos)
            prune.remove(module, 'weight')

# optimize_model(model, "dense", False)
optimize_model_directly(model)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"post params: {total_params}")
print(f"has prune_heads: {hasattr(model, 'prune_heads')}")

training_args = TrainingArguments(
  output_dir="results",
  group_by_length=True,
  per_device_train_batch_size=32,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=False,
  gradient_checkpointing=True, 
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2
)



trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=timit["train"],
    eval_dataset=timit["test"],
    tokenizer=processor.feature_extractor,
)

trainer.train()
