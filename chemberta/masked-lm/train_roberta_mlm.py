""" Script for training a Roberta Masked-Language Model

Usage [SMILES tokenizer]:
    python train_roberta_mlm.py --dataset_path=<DATASET_PATH> --output_dir=<OUTPUT_DIR> --model_name=<MODEL_NAME> --tokenizer_type=smiles --tokenizer_path="seyonec/SMILES_tokenized_PubChem_shard00_160k"
    
Usage [BPE tokenizer]:
    python train_roberta_mlm.py --dataset_path=<DATASET_PATH> --output_dir=<OUTPUT_DIR> --model_name=<MODEL_NAME> --tokenizer_type=bpe
"""
import os
from absl import app
from absl import flags

import transformers

import torch

import wandb
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM

from chemberta.utils.raw_text_dataset import RawTextDataset 

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from tokenizers import ByteLevelBPETokenizer

FLAGS = flags.FLAGS

# RobertaConfig params
flags.DEFINE_integer(name="vocab_size", default=52_000, help="")
flags.DEFINE_integer(name="max_position_embeddings", default=512, help="")
flags.DEFINE_integer(name="num_attention_heads", default=12, help="")
flags.DEFINE_integer(name="num_hidden_layers", default=6, help="")
flags.DEFINE_integer(name="type_vocab_size", default=1, help="")

# Tokenizer params
flags.DEFINE_enum(name="tokenizer_type", default="smiles", enum_values=["smiles", "bpe", "SMILES", "BPE"], help="")
flags.DEFINE_string(name="tokenizer_path", default="", help="")
flags.DEFINE_integer(name="BPE_min_frequency", default=2, help="")
flags.DEFINE_string(name="output_tokenizer_dir", default="tokenizer_dir", help="")
flags.DEFINE_integer(name="max_tokenizer_len", default=512, help="")
flags.DEFINE_integer(name="tokenizer_block_size", default=512, help="")


# Dataset params
flags.DEFINE_string(name="dataset_path", default="pubchem-10m.txt", help="")
flags.DEFINE_string(name="output_dir", default="PubChem_10M_SMILES_Tokenizer", help="")
flags.DEFINE_string(name="model_name", default="PubChem_10M_SMILES_Tokenizer", help="")

# MLM params
flags.DEFINE_float(name="mlm_probability", default=0.15, lower_bound=0.0, upper_bound=1.0, help="")

# Train params
flags.DEFINE_boolean(name="overwrite_output_dir", default=True, help="")
flags.DEFINE_integer(name="num_train_epochs", default=10, help="")
flags.DEFINE_integer(name="per_device_train_batch_size", default=64, help="")
flags.DEFINE_integer(name="save_steps", default=10_000, help="")
flags.DEFINE_integer(name="save_total_limit", default=2, help="")


def main(argv):
    wandb.login()

    is_gpu = torch.cuda.is_available()

    config = RobertaConfig(
        vocab_size=FLAGS.vocab_size,
        max_position_embeddings=FLAGS.max_position_embeddings,
        num_attention_heads=FLAGS.num_attention_heads,
        num_hidden_layers=FLAGS.num_hidden_layers,
        type_vocab_size=FLAGS.type_vocab_size,
    )
    
    if FLAGS.tokenizer_path:
        tokenizer_path = FLAGS.tokenizer_path
    elif FLAGS.tokenizer_type.upper() == "BPE":
        tokenizer_path = FLAGS.output_tokenizer_dir
        if not os.path.isdir(tokenizer_path):
            os.makedirs(tokenizer_path)
        
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=FLAGS.dataset_path, vocab_size=FLAGS.vocab_size, min_frequency=FLAGS.BPE_min_frequency, special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"])
        tokenizer.save_model(tokenizer_path)
    else:
        print("Please provide a tokenizer path if using the SMILES tokenizer")

    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=FLAGS.max_tokenizer_len)

    model = RobertaForMaskedLM(config=config)
    model.num_parameters()

    dataset = RawTextDataset(tokenizer=tokenizer, file_path=FLAGS.dataset_path, block_size=FLAGS.tokenizer_block_size)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=FLAGS.mlm_probability
    )

    training_args = TrainingArguments(
        output_dir=FLAGS.output_dir,
        overwrite_output_dir=FLAGS.overwrite_output_dir,
        num_train_epochs=FLAGS.num_train_epochs,
        per_device_train_batch_size=FLAGS.per_device_train_batch_size,
        save_steps=FLAGS.save_steps,
        save_total_limit=FLAGS.save_total_limit,
        fp16 = is_gpu, # fp16 only works on CUDA devices
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(FLAGS.model_name)

if __name__ == '__main__':
    app.run(main)