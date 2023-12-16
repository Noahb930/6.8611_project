# NLP Final Project

## Step 1
Setup the neccessary environment according to instructions at the original repo: https://github.com/seyonechithrananda/bert-loves-chemistry

## Step 2
Datsets were created using the code in `chemberta/data/dataset.ipynb`:

## Step 3: Pretrain Models
To pretrain a model with each dataset, run the following command from within `chemberta/data/train`:

```
python train_roberta_py \
    --model_type mlm \
    --dataset_path DATASET_PATH \
    --eval_path ../data/lipo.csv \
    --output_dir ../saved_models \
    --run_name RUN_NAME \
    --per_device_train_batch_size 32 \
    --num_hidden_layers 3 \
    --num_attention_heads 3 \
    --intermediate_size 1024 \
    --num_train_epochs 10
```


## Step 4: Finetune Models
Finetuning code is located at `Simple_Transformers_ChemBERTa.ipynb`


## Step 5: Attention Visualization

Attention visualization experiments are located at `chemberta/visualization/attention_experiments.ipynb`