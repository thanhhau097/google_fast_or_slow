# Google Fast or Slow

## Data preprocessing

We used the dataset below for training in the layout datasets, which is identical to the original dataset except changing the padding to `-1` in [this line](https://github.com/google-research-datasets/tpu_graphs/blob/main/tpu_graphs/process_data/xla/featurizers.h#L137):

https://www.kaggle.com/datasets/tomirol/layout-npz-padding

After downloading the dataset above and extracting it (`layout_npz` folder), we can run:

```
git clone https://github.com/thanhhau097/google_fast_or_slow
cd google_fast_or_slow/data

kaggle competitions download -c predict-ai-model-runtime
kaggle datasets download tomirol/layout-npz-padding

unzip predict-ai-model-runtime.zip
unzip layout-npz-padding.zip
rm *.zip

rm -rf ./npz_all/npz/layout
mv ./layout_npz ./npz_all/npz/layout

python data_compression.py
cd ../
```


Download and unzip the model weights to reproduce the solution

```
kaggle datasets download tomirol/tile-models-google-runtime
kaggle datasets download arc144/google-fast-slow-viet-br-connection-weights

unzip tile_models.zip
unzip google-fast-slow-viet-br-connection-weights.zip
```

## Inference
In order to predict with our models and generate the `submission.csv` file you can run `./predict.sh` after having downloaded and unzipped the model weights.


## Training:
For each type, we train 10+ models (except for tile, our best submission used only 5 models) with different seeds, then ensemble the predictions (see section 3) the result by `mean` aggregation.

- Tile XLA:
```
./train_tile.sh
```

- Layout:XLA:Default
```
python train_kfold.py --do_train=true --do_eval=true --do_predict=true --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --learning_rate=1e-3 --warmup_ratio=0.05 --lr_scheduler_type=cosine --save_strategy=steps --save_steps=2000 --eval_steps=2000 --evaluation_strategy=steps --logging_strategy=steps --logging_steps=200 --save_total_limit=2 --load_best_model_at_end=True --optim=adamw_torch --weight_decay=1e-5 --num_train_epochs=750 --metric_for_best_model=eval_kendalltau --greater_is_better=True --dataloader_num_workers=8 --max_grad_norm=1.0 --data_type=layout --source=xla --search=default --overwrite_output_dir=True --output_dir=./outputs_xla_default/ --report_to=none --load_best_model_at_end=True --hidden_channels=256,256 --bf16 --dropout=0.2 --gat_dropout=0.2 --op_embedding_dim=16 --max_configs=64 --max_configs_eval=128 --select_close_runtimes=false --use_cross_attn=true --use_compressed
```

- Layout:XLA:Random
```
python train_kfold.py --do_train=true --do_eval=true --do_predict=true --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --learning_rate=1e-3 --warmup_ratio=0.05 --lr_scheduler_type=cosine --save_strategy=steps --save_steps=2000 --eval_steps=2000 --evaluation_strategy=steps --logging_strategy=steps --logging_steps=200 --save_total_limit=2 --load_best_model_at_end=True --optim=adamw_torch --weight_decay=1e-5 --num_train_epochs=750 --metric_for_best_model=eval_kendalltau --greater_is_better=True --dataloader_num_workers=8 --max_grad_norm=1.0 --data_type=layout --source=xla --search=random --overwrite_output_dir=True --output_dir=./outputs_xla_random --report_to=none --load_best_model_at_end=True --hidden_channels=256,256 --fp16 --dropout=0.2 --gat_dropout=0.2 --op_embedding_dim=16 --max_configs=128 --select_close_runtimes=false --use_cross_attn=false --use_compressed
```

- Layout:NLP:Default
```
python train_kfold.py --do_train=true --do_eval=true --do_predict=true --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --learning_rate=1e-3 --warmup_ratio=0.05 --lr_scheduler_type=cosine --save_strategy=steps --save_steps=6000 --eval_steps=6000 --evaluation_strategy=steps --logging_strategy=steps --logging_steps=200 --save_total_limit=2 --load_best_model_at_end=True --optim=adamw_torch --weight_decay=1e-5 --num_train_epochs=750 --metric_for_best_model=eval_kendalltau --greater_is_better=True --dataloader_num_workers=8 --max_grad_norm=1.0 --data_type=layout --source=nlp --search=default --overwrite_output_dir=True --output_dir=./outputs_nlp_default --report_to=none --load_best_model_at_end=True --hidden_channels=256,256 --bf16 --dropout=0.2 --gat_dropout=0.2 --op_embedding_dim=16 --max_configs=64 --max_configs_eval=128 --select_close_runtimes=false --use_cross_attn=true --use_compressed
```
- Layout:XLA:Random
```
python train_kfold.py --do_train=true --do_eval=true --do_predict=true --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --learning_rate=1e-3 --warmup_ratio=0.05 --lr_scheduler_type=cosine --save_strategy=steps --save_steps=6000 --eval_steps=6000 --evaluation_strategy=steps --logging_strategy=steps --logging_steps=200 --save_total_limit=2 --load_best_model_at_end=True --optim=adamw_torch --weight_decay=1e-5 --num_train_epochs=750 --metric_for_best_model=eval_kendalltau --greater_is_better=True --dataloader_num_workers=8 --max_grad_norm=1.0 --data_type=layout --source=nlp --search=random --overwrite_output_dir=True --output_dir=./outputs_nlp_random --report_to=none --load_best_model_at_end=True --hidden_channels=256,256 --fp16 --dropout=0.2 --gat_dropout=0.2 --op_embedding_dim=16 --max_configs=128 --select_close_runtimes=false --use_cross_attn=false --use_compressed
```




### Weights
1. Tile XLA: https://www.kaggle.com/datasets/tomirol/tile-models-google-runtime
2. Layout: https://www.kaggle.com/datasets/arc144/google-fast-slow-viet-br-connection-weights/data