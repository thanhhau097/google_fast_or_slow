# !/bin/bash

python train.py --do_train=false --do_eval=true --do_predict=true --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --learning_rate=1e-3 --warmup_ratio=0.05 --lr_scheduler_type=cosine --save_strategy=steps --save_steps=500 --eval_steps=500 --evaluation_strategy=steps --logging_strategy=steps --logging_steps=50 --save_total_limit=10 --load_best_model_at_end=True --optim=adamw_torch --weight_decay=1e-5 --max_steps=10000 --metric_for_best_model=eval_kendalltau --greater_is_better=True --dataloader_num_workers=8 --max_grad_norm=1.0 --data_type=tile --source=xla --search=default --output_dir=./output_csv --report_to=none --load_best_model_at_end=True --hidden_channels=256,256 --bf16 --dropout=0.2 --gat_dropout=0.2 --op_embedding_dim=16 --max_configs=96 --max_configs_eval=256 --select_close_runtimes=false --use_cross_attn=true --seed 101 --resume=./tile_models/tile_models_new/outputs_tile_1/pytorch_model.bin
mv ./outputs_csv/tile:xla:submission.csv ./outputs_csv/tile:xla:submission_1.csv

python train.py --do_train=false --do_eval=true --do_predict=true --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --learning_rate=1e-3 --warmup_ratio=0.05 --lr_scheduler_type=cosine --save_strategy=steps --save_steps=500 --eval_steps=500 --evaluation_strategy=steps --logging_strategy=steps --logging_steps=50 --save_total_limit=10 --load_best_model_at_end=True --optim=adamw_torch --weight_decay=1e-5 --max_steps=10000 --metric_for_best_model=eval_kendalltau --greater_is_better=True --dataloader_num_workers=8 --max_grad_norm=1.0 --data_type=tile --source=xla --search=default --output_dir=./output_csv --report_to=none --load_best_model_at_end=True --hidden_channels=256,256 --bf16 --dropout=0.2 --gat_dropout=0.2 --op_embedding_dim=16 --max_configs=96 --max_configs_eval=256 --select_close_runtimes=false --use_cross_attn=true --seed 102 --resume=./tile_models/tile_models_new/outputs_tile_2/pytorch_model.bin
mv ./outputs_csv/tile:xla:submission.csv ./outputs_csv/tile:xla:submission_2.csv

python train.py --do_train=false --do_eval=true --do_predict=true --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --learning_rate=1e-3 --warmup_ratio=0.05 --lr_scheduler_type=cosine --save_strategy=steps --save_steps=500 --eval_steps=500 --evaluation_strategy=steps --logging_strategy=steps --logging_steps=50 --save_total_limit=10 --load_best_model_at_end=True --optim=adamw_torch --weight_decay=1e-5 --max_steps=10000 --metric_for_best_model=eval_kendalltau --greater_is_better=True --dataloader_num_workers=8 --max_grad_norm=1.0 --data_type=tile --source=xla --search=default --output_dir=./output_csv --report_to=none --load_best_model_at_end=True --hidden_channels=256,256 --bf16 --dropout=0.2 --gat_dropout=0.2 --op_embedding_dim=16 --max_configs=96 --max_configs_eval=256 --select_close_runtimes=false --use_cross_attn=true --seed 103 --resume=./tile_models/tile_models_new/outputs_tile_3/pytorch_model.bin
mv ./outputs_csv/tile:xla:submission.csv ./outputs_csv/tile:xla:submission_3.csv

python train.py --do_train=false --do_eval=true --do_predict=true --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --learning_rate=1e-3 --warmup_ratio=0.05 --lr_scheduler_type=cosine --save_strategy=steps --save_steps=500 --eval_steps=500 --evaluation_strategy=steps --logging_strategy=steps --logging_steps=50 --save_total_limit=10 --load_best_model_at_end=True --optim=adamw_torch --weight_decay=1e-5 --max_steps=10000 --metric_for_best_model=eval_kendalltau --greater_is_better=True --dataloader_num_workers=8 --max_grad_norm=1.0 --data_type=tile --source=xla --search=default --output_dir=./output_csv --report_to=none --load_best_model_at_end=True --hidden_channels=256,256 --bf16 --dropout=0.2 --gat_dropout=0.2 --op_embedding_dim=16 --max_configs=96 --max_configs_eval=256 --select_close_runtimes=false --use_cross_attn=true --seed 104 --resume=./tile_models/tile_models_new/outputs_tile_4/pytorch_model.bin
mv ./outputs_csv/tile:xla:submission.csv ./outputs_csv/tile:xla:submission_4.csv

python train.py --do_train=false --do_eval=true --do_predict=true --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --learning_rate=1e-3 --warmup_ratio=0.05 --lr_scheduler_type=cosine --save_strategy=steps --save_steps=500 --eval_steps=500 --evaluation_strategy=steps --logging_strategy=steps --logging_steps=50 --save_total_limit=10 --load_best_model_at_end=True --optim=adamw_torch --weight_decay=1e-5 --max_steps=10000 --metric_for_best_model=eval_kendalltau --greater_is_better=True --dataloader_num_workers=8 --max_grad_norm=1.0 --data_type=tile --source=xla --search=default --output_dir=./output_csv --report_to=none --load_best_model_at_end=True --hidden_channels=256,256 --bf16 --dropout=0.2 --gat_dropout=0.2 --op_embedding_dim=16 --max_configs=96 --max_configs_eval=256 --select_close_runtimes=false --use_cross_attn=true --seed 105 --resume=./tile_models/tile_models_new/outputs_tile_5/pytorch_model.bin
mv ./outputs_csv/tile:xla:submission.csv ./outputs_csv/tile:xla:submission_5.csv

python merge_preds_tile.py