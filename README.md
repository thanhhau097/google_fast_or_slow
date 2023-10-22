# Introduction

- Current baseline is using MSE loss
- Only tile model is working, layout model hasn't been converged yet

# Ideas
- Change to ranking loss
- Extract more information from `pb` data
- Better embedding/representation for each type of feature
- Better model architecture

# Training scripts

1. Tile
```
python train.py --do_train --do_eval --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 1e-4 --warmup_ratio 0.01 --lr_scheduler_type cosine --save_strategy epoch --evaluation_strategy epoch --logging_strategy steps --logging_steps 200 --save_total_limit 2 --load_best_model_at_end True --optim adamw_torch --weight_decay 1e-2 --num_train_epochs 20 --metric_for_best_model eval_score_tile_mean --greater_is_better=True --dataloader_num_workers=32 --max_grad_norm=1.0 --overwrite_output_dir=True --output_dir ./outputs/ --report_to none
```

2. Layout
- 2.1. layout-xla-random
```
python train.py --do_train --do_eval --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 1e-4 --warmup_ratio 0.01 --lr_scheduler_type cosine --save_strategy epoch --evaluation_strategy epoch --logging_strategy steps --logging_steps 200 --save_total_limit 2 --load_best_model_at_end True --optim adamw_torch --weight_decay 1e-2 --num_train_epochs 1000 --metric_for_best_model eval_kendalltau --greater_is_better=True --dataloader_num_workers=32 --max_grad_norm=1.0 --data_type layout --source xla --search random --overwrite_output_dir=True --output_dir ./outputs/ --report_to none --load_best_model_at_end True --hidden_channels 32,64,64,64,64,64,84
```