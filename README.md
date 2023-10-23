# Introduction

- Current baseline is using MSE loss
- Only tile model is working, layout model hasn't been converged yet
- Organizer's codebase: https://github.com/google-research-datasets/tpu_graphs

# Ideas
- Change to ranking loss
- Extract more information from `pb` data
- Better embedding/representation for each type of feature
- Better model architecture
- ** Layout: Segment the graphs into multiple parts: Our results agree with the results from the prior paper [10] that the quality of the model improves significantly with the Graph Segment Training method (Best) over a typical full graph training (Full Graph), as GST potentially introduces a better hierarchical graph pooling mechanism that leads to better generalization
- Tile: results show that combining configuration features with node features early (early-join) is superior than combining configuration features with a reduced graph embedding later (late-join).
- more ideas from baseline paper: https://arxiv.org/pdf/2308.13490.pdf
- Graph Segment Training: https://github.com/kaidic/GST
# Training scripts

1. Tile
```
python train.py --do_train --do_eval --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 1e-4 --warmup_ratio 0.01 --lr_scheduler_type cosine --save_strategy epoch --evaluation_strategy epoch --logging_strategy steps --logging_steps 200 --save_total_limit 2 --load_best_model_at_end True --optim adamw_torch --weight_decay 1e-2 --num_train_epochs 20 --metric_for_best_model eval_score_tile_mean --greater_is_better=True --dataloader_num_workers=32 --max_grad_norm=1.0 --overwrite_output_dir=True --output_dir ./outputs/ --report_to none
```

2. Layout
- 2.1. layout-xla-random
```
python train.py --do_train --do_eval --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 1e-4 --warmup_ratio 0.01 --lr_scheduler_type cosine --save_strategy steps --save_steps 1000 --evaluation_strategy steps --eval_steps 1000 --logging_strategy steps --logging_steps 200 --save_total_limit 2 --load_best_model_at_end True --optim adamw_torch --weight_decay 1e-2 --num_train_epochs 1000 --metric_for_best_model eval_kendalltau --greater_is_better=True --dataloader_num_workers=8 --max_grad_norm=1.0 --data_type layout --source xla --search random --overwrite_output_dir=True --output_dir ./outputs/ --report_to none --load_best_model_at_end True --hidden_channels 32,64,64,84
```

# Some notes from kaggle discussion
1. https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/438074

Nice observation! A few operations (such as `while`, `call`, and `custom-call`) in an HLO graph are special in a sense that they introduce nested computations (graphs). You can find the semantics of HLO operations [here](https://www.tensorflow.org/xla/operation_semantics) (e.g. [while](https://www.tensorflow.org/xla/operation_semantics#while)).

The way we handle these nodes and the nested graphs is to conceptually flattening everything into one graph. For example, a [while](https://www.tensorflow.org/xla/operation_semantics#while) operation contains nested computations for condition, body, and init value. We handle a while node by connecting the while node with the root (output) nodes of the nested computation graphs representing condition, body, and init value. You can see this logic in our code that extracts edges [here](https://github.com/google-research-datasets/tpu_graphs/blob/main/tpu_graphs/process_data/xla/hlo_encoder.cc#L435).

1. Extracting additional features from protobuf

https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/445225

1. I forgot to add that the same layout config may appear multiple times because we use random search (for random collection) and genetic search (for default collection) to generate layout configs. Therefore, the same config maybe explored and measured multiple times. The runtimes from different runs won't be exactly the same due to noises (like when you run the same program on your laptop multiple times). However, the runtime variation should be <%1.*reply* 
    1. Idea: ranking but consider the duration differences between config_runtime

2. Additional resources: 
    1. https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/436629
    2. https://arxiv.org/pdf/2308.13490.pdf