export NEPTUNE_PROJECT="thanhhau097/lecr"
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMTRjM2ExOC1lYTA5LTQwODctODMxNi1jZjEzMjdlMjkxYTgifQ=="

for i in {1..32}
do
    python train_kfold.py --do_train=true --do_eval=true --do_predict=true --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --learning_rate=1e-3 --warmup_ratio=0.05 --lr_scheduler_type=cosine --save_strategy=steps --save_steps=12000 --eval_steps=12000 --evaluation_strategy=steps --logging_strategy=steps --logging_steps=200 --save_total_limit=2 --load_best_model_at_end=True --optim=adamw_torch --weight_decay=1e-5 --num_train_epochs=1000 --metric_for_best_model=eval_kendalltau --greater_is_better=True --dataloader_num_workers=8 --max_grad_norm=1.0 --data_type=layout --source=nlp --search=default --overwrite_output_dir=True --output_dir=./outputs_nlp_default_seed$i --report_to=neptune --load_best_model_at_end=True --hidden_channels=256,256 --fp16 --dropout=0.2 --gat_dropout=0.2 --op_embedding_dim=16 --max_configs=128 --select_close_runtimes=false --use_cross_attn=true --data_folder ./data/npz_all_pad/npz --use_compressed --seed $i
done