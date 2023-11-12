import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import transformers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from data_args import DataArguments
from dataset import LayoutDataset, TileDataset, layout_collate_fn, tile_collate_fn, DatasetFactory
from engine import CustomTrainer, LayoutComputeMetricsFn, TileComputeMetricsFn
from model import LayoutModel, TileModel, LayoutModel
from model_args import ModelArguments
from train import predict
from safetensors import safe_open

torch.set_float32_matmul_precision("high")
logger = logging.getLogger(__name__)

NB_FOLDS = 7
TTA = None


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train"
                " from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        # transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)
    output_dir = training_args.output_dir
    # Load dataset
    print("Loading dataset...")

    dataset_factory = DatasetFactory(
        data_type=data_args.data_type,
        source=data_args.source,
        search=data_args.search,
        data_folder=data_args.data_folder,
        scaler=StandardScaler(),
        tgt_scaler=StandardScaler(),
        use_compressed=data_args.use_compressed,
        max_configs=data_args.max_configs,
        max_configs_eval=data_args.max_configs_eval,  # note that for models with cross attn this matters A LOT. Higher the better
        data_concatenation=data_args.data_concatenation,
        select_close_runtimes=data_args.select_close_runtimes,
        select_close_runtimes_prob=data_args.select_close_runtimes_prob,
        filter_random_configs=data_args.filter_random_configs,
        add_pseudo=data_args.add_pseudo,
        # kfold=NB_FOLDS,
    )

    all_predictions_probs = []
    all_val_probs = []
    for fold in range(NB_FOLDS):
        if not training_args.do_train and model_args.weights_folder:
            ckpt_path = (
                Path(model_args.weights_folder)
                / f"fold_{fold}"
                / "checkpoint"
                / "pytorch_model.bin"
                # / "model.safetensors"
            )
            if not ckpt_path.exists():
                print(f"Checkpoint {ckpt_path} not found. Skipping fold {fold}")
                continue
            last_checkpoint = str(ckpt_path)

        (prediction_files, predictions_prob), (val_files, val_probs, val_gts) = train_on_fold(
            output_dir,
            data_args,
            model_args,
            training_args,
            dataset_factory,
            fold,
            last_checkpoint,
        )
        all_predictions_probs.append(predictions_prob)
        all_val_probs.append(val_probs)

    # Get ensemble validation result
    avg_val_probs = []
    for i in range(len(val_files)):
        avg_val_probs.append(np.stack([x[i] for x in all_val_probs], axis=0).mean(0))
    metric_fn = LayoutComputeMetricsFn(dataset_factory.valid_df, split="valid")
    metrics = metric_fn((avg_val_probs, val_gts))
    print(f"Final ensemble metrics: {metrics}")

    prediction_indices = []
    avg_prediction_probs = []
    for i in range(len(prediction_files)):
        pred_probs = np.stack([x[i] for x in all_predictions_probs], axis=0).mean(0)
        prediction = np.argsort(pred_probs)
        prediction_indices.append(";".join([str(int(e)) for e in prediction]))
        avg_prediction_probs.append(pred_probs)

    # dump to csv files
    submission_df = pd.DataFrame.from_dict(
        {
            "ID": prediction_files,
            "TopConfigs": prediction_indices,
        }
    )
    if not os.path.exists("outputs_csv"):
        os.makedirs("outputs_csv")

    if data_args.data_type == "tile":
        submission_df.to_csv(os.path.join("outputs_csv", "tile:xla:submission.csv"), index=False)
    else:
        submission_df.to_csv(
            os.path.join(
                "outputs_csv",
                f"{data_args.data_type}:{data_args.source}:{data_args.search}:submission.csv",
            ),
            index=False,
        )

    # also save extra stuff
    pred_df = pd.DataFrame.from_dict(
        {
            "ID": prediction_files,
            "probs": avg_prediction_probs,
        }
    )
    pred_df["probs"] = pred_df["probs"].apply(lambda x: ";".join([str(e) for e in x]))
    pred_df.to_csv(
        os.path.join(
            "outputs_probs",
            f"{data_args.data_type}:{data_args.source}:{data_args.search}:pred_probs.csv",
        ),
    )
    val_df = pd.DataFrame.from_dict(
        {
            "ID": val_files,
            "probs": avg_val_probs,
            "gts": val_gts,
        }
    )
    val_df["probs"] = val_df["probs"].apply(lambda x: ";".join([str(e) for e in x]))
    val_df["gts"] = val_df["gts"].apply(lambda x: ";".join([str(e) for e in x]))
    val_df.to_csv(
        os.path.join(
            "outputs_probs",
            f"{data_args.data_type}:{data_args.source}:{data_args.search}:val_probs.csv",
        ),
    )


def train_on_fold(
    output_dir, data_args, model_args, training_args, dataset_factory, fold, last_checkpoint
):
    print(f"Starting fold {fold}")
    set_seed(training_args.seed + fold)
    training_args.output_dir = Path(output_dir) / f"fold_{fold}"
    training_args.output_dir.mkdir(exist_ok=True, parents=True)
    training_args.output_dir = str(training_args.output_dir)

    train_dataset, val_dataset, test_dataset = dataset_factory.get_datasets(
        # fold, output_dir=training_args.output_dir
    )
    if data_args.data_type == "tile":
        model = TileModel(
            hidden_channels=[int(x) for x in model_args.hidden_channels.split(",")],
            graph_in=model_args.graph_in,
            graph_out=model_args.graph_out,
            hidden_dim=model_args.hidden_dim,
            dropout=model_args.dropout,
        )
        collate_fn = tile_collate_fn
        compute_metrics = TileComputeMetricsFn(val_dataset.df)
    else:
        model = LayoutModel(
            hidden_channels=[int(x) for x in model_args.hidden_channels.split(",")],
            graph_in=model_args.graph_in,
            graph_out=model_args.graph_out,
            hidden_dim=model_args.hidden_dim,
            dropout=model_args.dropout,
            gat_droprate=model_args.gat_dropout,
            op_embedding_dim=model_args.op_embedding_dim,
            layout_embedding_dim=model_args.layout_embedding_dim,
            norm=model_args.norm,
            use_cross_attn=model_args.use_cross_attn,
        )
        collate_fn = layout_collate_fn
        compute_metrics = LayoutComputeMetricsFn(val_dataset.df)

    # Initialize trainer
    print("Initializing model...")

    if last_checkpoint is not None:
        logger.info(f"Loading {last_checkpoint} ...")
        checkpoint = torch.load(last_checkpoint, "cpu")
        # checkpoint = safe_open(last_checkpoint, "pt")
        # checkpoint = {k: checkpoint.get_tensor(k) for k in checkpoint.keys()}
        if "state_dict" in checkpoint:
            checkpoint = checkpoint.pop("state_dict")
        model.load_state_dict(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("Start training...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        data_type=data_args.data_type,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        val_files, val_probs, _ = predict(data_args, "valid", trainer, dataset_factory, tta=TTA)
        metric_fn = LayoutComputeMetricsFn(dataset_factory.valid_df, split="valid")
        val_gts = [
            metric_fn.df.set_index("file").loc[file.split(":")[-1] + ".npz", "config_runtime"]
            for file in val_files
        ]
        metrics = metric_fn((val_probs, val_gts))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Inference
    logger.info("*** Predict ***")
    prediction_files, predictions_probs, _ = predict(
        data_args, "test", trainer, dataset_factory, tta=TTA
    )
    return (prediction_files, predictions_probs), (val_files, val_probs, val_gts)


if __name__ == "__main__":
    main()
