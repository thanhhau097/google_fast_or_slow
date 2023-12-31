import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from data_args import DataArguments
from dataset import LayoutDataset, TileDataset, layout_collate_fn, tile_collate_fn
from engine import CustomTrainer, LayoutComputeMetricsFn, TileComputeMetricsFn
from model import LayoutModel, TileModel, LayoutModel
from model_args import ModelArguments

torch.set_float32_matmul_precision("high")
logger = logging.getLogger(__name__)


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
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        # transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    print("Loading dataset...")

    if data_args.data_type == "tile":
        dataset_cls = TileDataset
    else:
        dataset_cls = LayoutDataset

    train_dataset = dataset_cls(
        data_type=data_args.data_type,
        source=data_args.source,
        search=data_args.search,
        data_folder=data_args.data_folder,
        split="train",
        scaler=StandardScaler(),
        tgt_scaler=StandardScaler(),
        cfg_scaler=StandardScaler(),
        use_compressed=data_args.use_compressed,
        max_configs=data_args.max_configs,
        data_concatenation=data_args.data_concatenation,
        select_close_runtimes=data_args.select_close_runtimes,
        select_close_runtimes_prob=data_args.select_close_runtimes_prob,
        filter_random_configs=data_args.filter_random_configs,
    )
    val_dataset = dataset_cls(
        data_type=data_args.data_type,
        source=data_args.source,
        search=data_args.search,
        data_folder=data_args.data_folder,
        split="valid",
        use_compressed=data_args.use_compressed,
        max_configs=data_args.max_configs_eval,
        data_concatenation=data_args.data_concatenation,
    )
    val_dataset.scaler = train_dataset.scaler
    val_dataset.tgt_scaler = train_dataset.tgt_scaler

    if data_args.data_type == "layout":
        model_cls = LayoutModel
        collate_fn = layout_collate_fn
        compute_metrics = LayoutComputeMetricsFn(val_dataset.df)

    else:
        val_dataset.cfg_scaler = train_dataset.cfg_scaler
        model_cls = TileModel
        collate_fn = tile_collate_fn
        compute_metrics = TileComputeMetricsFn(val_dataset.df)

    model = model_cls(
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

    # Initialize trainer
    print("Initializing model...")

    if last_checkpoint is None and model_args.resume is not None:
        logger.info(f"Loading {model_args.resume} ...")
        checkpoint = torch.load(model_args.resume, "cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint.pop("state_dict")
        # checkpoint = {k[6:]: v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

        # if "fc.weight" in checkpoint:
        #     model.fc.load_state_dict(
        #         {"weight": checkpoint["fc.weight"], "bias": checkpoint["fc.bias"]}
        #     )

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
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Inference
    if training_args.do_predict:
        logger.info("*** Predict ***")
        # test_dataset = val_dataset
        test_dataset = dataset_cls(
            data_type=data_args.data_type,
            source=data_args.source,
            search=data_args.search,
            data_folder=data_args.data_folder,
            split="test",
            max_configs=data_args.max_configs_eval,  # note that for model with GraphNorm this matters A LOT. Higher the better
            use_compressed=data_args.use_compressed,
        )

        test_dataset.scaler = train_dataset.scaler
        test_dataset.tgt_scaler = train_dataset.tgt_scaler

        if data_args.data_type == "layout":
            trainer.compute_metrics = LayoutComputeMetricsFn(
                test_dataset.df, split="test"
            )
        else:
            test_dataset.cfg_scaler = train_dataset.cfg_scaler
            trainer.compute_metrics = TileComputeMetricsFn(
                test_dataset.df, split="test"
            )

        logits = trainer.predict(test_dataset).predictions

        predictions = []
        new_logits = []
        for e in logits:
            logit = np.array([x for x in e if x != -9999])
            new_logits.append(logit)
            predictions.append(np.argsort(logit)[:50])

        logits = new_logits

        prediction_files = []
        prediction_indices = []
        logits_indices = []
        runtime_indices = []
        if data_args.data_type == "tile":
            predictions = [pred[:5] for pred in predictions]

            for file_id, prediction, lg in zip(
                test_dataset.df["file"], predictions, logits
            ):
                # for file_id, prediction, rt, lg in zip(test_dataset.df["file"], predictions, test_dataset.df["config_runtime"], logits):
                prediction_files.append("tile:xla:" + file_id[:-4])
                prediction_indices.append(";".join([str(int(e)) for e in prediction]))
                logits_indices.append(lg.tolist())
                # runtime_indices.append(rt.tolist())
        else:
            for file_id, rows in test_dataset.df.groupby("file"):
                idx = rows.index.tolist()
                prediction = np.concatenate([predictions[i] for i in idx])
                prediction = np.argsort(prediction)
                prediction_files.append(
                    f"{data_args.data_type}:{data_args.source}:{data_args.search}:"
                    + file_id[:-4]
                )
                prediction_indices.append(";".join([str(int(e)) for e in prediction]))

        # dump to csv files
        submission_df = pd.DataFrame.from_dict(
            {
                "ID": prediction_files,
                "TopConfigs": prediction_indices,
                "logits": logits_indices,
                # "runtime": runtime_indices,
            }
        )
        if not os.path.exists("outputs_csv"):
            os.makedirs("outputs_csv")

        if data_args.data_type == "tile":
            submission_df.to_csv(
                os.path.join("outputs_csv", "tile:xla:submission.csv"), index=False
            )
        else:
            submission_df.to_csv(
                os.path.join(
                    "outputs_csv",
                    f"{data_args.data_type}:{data_args.source}:{data_args.search}:submission.csv",
                ),
                index=False,
            )


if __name__ == "__main__":
    main()
