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
from dataset import LayoutDataset, TileDataset, layout_collate_fn, tile_collate_fn, DatasetFactory
from engine import CustomTrainer, LayoutComputeMetricsFn, TileComputeMetricsFn
from model import LayoutModel, TileModel, LayoutModel
from model_args import ModelArguments
from tqdm.auto import tqdm

torch.set_float32_matmul_precision("high")
logger = logging.getLogger(__name__)

TTA = None


def _predict_single(data_args, trainer, test_dataset, padding_value: int = -9999):
    if data_args.data_type == "layout":
        trainer.compute_metrics = LayoutComputeMetricsFn(test_dataset.df, split="test")
    else:
        trainer.compute_metrics = TileComputeMetricsFn(test_dataset.df, split="test")

    predictions = trainer.predict(test_dataset).predictions

    new_predictions = []
    for e in predictions:
        # only get top 5
        new_predictions.append(np.array([x for x in e if x != padding_value]))

    predictions = new_predictions

    prediction_files = []
    predictions_probs = []

    if data_args.data_type == "tile":
        predictions = [pred[:5] for pred in predictions]

        for file_id, prediction in zip(test_dataset.df["file"], predictions):
            prediction_files.append("tile:xla:" + file_id[:-4])
            predictions_probs.append(prediction)
    else:
        for file_id, rows in test_dataset.df.groupby("file"):
            idx = rows.index.tolist()
            prediction = np.concatenate([predictions[i] for i in idx])
            predictions_probs.append(prediction)
            prediction_files.append(
                f"{data_args.data_type}:{data_args.source.replace('_pruned', '')}:{data_args.search}:"
                + file_id[:-4]
            )
    return prediction_files, predictions_probs


def predict(data_args, split, trainer, dataset_factory, tta=None):
    if tta is None:
        test_dataset = dataset_factory.get_dataset_for_inference(split)
        prediction_files, predictions_probs = _predict_single(data_args, trainer, test_dataset)
        prediction_indices = []
        for pred_prob in predictions_probs:
            prediction = np.argsort(pred_prob)
            prediction_indices.append(";".join([str(int(e)) for e in prediction]))
        return prediction_files, predictions_probs, prediction_indices

    all_probs = []
    for test_dataset, permutations in tqdm(
        dataset_factory.get_tta_dataset(split, nb_permutations=tta, seed=101), total=tta
    ):
        prediction_files, predictions_probs = _predict_single(data_args, trainer, test_dataset)
        # unshuffle predictions using the indexes
        new_predictions_probs = []
        for pred_file, pred_prob in zip(prediction_files, predictions_probs):
            idx = permutations[pred_file.split(":")[-1] + ".npz"]
            new_predictions_probs.append(pred_prob[np.argsort(idx)])
        all_probs.append(new_predictions_probs)
    # average predictions
    final_probs = []
    for i in range(len(prediction_files)):
        final_probs.append(np.stack([e[i] for e in all_probs], axis=0).mean(0))

    prediction_indices = []
    for pred_prob in final_probs:
        prediction = np.argsort(pred_prob)
        prediction_indices.append(";".join([str(int(e)) for e in prediction]))
    return prediction_files, final_probs, prediction_indices


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
    )
    train_dataset, val_dataset, test_dataset = dataset_factory.get_datasets()

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
        prediction_files, predictions_probs, _ = predict(
            data_args, "valid", trainer, dataset_factory, tta=TTA
        )
        metric_fn = LayoutComputeMetricsFn(dataset_factory.valid_df, split="valid")
        gts = [
            metric_fn.df.set_index("file").loc[file.split(":")[-1] + ".npz", "config_runtime"]
            for file in prediction_files
        ]
        metrics = metric_fn((predictions_probs, gts))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Inference
    if training_args.do_predict:
        logger.info("*** Predict ***")
        prediction_files, predictions_probs, prediction_indices = predict(
            data_args, "test", trainer, dataset_factory, tta=TTA
        )
        # save to numpy file
        save_dict = {
            "prediction_files": prediction_files,
            "predictions_probs": predictions_probs,
        }
        np.save(
            os.path.join(training_args.output_dir, "predictions.npy"),
            save_dict,
            allow_pickle=True,
        )

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
