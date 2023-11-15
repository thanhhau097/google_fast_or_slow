from copy import deepcopy
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
        kfold=False,
    )

    NB_FOLDS = 9
    if data_args.fold == -1:
        training_folds = list(range(NB_FOLDS))
    else:
        training_folds = [data_args.fold]

    predictions_probs = []
    for fold in training_folds:
        if not training_args.do_train and training_args.output_dir: #model_args.weights_folder:
            ckpt_path = (Path(training_args.output_dir) / f"fold_{fold}" / "pytorch_model.bin")
            if not ckpt_path.exists():
                print(f"Checkpoint {ckpt_path} not found. Skipping fold {fold}")
                continue
            last_checkpoint = str(ckpt_path)

        prediction_files, predictions_prob = train_on_fold(
            output_dir,
            data_args,
            model_args,
            training_args,
            dataset_factory,
            fold,
            last_checkpoint,
        )
        predictions_probs.append(predictions_prob)

    prediction_indices = []
    nb_test_files = len(prediction_files)
    for i in range(nb_test_files):
        prediction = np.stack([x[i] for x in predictions_probs], axis=0).mean(0)
        prediction = np.argsort(prediction)
        prediction_indices.append(";".join([str(int(e)) for e in prediction]))

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


def train_on_fold(
    output_dir, data_args, model_args, training_args, dataset_factory, fold, last_checkpoint
):
    print(f"Starting fold {fold}")
    set_seed(training_args.seed)
    training_args.output_dir = Path(output_dir) / f"fold_{str(fold)}"
    training_args.output_dir.mkdir(exist_ok=True, parents=True)
    training_args.output_dir = str(training_args.output_dir)

    train_dataset, val_dataset, test_dataset = dataset_factory.get_datasets(
        fold, output_dir=training_args.output_dir
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

    def get_prediction_results(predictions, dataset):
        new_predictions = []
        for e in predictions:
            # only get top 5
            new_predictions.append(np.array([x for x in e if x != -100]))

        predictions = new_predictions

        prediction_files = []
        predictions_probs = []
        prediction_indices = []
        if data_args.data_type == "tile":
            predictions = [pred[:5] for pred in predictions]

            for file_id, prediction in zip(dataset.df["file"], predictions):
                prediction_files.append("tile:xla:" + file_id[:-4])
                prediction_indices.append(";".join([str(int(e)) for e in prediction]))
        else:
            for file_id, rows in dataset.df.groupby("file"):
                idx = rows.index.tolist()
                prediction = np.concatenate([predictions[i] for i in idx])
                predictions_probs.append(prediction)
                prediction = np.argsort(prediction)
                prediction_files.append(
                    f"{data_args.data_type}:{data_args.source}:{data_args.search}:" + file_id[:-4]
                )
                prediction_indices.append(";".join([str(int(e)) for e in prediction]))
        
        return prediction_files, predictions_probs

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        val_predictions = trainer.predict(val_dataset).predictions
        prediction_files, predictions_probs = get_prediction_results(val_predictions, val_dataset)
        # save to numpy file
        save_dict = {
            "prediction_files": prediction_files,
            "predictions_probs": predictions_probs,
        }
        np.save(
            os.path.join(training_args.output_dir, "val_predictions.npy"),
            save_dict,
            allow_pickle=True,
        )

    # Inference
    logger.info("*** Predict ***")
    if data_args.data_type == "layout":
        trainer.compute_metrics = LayoutComputeMetricsFn(test_dataset.df, split="test")
    else:
        trainer.compute_metrics = TileComputeMetricsFn(test_dataset.df, split="test")

    predictions = trainer.predict(test_dataset).predictions

    prediction_files, predictions_probs = get_prediction_results(predictions, test_dataset)
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

    # architecture fine-tuning
    if data_args.architecture_finetune:
        # save current best model then load it back
        best_model_path = os.path.join(training_args.output_dir, "pytorch_model.bin")
        # trainer.save_model(os.path.join(training_args.output_dir, str(fold)))
        for filename in dataset_factory.test_mapping.keys():
            if os.path.exists(os.path.join(training_args.output_dir, "finetuning", filename, "predictions.npy")):
                print(f"Skipping {filename} as it already exists")
                continue
            # create new datasets
            train_dataset, val_dataset, test_dataset = dataset_factory.get_datasets(
                fold, output_dir=training_args.output_dir,
                architecture_finetune=data_args.architecture_finetune,
                architecture_finetune_test_file_names=filename,
            )

            model.load_state_dict(torch.load(best_model_path))
            
            new_training_args = deepcopy(training_args)
            new_training_args.num_train_epochs = data_args.architecture_finetune_epochs
            new_training_args.eval_steps = data_args.architecture_finetune_eval_steps
            new_training_args.save_steps = data_args.architecture_finetune_eval_steps
            new_training_args.output_dir = os.path.join(training_args.output_dir, "finetuning", filename)
            new_training_args.do_train = True
            new_training_args.do_eval = True
            new_training_args.do_predict = True

            if data_args.data_type == "tile":
                compute_metrics = TileComputeMetricsFn(val_dataset.df)
            else:
                compute_metrics = LayoutComputeMetricsFn(val_dataset.df)

            new_trainer = CustomTrainer(
                model=model,
                args=new_training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=collate_fn,
                compute_metrics=compute_metrics,
                data_type=data_args.data_type,
            )

            # finetune with new dataset
            logger.info("*** Evaluate before Finetune ***")
            new_trainer.evaluate()
            logger.info("*** Finetune ***")
            train_result = new_trainer.train()
            metrics = train_result.metrics
            new_trainer.save_model()
            new_trainer.log_metrics("train", metrics)
            new_trainer.save_metrics("train", metrics)
            new_trainer.save_state()

            logger.info("*** Evaluate ***")
            metrics = new_trainer.evaluate()
            new_trainer.log_metrics("eval", metrics)
            new_trainer.save_metrics("eval", metrics)
            val_predictions = new_trainer.predict(val_dataset).predictions
            prediction_files, predictions_probs = get_prediction_results(val_predictions, val_dataset)
            # save to numpy file
            save_dict = {
                "prediction_files": prediction_files,
                "predictions_probs": predictions_probs,
            }
            np.save(
                os.path.join(new_training_args.output_dir, "val_predictions.npy"),
                save_dict,
                allow_pickle=True,
            )

            # export prediction probs then save to folder
            logger.info("*** Predict ***")
            if data_args.data_type == "tile":
                new_trainer.compute_metrics = TileComputeMetricsFn(test_dataset.df, split="test")
            else:
                new_trainer.compute_metrics = LayoutComputeMetricsFn(test_dataset.df, split="test")

            predictions = new_trainer.predict(test_dataset).predictions
            prediction_files, predictions_probs = get_prediction_results(predictions, test_dataset)
            
                # save to numpy file
            save_dict = {
                "prediction_files": prediction_files,
                "predictions_probs": predictions_probs,
            }
            np.save(
                os.path.join(new_training_args.output_dir, "predictions.npy"),
                save_dict,
                allow_pickle=True,
            )

    return prediction_files, predictions_probs


if __name__ == "__main__":
    main()
