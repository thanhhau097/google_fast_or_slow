import gc
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from loss import PairwiseHingeLoss, listMLE, PairwiseLogisticLoss
from scipy.stats import kendalltau
from torch import nn
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from torch.utils.data import DataLoader
from typing import Optional, List, Union
from packaging import version

from transformers.integrations.deepspeed import deepspeed_init

from transformers.trainer_utils import (
    EvalPrediction,
    has_length,
    EvalLoopOutput,
    denumpify_detensorize,
)
from transformers.trainer_pt_utils import (
    nested_concat,
    DistributedTensorGatherer,
    SequentialDistributedSampler,
    IterableDatasetShard,
    find_batch_size,
    nested_numpify,
)
from transformers.utils import logging, is_torch_tpu_available, is_accelerate_available

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

logger = logging.get_logger(__name__)


# https://pytorchltr.readthedocs.io/en/stable/loss.html
def pairwise_hinge_loss(y_pred, y_true):
    loss_fn = PairwiseHingeLoss()
    # loss_fn = PairwiseLogisticLoss(0.1)

    y_pred = y_pred.unsqueeze(0)
    y_true = y_true.unsqueeze(0)
    return loss_fn(
        y_pred, y_true, n=torch.tensor([y_pred.shape[1]], device=y_pred.device)
    ).mean()


class CustomTrainer(Trainer):
    def __init__(self, data_type="tile", **kwargs):
        super().__init__(**kwargs)
        self.data_type = data_type
        # self.loss_fn = nn.MSELoss()
        # self.loss_fn = listMLE
        self.loss_fn = pairwise_hinge_loss

    def compute_loss(self, model: any, inputs: Dict, return_outputs=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.data_type == "tile":
            outputs = model(
                inputs["config_feat"],
                inputs["node_feat"],
                inputs["node_layout_feat"],
                inputs["node_opcode"],
                inputs["edge_index"],
            )
        else:
            outputs = model(
                inputs["node_config_feat"],
                inputs["node_feat"],
                inputs["node_layout_feat"],
                inputs["node_opcode"],
                inputs["edge_index"],
                inputs["node_config_ids"],
            )
        loss = self.loss_fn(outputs, inputs["target"].to(device))

        if return_outputs:
            return (loss, outputs)
        return loss

    def create_optimizer(self):
        model = self.model
        no_decay = []
        for n, m in model.named_modules():
            if isinstance(
                m,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.LayerNorm,
                    torch.nn.LayerNorm,
                    torch.nn.GroupNorm,
                ),
            ):
                no_decay.append(n)

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        padding_value: int = -9999,
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        if not has_length(dataloader):
            raise ValueError("dataloader must implement a working __len__")

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Batch size = {batch_size}")
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
        inputs_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = max(1, args.world_size)

        eval_losses_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(
                dataloader.sampler, SequentialDistributedSampler
            ):
                make_multiple_of = dataloader.sampler.batch_size
            preds_gatherer = DistributedTensorGatherer(
                world_size, num_examples, make_multiple_of=make_multiple_of
            )
            labels_gatherer = DistributedTensorGatherer(
                world_size, num_examples, make_multiple_of=make_multiple_of
            )
            inputs_gatherer = DistributedTensorGatherer(
                world_size, num_examples, make_multiple_of=make_multiple_of
            )

        model.eval()

        if args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name])
                if args.include_inputs_for_metrics
                else None
            )

            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = (
                    losses
                    if losses_host is None
                    else torch.cat((losses_host, losses), dim=0)
                )
            if logits is not None:
                preds_host = (
                    logits
                    if preds_host is None
                    else nested_concat(preds_host, logits, padding_index=padding_value)
                )
            if labels is not None:
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=padding_value)
                )
            if inputs_decode is not None:
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(
                        inputs_host, inputs_decode, padding_index=padding_value
                    )
                )
            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
            ):
                eval_losses_gatherer.add_arrays(
                    self._gather_and_numpify(losses_host, "eval_losses")
                )
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(
                        self._gather_and_numpify(preds_host, "eval_preds")
                    )
                    labels_gatherer.add_arrays(
                        self._gather_and_numpify(labels_host, "eval_label_ids")
                    )
                    inputs_gatherer.add_arrays(
                        self._gather_and_numpify(inputs_host, "eval_inputs_ids")
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host, inputs_host = (
                    None,
                    None,
                    None,
                    None,
                )

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(
            self._gather_and_numpify(losses_host, "eval_losses")
        )
        if not prediction_loss_only:
            preds_gatherer.add_arrays(
                self._gather_and_numpify(preds_host, "eval_preds")
            )
            labels_gatherer.add_arrays(
                self._gather_and_numpify(labels_host, "eval_label_ids")
            )
            inputs_gatherer.add_arrays(
                self._gather_and_numpify(inputs_host, "eval_inputs_ids")
            )

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None
        inputs_ids = inputs_gatherer.finalize() if not prediction_loss_only else None

        if (
            self.compute_metrics is not None
            and preds is not None
            and label_ids is not None
        ):
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(
                        predictions=preds, label_ids=label_ids, inputs=inputs_ids
                    )
                )
            else:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=preds, label_ids=label_ids)
                )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=preds,
            label_ids=label_ids,
            metrics=metrics,
            num_samples=num_examples,
        )

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        padding_value: int = -9999,
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name])
                if args.include_inputs_for_metrics
                else None
            )

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = (
                    losses
                    if losses_host is None
                    else nested_concat(losses_host, losses, padding_index=padding_value)
                )
            if labels is not None:
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=padding_value
                )
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(
                    inputs_decode, dim=1, pad_index=padding_value
                )
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(
                        inputs_host, inputs_decode, padding_index=padding_value
                    )
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(
                    logits, dim=1, pad_index=padding_value
                )
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = (
                    logits
                    if preds_host is None
                    else nested_concat(preds_host, logits, padding_index=padding_value)
                )

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=padding_value)
                )

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
                and (
                    self.accelerator.sync_gradients
                    or version.parse(accelerate_version) > version.parse("0.20.3")
                )
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = (
                        losses
                        if all_losses is None
                        else np.concatenate((all_losses, losses), axis=0)
                    )
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = (
                        logits
                        if all_preds is None
                        else nested_concat(
                            all_preds, logits, padding_index=padding_value
                        )
                    )
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(
                            all_inputs, inputs_decode, padding_index=padding_value
                        )
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels
                        if all_labels is None
                        else nested_concat(
                            all_labels, labels, padding_index=padding_value
                        )
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = (
                    None,
                    None,
                    None,
                    None,
                )

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = (
                losses
                if all_losses is None
                else np.concatenate((all_losses, losses), axis=0)
            )
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = (
                logits
                if all_preds is None
                else nested_concat(all_preds, logits, padding_index=padding_value)
            )
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode
                if all_inputs is None
                else nested_concat(
                    all_inputs, inputs_decode, padding_index=padding_value
                )
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = (
                labels
                if all_labels is None
                else nested_concat(all_labels, labels, padding_index=padding_value)
            )

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif (
            isinstance(eval_dataset, IterableDatasetShard)
            and getattr(eval_dataset, "num_examples", 0) > 0
        ):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
        ):
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(
                        predictions=all_preds, label_ids=all_labels, inputs=all_inputs
                    )
                )
            else:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels)
                )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[
                f"{metric_key_prefix}_jit_compilation_time"
            ] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples,
        )

    def prediction_step(
        self, model, inputs, prediction_loss_only=False, ignore_keys=None
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        if self.data_type == "tile":
            # TODO: why 50 here?
            # predictions = np.argsort(outputs.cpu().detach().numpy())[:50]
            # predictions = torch.sigmoid(outputs).cpu().detach().numpy()
            predictions = (outputs / 100).cpu().detach().numpy()
            return loss, torch.tensor([predictions]), inputs["target"]
        else:
            predictions = outputs.cpu().detach().numpy()
            target = inputs["target"].cpu().detach().unsqueeze(0)

            return (
                loss,
                torch.tensor([predictions]),
                target,
            )


def score_tile_mean(predictions, df):
    score = 0
    for i in range(len(df)):
        preds_idx = np.argsort(predictions[i])[:50]
        predbest = np.mean(df.iloc[i]["config_runtime"][preds_idx])
        best = np.mean(np.sort(df.iloc[i]["config_runtime"])[:50])
        score += 2 - predbest / best
    score /= len(df)
    return score


def score_tile_max(predictions, df):
    score = 0
    for i in range(len(df)):
        preds_idx = np.argsort(predictions[i])[:50]
        predbest = np.min(df.iloc[i]["config_runtime"][preds_idx[:5]])
        best = np.min(df.iloc[i]["config_runtime"])
        # print(best,predbest)
        score += 2 - predbest / best
    score /= len(df)
    return score


class TileComputeMetricsFn:
    def __init__(self, df, split="valid"):
        self.df = df
        self.split = split

    def __call__(self, eval_preds, padding_value: int = -9999):
        if self.split == "test":
            return {
                "score_tile_mean": 0.0,
                "score_tile_max": 0.0,
            }

        # calculate accuracy using sklearn's function
        predictions, labels = eval_preds

        # filter -100 from predictions
        new_predictions = []
        for e in predictions:
            new_predictions.append(np.array([x for x in e if x != padding_value]))

        predictions = new_predictions

        return {
            "score_tile_mean": score_tile_mean(predictions, self.df),
            "score_tile_max": score_tile_max(predictions, self.df),
        }


class LayoutComputeMetricsFn:
    def __init__(self, df, split="valid"):
        self.df = df
        self.split = split

    def __call__(self, eval_preds, padding_value: int = -9999):
        if self.split == "test":
            return {"kendalltau": 0.0}

        # calculate accuracy using sklearn's function
        predictions, labels = eval_preds

        # filter -100 from predictions and labels -- hf padds if the batch is not full
        new_predictions = []
        new_labels = []
        for i in range(len(predictions)):
            to_keep_ids = np.where(labels[i] != padding_value)[0]
            new_predictions.append(predictions[i][to_keep_ids])
            new_labels.append(labels[i][to_keep_ids])

        predictions = new_predictions
        labels = new_labels
        assert len(predictions) == len(self.df)

        searches = self.df["search"].unique()
        score_dict = {}
        for search in searches:
            scores = []
            for file_id, rows in self.df[self.df["search"] == search].groupby("file"):
                idx = rows.index.tolist()
                prediction = np.concatenate([predictions[i] for i in idx])
                gt_ranks = np.concatenate([labels[i] for i in idx])
                if sum([x.shape[0] for x in rows["config_runtime"]]) != len(prediction):
                    print(
                        f"WARNING: shape not mathing {len(prediction)}, {len(gt_ranks)}, {sum([x.shape[0] for x in rows['config_runtime']])}"
                    )

                score = kendalltau(prediction, gt_ranks).statistic
                scores.append(score)

            score_dict["kendalltau_" + search] = np.mean(scores)

        score_dict["kendalltau"] = np.mean(list(score_dict.values()))
        return score_dict
