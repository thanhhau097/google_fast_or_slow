import gc
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau
from torch import nn
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach


class CustomTrainer(Trainer):
    def __init__(self, data_type="tile", **kwargs):
        super().__init__(**kwargs)
        self.data_type = data_type
        self.loss_fn = nn.MSELoss()

    def compute_loss(self, model: any, inputs: Dict, return_outputs=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.data_type == "tile":
            outputs = model(
                inputs["config_feat"],
                inputs["node_feat"],
                inputs["node_opcode"],
                inputs["edge_index"],
            )
        else:
            outputs = model(
                inputs["node_config_feat"],
                inputs["node_feat"],
                inputs["node_opcode"],
                inputs["edge_index"],
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
            del inputs["config_feat"]
        else:
            del inputs["node_config_feat"]

        del inputs["node_feat"]
        del inputs["node_opcode"]
        del inputs["edge_index"]

        gc.collect()

        if self.data_type == "tile":
            # TODO: why 50 here?
            predictions = np.argsort(outputs.cpu().detach().numpy())[:50]
            return loss, torch.tensor([predictions]), inputs["target"]
        else:
            predictions = np.argsort(outputs.cpu().detach().numpy())
            return (
                loss,
                torch.tensor([predictions]),
                inputs["target"].cpu().detach().unsqueeze(0),
            )


def score_tile_mean(predictions, df):
    score = 0
    for i in range(len(df)):
        predbest = np.mean(df.iloc[i]["config_runtime"][predictions[i]])
        best = np.mean(np.sort(df.iloc[i]["config_runtime"])[:50])
        score += 2 - predbest / best
    score /= len(df)
    return score


def score_tile_max(predictions, df):
    score = 0
    for i in range(len(df)):
        predbest = np.min(df.iloc[i]["config_runtime"][predictions[i][:5]])
        best = np.min(df.iloc[i]["config_runtime"])
        # print(best,predbest)
        score += 2 - predbest / best
    score /= len(df)
    return score


class TileComputeMetricsFn:
    def __init__(self, df):
        self.df = df

    def __call__(self, eval_preds):
        # calculate accuracy using sklearn's function
        predictions, labels = eval_preds

        # filter -100 from predictions
        new_predictions = []
        for e in predictions:
            new_predictions.append(np.array([x for x in e if x != -100]))

        predictions = new_predictions

        return {
            "score_tile_mean": score_tile_mean(predictions, self.df),
            "score_tile_max": score_tile_max(predictions, self.df),
        }


class LayoutComputeMetricsFn:
    def __init__(self, df):
        self.df = df

    def __call__(self, eval_preds):
        # calculate accuracy using sklearn's function
        predictions, labels = eval_preds

        # filter -100 from predictions
        new_predictions = []
        for e in predictions:
            new_predictions.append(np.array([x for x in e if x != -100]))

        predictions = new_predictions

        scores = []
        for i in range(len(self.df)):
            prediction = predictions[i]
            gt_ranks = np.argsort(labels[i])

            score = kendalltau(prediction, gt_ranks).statistic
            scores.append(score)

        return {"kendalltau": np.mean(scores)}
