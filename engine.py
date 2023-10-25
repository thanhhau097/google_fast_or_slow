import gc
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from pytorchltr.loss import PairwiseHingeLoss
from scipy.stats import kendalltau
from torch import nn
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach


# https://pytorchltr.readthedocs.io/en/stable/loss.html
def pairwise_hinge_loss(y_pred, y_true):
    loss_fn = PairwiseHingeLoss()

    y_pred = y_pred.unsqueeze(0)
    y_true = y_true.unsqueeze(0)
    return loss_fn(
        y_pred, y_true, n=torch.tensor([y_pred.shape[1]], device=y_pred.device)
    ).mean()


# https://github.dev/allegro/allRank/blob/master/allrank/models/losses/listMLE.py
# TODO: not converged yet
def listMLE(y_pred, y_true, eps=1e-10, padded_value_indicator=-float("inf")):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    if len(y_pred.shape) == 1:
        y_pred = y_pred.unsqueeze(0)

    if len(y_true.shape) == 1:
        y_true = y_true.unsqueeze(0)

    random_indices = torch.randperm(y_pred.shape[-1])

    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(
        preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1
    ).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))


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
                inputs["node_opcode"],
                inputs["edge_index"],
            )
        else:
            outputs = model(
                inputs["node_config_feat"],
                inputs["node_feat"],
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
            predictions = outputs.cpu().detach().numpy()
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

        # filter -100 from predictions and labels -- hf padds if the batch is not full
        new_predictions = []
        new_labels = []
        for i in range(len(predictions)):
            new_predictions.append(np.array([x for x in predictions[i] if x != -100]))
            new_labels.append(np.array([x for x in labels[i] if x != -100]))

        predictions = new_predictions
        labels = new_labels
        assert len(predictions) == len(self.df)

        scores = []
        for file_id, rows in self.df.groupby("file"):
            idx = rows.index.tolist()
            prediction = np.concatenate([predictions[i] for i in idx])
            gt_ranks = np.concatenate([labels[i] for i in idx])
            assert sum([x.shape[0] for x in rows["config_runtime"]]) == len(prediction)

            score = kendalltau(prediction, gt_ranks).statistic
            scores.append(score)

        return {"kendalltau": np.mean(scores)}
