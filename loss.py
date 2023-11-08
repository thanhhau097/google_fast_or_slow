import torch


def batch_pairs(x: torch.Tensor) -> torch.Tensor:
    """Returns a pair matrix

    This matrix contains all pairs (i, j) as follows:
        p[_, i, j, 0] = x[_, i]
        p[_, i, j, 1] = x[_, j]

    Args:
        x: The input batch of dimension (batch_size, list_size) or
            (batch_size, list_size, 1).

    Returns:
        Two tensors of size (batch_size, list_size ^ 2, 2) containing
        all pairs.
    """

    if x.dim() == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))

    # Construct broadcasted x_{:,i,0...list_size}
    x_ij = torch.repeat_interleave(x, x.shape[1], dim=2)

    # Construct broadcasted x_{:,0...list_size,i}
    x_ji = torch.repeat_interleave(x.permute(0, 2, 1), x.shape[1], dim=1)

    return torch.stack([x_ij, x_ji], dim=3)


class _PairwiseAdditiveLoss(torch.nn.Module):
    """Pairwise additive ranking losses.

    Implementation of linearly decomposible additive pairwise ranking losses.
    This includes RankSVM hinge loss and variations.
    """

    def __init__(self):
        r""""""
        super().__init__()

    def _loss_per_doc_pair(
        self, score_pairs: torch.FloatTensor, rel_pairs: torch.LongTensor
    ) -> torch.FloatTensor:
        """Computes a loss on given score pairs and relevance pairs.

        Args:
            score_pairs: A tensor of shape (batch_size, list_size,
                list_size, 2), where each entry (:, i, j, :) indicates a pair
                of scores for doc i and j.
            rel_pairs: A tensor of shape (batch_size, list_size, list_size, 2),
                where each entry (:, i, j, :) indicates the relevance
                for doc i and j.

        Returns:
            A tensor of shape (batch_size, list_size, list_size) with a loss
            per document pair.
        """
        raise NotImplementedError

    def _loss_reduction(self, loss_pairs: torch.FloatTensor) -> torch.FloatTensor:
        """Reduces the paired loss to a per sample loss.

        Args:
            loss_pairs: A tensor of shape (batch_size, list_size, list_size)
                where each entry indicates the loss of doc pair i and j.

        Returns:
            A tensor of shape (batch_size) indicating the loss per training
            sample.
        """
        # return loss_pairs.view(loss_pairs.shape[0], -1).sum(1)
        return loss_pairs.view(loss_pairs.shape[0], -1).mean(1)

    def _loss_modifier(self, loss: torch.FloatTensor) -> torch.FloatTensor:
        """A modifier to apply to the loss."""
        return loss

    def forward(
        self, scores: torch.FloatTensor, relevance: torch.LongTensor, n: torch.LongTensor
    ) -> torch.FloatTensor:
        """Computes the loss for given batch of samples.

        Args:
            scores: A batch of per-query-document scores.
            relevance: A batch of per-query-document relevance labels.
            n: A batch of per-query number of documents (for padding purposes).
        """
        # Reshape relevance if necessary.
        if relevance.ndimension() == 2:
            relevance = relevance.reshape((relevance.shape[0], relevance.shape[1], 1))
        if scores.ndimension() == 2:
            scores = scores.reshape((scores.shape[0], scores.shape[1], 1))

        # Compute pairwise differences for scores and relevances.
        score_pairs = batch_pairs(scores)
        rel_pairs = batch_pairs(relevance)

        # Compute loss per doc pair.
        loss_pairs = self._loss_per_doc_pair(score_pairs, rel_pairs)

        # Mask out padded documents per query in the batch
        n_grid = n[:, None, None].repeat(1, score_pairs.shape[1], score_pairs.shape[2])
        arange = torch.arange(score_pairs.shape[1], device=score_pairs.device)
        range_grid = torch.max(*torch.meshgrid([arange, arange]))
        range_grid = range_grid[None, :, :].repeat(n.shape[0], 1, 1)
        loss_pairs[n_grid <= range_grid] = 0.0

        # Reduce final list loss from per doc pair loss to a per query loss.
        loss = self._loss_reduction(loss_pairs)

        # Apply a loss modifier.
        loss = self._loss_modifier(loss)

        # Return loss
        return loss


class PairwiseHingeLoss(_PairwiseAdditiveLoss):
    r"""Pairwise hinge loss formulation of SVMRank:

    .. math::
        l(\mathbf{s}, \mathbf{y}) = \sum_{y_i > y _j} max\left(
        0, 1 - (s_i - s_j)
        \right)

    Shape:
        - input scores: :math:`(N, \texttt{list_size})`
        - input relevance: :math:`(N, \texttt{list_size})`
        - input n: :math:`(N)`
        - output: :math:`(N)`
    """

    def _loss_per_doc_pair(self, score_pairs, rel_pairs):
        score_pair_diffs = score_pairs[:, :, :, 0] - score_pairs[:, :, :, 1]
        rel_pair_diffs = rel_pairs[:, :, :, 0] - rel_pairs[:, :, :, 1]
        loss = 1.0 - score_pair_diffs
        loss[rel_pair_diffs <= 0.0] = 0.0
        loss[loss < 0.0] = 0.0
        return loss


class PairwiseDCGHingeLoss(PairwiseHingeLoss):
    r"""Pairwise DCG-modified hinge loss:

    .. math::
        l(\mathbf{s}, \mathbf{y}) =
        \frac{-1}{\log\left(
        2 + \sum_{y_i > y_j}
        max\left(0, 1 - (s_i - s_j)\right)
        \right)}

    Shape:
        - input scores: :math:`(N, \texttt{list_size})`
        - input relevance: :math:`(N, \texttt{list_size})`
        - input n: :math:`(N)`
        - output: :math:`(N)`
    """

    def _loss_modifier(self, loss):
        return -1.0 / torch.log(2.0 + loss)


class PairwiseLogisticLoss(_PairwiseAdditiveLoss):
    r"""Pairwise logistic loss formulation of RankNet:

    .. math::
        l(\mathbf{s}, \mathbf{y}) = \sum_{y_i > y_j} \log_2\left(1 + e^{
        -\sigma \left(s_i - s_j\right)
        }\right)

    Shape:
        - input scores: :math:`(N, \texttt{list_size})`
        - input relevance: :math:`(N, \texttt{list_size})`
        - input n: :math:`(N)`
        - output: :math:`(N)`
    """

    def __init__(self, sigma: float = 1.0):
        """
        Args:
            sigma: Steepness of the logistic curve.
        """
        super().__init__()
        self.sigma = sigma

    def _loss_per_doc_pair(self, score_pairs, rel_pairs):
        score_pair_diffs = score_pairs[:, :, :, 0] - score_pairs[:, :, :, 1]
        rel_pair_diffs = rel_pairs[:, :, :, 0] - rel_pairs[:, :, :, 1]
        loss = torch.log2(1.0 + torch.exp(-self.sigma * score_pair_diffs))
        loss[rel_pair_diffs <= 0.0] = 0.0
        return loss


# https://github.dev/allegro/allRank/blob/master/allrank/models/losses/listMLE.py
# TODO: not converged yet
def listMLE(y_pred, y_true, eps=1e-10, padded_value_indicator=-100):
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

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(
        dims=[1]
    )

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))
