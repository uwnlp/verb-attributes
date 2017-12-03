"""
Experimental loss function from https://github.com/pytorch/pytorch/pull/1792/files
TODO: Import the real thing once it gets merged into pytorch pip version
"""

from torch import nn
from torch.autograd import Variable

class BCEWithLogitsLoss(nn.Module):
    r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single class.
    This version is more numerically stable than using a plain `Sigmoid` followed by a `BCELoss` as, by combining the
    operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
    This Binary Cross Entropy between the target and the output logits (no sigmoid applied) is:
    .. math:: loss(o, t) = - 1/n \sum_i (t[i] * log(sigmoid(o[i])) + (1 - t[i]) * log(1 - sigmoid(o[i])))
    or in the case of the weights argument being specified:
    .. math:: loss(o, t) = - 1/n \sum_i weights[i] * (t[i] * log(sigmoid(o[i])) + (1 - t[i]) * log(1 - sigmoid(o[i])))
    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers between 0 and 1.
    By default, the losses are averaged for each minibatch over observations
    *as well as* over dimensions. However, if the field `size_average` is set
    to `False`, the losses are instead summed.
    """
    def __init__(self, weight=None, size_average=True):
        super(BCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        if self.weight is not None:
            return binary_cross_entropy_with_logits(input, target, Variable(self.weight), self.size_average)
        else:
            return binary_cross_entropy_with_logits(input, target, size_average=self.size_average)


# This could go under functional
def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True):
    r"""Function that measures Binary Cross Entropy between target and output logits:
    See :class:`~torch.nn.BCEWithLogitsLoss` for details.
    Args:
        input: Variable of arbitrary shape
        target: Variable of the same shape as input
        weight (Variable, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch.
    """
    if weight is not None and target.dim() != 1:
        weight = weight.view(1, target.size(1)).expand_as(target)

    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target.float() + (1 + neg_abs.exp()).log()

    if weight is not None:
        loss = loss * weight

    if size_average:
        return loss.mean()
    else:
        return loss.sum()
