# Fenchel-Young Label Smoothing, as introduced in the paper "Smoothing and
# shrinking the sparse seq2seq search space". This criterion appears to
# currently be incompatible with fp16 training (due to the sum in the
# _compute_smoothing method).

from functools import partial
from dataclasses import dataclass, field

from entmax import Entmax15Loss, SparsemaxLoss, EntmaxBisectLoss

import torch.nn as nn
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class FenchelYoungLabelSmoothingLossCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

    loss_alpha: float = field(
        default=1.5,
        metadata={"help": "alpha value for entmax loss"}
    )

    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )


@register_criterion("fyls_loss", dataclass=FenchelYoungLabelSmoothingLossCriterionConfig)
class FenchelYoungLabelSmoothingLossCriterion(FairseqCriterion):
    def __init__(self, task, loss_alpha, label_smoothing, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.label_smoothing = label_smoothing

        # criterion is the same as in the unsmoothed case
        assert loss_alpha >= 1
        if loss_alpha == 1:
            self.criterion = nn.CrossEntropyLoss
        if loss_alpha == 1.5:
            self.criterion = Entmax15Loss
        elif loss_alpha == 2:
            self.criterion = SparsemaxLoss
        else:
            self.criterion = partial(EntmaxBisectLoss, alpha=loss_alpha)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, loss_ystar = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "loss_ystar": loss_ystar.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def _compute_smoothing(self, logits, target, reduce=True):
        z_ystar = logits.gather(1, target.unsqueeze(1)).squeeze(1)
        if 0 <= self.padding_idx < logits.size(-1):
            z_sum = logits.sum(dim=1) - logits[:, self.padding_idx]
            z_bar = z_sum / (logits.size(-1) - 1)
        else:
            z_bar = logits.mean(dim=1)

        loss_smooth = self.label_smoothing * (z_ystar - z_bar)  # size n
        loss_smooth.masked_fill_(target == self.padding_idx, 0.0)
        if reduce:
            loss_smooth = loss_smooth.sum()
        return loss_smooth

    def compute_loss(self, model, net_output, sample, reduce=True):
        logits = net_output[0]
        logits = logits.view(-1, logits.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss_ystar = self.criterion(
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none"
        )(logits, target)

        loss_smooth = self._compute_smoothing(logits, target, reduce)

        loss = loss_ystar + loss_smooth

        return loss, loss_ystar

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        base_loss_sum = sum(log.get("loss_ystar", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "base_loss", base_loss_sum / sample_size, sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
