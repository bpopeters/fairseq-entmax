import math
from functools import partial
from dataclasses import dataclass, field

from entmax import Entmax15Loss, SparsemaxLoss, EntmaxBisectLoss

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class EntmaxLossCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

    loss_alpha: float = field(
        default=1.5,
        metadata={"help": "alpha value for entmax loss"}
    )


@register_criterion("entmax_loss", dataclass=EntmaxLossCriterionConfig)
class EntmaxLossCriterion(FairseqCriterion):
    def __init__(self, task, loss_alpha, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        assert loss_alpha > 1
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
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        logits = net_output[0]
        logits = logits.view(-1, logits.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = self.criterion(
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none"
        )(logits, target)
        return loss, loss  # weird

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
