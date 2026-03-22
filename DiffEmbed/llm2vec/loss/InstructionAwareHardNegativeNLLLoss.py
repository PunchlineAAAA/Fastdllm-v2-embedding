import torch
from torch import nn, Tensor

from .loss_utils import mismatched_sizes_all_gather


class InstructionAwareHardNegativeNLLLoss:
    def __init__(
        self,
        scale: float = 20.0,
        topic_weight: float = 1.0,
        instruction_weight: float = 1.0,
    ):
        self.scale = scale
        self.topic_weight = topic_weight
        self.instruction_weight = instruction_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def _split(self, reps: Tensor):
        if reps is None:
            return None, None
        hidden = reps.shape[-1]
        if hidden % 2 != 0:
            return reps, None
        half = hidden // 2
        return reps[:, :half], reps[:, half:]

    def _similarity(self, a: Tensor, b: Tensor):
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def _combined_similarity(self, q_reps: Tensor, d_reps: Tensor):
        q_topic, q_instruction = self._split(q_reps)
        d_topic, d_instruction = self._split(d_reps)
        scores = self.topic_weight * self._similarity(q_topic, d_topic)
        if q_instruction is not None and d_instruction is not None:
            scores = scores + self.instruction_weight * self._similarity(q_instruction, d_instruction)
        return scores

    def __call__(self, q_reps: Tensor, d_reps_pos: Tensor, d_reps_neg: Tensor = None, metadata=None):
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        if torch.distributed.is_initialized():
            full_d_reps_pos = torch.cat(mismatched_sizes_all_gather(d_reps_pos))
            full_q_reps = torch.cat(mismatched_sizes_all_gather(q_reps))
            full_d_reps_neg = torch.cat(mismatched_sizes_all_gather(d_reps_neg))
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_d_reps_neg = d_reps_neg

        d_reps = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)
        scores = self._combined_similarity(full_q_reps, d_reps) * self.scale
        labels = torch.arange(len(scores), dtype=torch.long, device=scores.device)
        return self.cross_entropy_loss(scores, labels)
