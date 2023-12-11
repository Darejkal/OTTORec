import numpy as np
import torch
a=torch.Tensor([[[0,0,1]],[[0,0,1]],[[0,0,1]]])
b=~torch.Tensor(torch.triu(torch.ones((3,3,3)))).bool()
print(a)
print(a* torch.sqrt(4))
exit()
future_mask_const = torch.triu(torch.ones(10, 10) * float('-inf'), diagonal=1)
seq_diag_const=~torch.diag(torch.ones(10, dtype=torch.bool))
def merge_attn_masks(padding_mask):
    """
    padding_mask:
    """
    batch_size = padding_mask.shape[0]
    seq_len = padding_mask.shape[1]
    padding_mask_broadcast = ~padding_mask.bool().unsqueeze(1)
    future_masks = torch.tile(future_mask_const[:seq_len, :seq_len], (batch_size, 1, 1))
    merged_masks = torch.logical_or(padding_mask_broadcast, future_masks)
    # Always allow self-attention to prevent NaN loss
    # See: https://github.com/pytorch/pytorch/issues/41508
    diag_masks = torch.tile(seq_diag_const[:seq_len, :seq_len], (batch_size, 1, 1))
    return torch.logical_and(diag_masks, merged_masks)
print(merge_attn_masks(torch.tile(torch.Tensor([0]+[1]*9),(10,1))))