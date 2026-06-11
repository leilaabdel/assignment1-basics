
import torch 
from einops import einsum, rearrange, reduce

def cross_entropy(logits: torch.Tensor, targets: torch.tensor):
    targets_expanded = rearrange(targets, 'batch_size -> batch_size 1')
    selected_logits = torch.gather(logits, -1, targets_expanded)

    values, indices = torch.max(logits, dim=-1, keepdim=True)
    subtracted_logits = logits - values

    exp_logits = torch.exp(subtracted_logits)

    sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)

    cross_entropy = - ( selected_logits - values ) + torch.log(sum_exp)
    average_cross_entropy = reduce(cross_entropy, '... cross_entropy -> cross_entropy', 'mean')
    return average_cross_entropy




