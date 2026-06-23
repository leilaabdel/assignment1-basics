
import torch 
from einops import einsum, rearrange, reduce
from typing import Optional 
import math 
from collections.abc import Callable, Iterable 

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

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay= 0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "lamb": weight_decay
        }
        super().__init__(params, defaults)

    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["betas"][0] 
            beta_2 = group["betas"][1]
            eps = group["eps"]
            lamb = group["lamb"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data

                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.grad))
                v = state.get("v", torch.zeros_like(p.grad))

                alpha = lr

                alpha_t = alpha * math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t) #new learning rate
                p.data = p.data - alpha * lamb * p.data

                state["m"] = beta_1 * m + (1 - beta_1) * grad
                state["v"] = beta_2 * v + (1 - beta_2) * grad ** 2
                m = state["m"]
                v = state["v"]

                p.data = p.data - alpha_t * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
        return loss


                






    



