import torch
from einops import einsum, reduce, rearrange

class Linear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(out_features, in_features, dtype=self.dtype, device=self.device)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = einsum(x, self.weight, " ... d_in, d_out d_in -> ... d_out")
        return y_hat


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, dtype=self.dtype, device=self.device)))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        one_hot_tokens = torch.nn.functional.one_hot(token_ids, num_classes=self.num_embeddings).float()
        y_hat = einsum(one_hot_tokens, self.weight, '... vocab_size, vocab_size d_model -> ... d_model')
        return y_hat

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype= dtype
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model, dtype=self.dtype, device=self.device)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_squared = torch.square(x)
        x_squared_mean = reduce(x_squared, ' ... d_model -> ... 1', 'mean')
        x_squared_mean_plus_eps = x_squared_mean + self.eps
        x_rms = torch.sqrt(x_squared_mean_plus_eps) #now is shape ... 1
        
        rms = x / x_rms
        
        rms_times_gain = einsum(rms, self.weight, '... d_model , d_model -> ... d_model')


        return rms_times_gain.to(in_dtype)









    
