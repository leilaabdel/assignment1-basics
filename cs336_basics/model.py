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

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dtype = dtype
        self.device = device
        self.w1_weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_ff, d_model, dtype=self.dtype, device=self.device)))
        self.w2_weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model, d_ff, dtype=self.dtype, device=self.device)))
        self.w3_weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_ff, d_model, dtype=self.dtype, device=self.device)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einsum(self.w1_weight, x, 'd_ff d_model, ... d_model ->  ... d_ff')

        # Gate branch
        sigmoid_w1_x = torch.sigmoid(w1_x)
        w1_x_sigmoid_w1_x = einsum(w1_x, sigmoid_w1_x, '... d_ff , ... d_ff -> ... d_ff')

        # Value branch
        w3_x = einsum(self.w3_weight, x, 'd_ff d_model, ... d_model -> ... d_ff')

        point_wise_apply_gate = einsum(w1_x_sigmoid_w1_x, w3_x, '... d_ff, ... d_ff -> ... d_ff')

        down_project = einsum(self.w2_weight, point_wise_apply_gate, 'd_model d_ff, ... d_ff -> ... d_model')
        
        return down_project

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta_base = theta
        self.d_k = d_k
        self.num_pairs = int(d_k / 2)
        self.max_seq_len = max_seq_len # length of M
        self.device = device
        
        positions = torch.arange(self.max_seq_len)
        thetas_index = torch.arange(self.num_pairs) 
        
        exponents = -2 / self.d_k * thetas_index
        thetas = self.theta_base ** exponents

        angles = einsum(positions, thetas, 'max_sequence_length, num_pairs -> max_sequence_length num_pairs')

        cos_table = torch.cos(angles)
        sin_table = torch.sin(angles)

        self.register_buffer('cos_table', cos_table)
        self.register_buffer('sin_table', sin_table)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_reshaped = rearrange(x, '... sequence_length (num_pairs pair_components) -> ... sequence_length num_pairs pair_components', num_pairs=self.num_pairs, pair_components=2)
        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]

        cos_selected = self.cos_table[token_positions]
        sin_selected = self.sin_table[token_positions]

        out_even = x_even * cos_selected - x_odd * sin_selected
        out_odd = x_even * sin_selected + x_odd * cos_selected
        out_interleaved = rearrange(torch.stack([out_even, out_odd], dim= -1 ), '... sequence_length num_pairs pair_components -> ... sequence_length (num_pairs pair_components)')
        return out_interleaved

def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    values, indices = torch.max(in_features, dim=dim, keepdim=True)
    subtracted_features = in_features - values

    exp_term = torch.exp(subtracted_features)
    sum_exp = torch.sum(exp_term, dim=dim, keepdim=True)

    soft_max = exp_term / sum_exp
    
    return soft_max

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    q_k_transpose = einsum(Q, K, '... queries d_k , ... keys d_k -> ... queries keys')

    d_k = K.shape[-1]
    sqrt_d_k = torch.sqrt(torch.tensor(d_k))

    scaled_q_k_transpose = q_k_transpose / sqrt_d_k

    if mask is not None:
       scaled_q_k_transpose = scaled_q_k_transpose.masked_fill(~mask, -float("inf"))

    print("scaled_q_k_transpose stats:")
    print("  min:", scaled_q_k_transpose.min().item())
    print("  max:", scaled_q_k_transpose.max().item())
    print("  any inf:", torch.isinf(scaled_q_k_transpose).any().item())
    print("  any nan:", torch.isnan(scaled_q_k_transpose).any().item())
    print("  any all-(-inf) rows:", (scaled_q_k_transpose == -float('inf')).all(dim=-1).any().item())

    softmax_q_k_transpose = softmax(scaled_q_k_transpose, -1)
    attention = einsum(softmax_q_k_transpose, V, '... queries keys, ... keys d_v -> ... queries d_v')

    return attention

class MultiheadSelfAttention(torch.nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, theta: int | None=None, token_positions: torch.Tensor | None = None, max_seq_len: int | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.d_k = self.d_v = int(d_model / num_heads)
        self.theta = theta
        self.token_positions = token_positions
        self.max_seq_len = max_seq_len
        self.q_proj_weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model, d_model, dtype=self.dtype, device=self.device)))
        self.k_proj_weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model, d_model, dtype=self.dtype, device=self.device)))
        self.v_proj_weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model, d_model, dtype=self.dtype, device=self.device)))
        self.o_proj_weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model, d_model, dtype=self.dtype, device=self.device)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #Torch convention is to have preserved access first D_out first and d_in on the inside
        
        Q = einsum(self.q_proj_weight, x, 'd_out d_in, ... sequence_length d_in  -> ... sequence_length d_out')
        K = einsum(self.k_proj_weight, x, 'd_out d_in, ... sequence_length d_in -> ... sequence_length d_out')
        V = einsum(self.v_proj_weight, x,'d_out d_in, ... sequence_length d_in -> ... sequence_length d_out')

        Q_heads = rearrange(Q, '... sequence_length (h d_k) ->  ... h sequence_length d_k', d_k=self.d_k, h=self.num_heads) 
        K_heads = rearrange(K, '... sequence_length (h d_k) -> ... h sequence_length d_k', d_k=self.d_k, h=self.num_heads)
        V_heads = rearrange(V, '... sequence_length (h d_k) -> ... h sequence_length d_k', d_k=self.d_k, h=self.num_heads)

        if self.theta is not None:
            rope_module = RotaryPositionalEmbedding(self.theta, self.d_k, self.max_seq_len)
            Q_heads = rope_module.forward(Q_heads, self.token_positions)
            K_heads = rope_module.forward(K_heads, self.token_positions)
        
        #Note causal mask shape is shape seq_len * seq_len
        causal_mask = torch.triu(torch.ones((x.shape[-2], x.shape[-2]), dtype=self.dtype, device=self.device), diagonal=1).bool()
        
        attention_matrix = scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask=~causal_mask) # ... h sequence_length queries d_v

        attention_concat = rearrange(attention_matrix, '... h sequence_length d_v -> ... sequence_length (h d_v)')

        weighted_attention = einsum(self.o_proj_weight, attention_concat, 'd_out d_in, ... sequence_length d_in -> ... sequence_length d_out')

        return weighted_attention








        


    














    
