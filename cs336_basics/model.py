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
        # self.w1_weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_ff, d_model, dtype=self.dtype, device=self.device)))
        # self.w2_weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model, d_ff, dtype=self.dtype, device=self.device)))
        # self.w3_weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_ff, d_model, dtype=self.dtype, device=self.device)))

        self.w1 = Linear(d_model, d_ff, self.dtype, self.device)
        self.w2 = Linear(d_ff, d_model, self.dtype, self.device)
        self.w3 = Linear(d_model, d_ff, self.dtype, self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1.forward(x)

        # Gate branch
        sigmoid_w1_x = torch.sigmoid(w1_x)
        w1_x_sigmoid_w1_x = einsum(w1_x, sigmoid_w1_x, '... d_ff , ... d_ff -> ... d_ff')

        # Value branch
        w3_x = self.w3.forward(x)

        point_wise_apply_gate = einsum(w1_x_sigmoid_w1_x, w3_x, '... d_ff, ... d_ff -> ... d_ff')

        down_project = self.w2.forward(point_wise_apply_gate)
        
        return down_project

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta_base = theta
        self.d_k = d_k
        self.num_pairs = int(d_k / 2)
        self.max_seq_len = max_seq_len # length of M (number of tokens)
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

    softmax_q_k_transpose = softmax(scaled_q_k_transpose, -1)
    attention = einsum(softmax_q_k_transpose, V, '... queries keys, ... keys d_v -> ... queries d_v')

    return attention

class MultiheadSelfAttention(torch.nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, theta: float | None=None, max_seq_len: int | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.d_k = self.d_v = int(d_model / num_heads)
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #Torch convention is to have preserved access first D_out first and d_in on the inside
        Q = self.q_proj.forward(x)
        K = self.k_proj.forward(x)
        V = self.v_proj.forward(x)

        Q_heads = rearrange(Q, '... sequence_length (h d_k) ->  ... h sequence_length d_k', d_k=self.d_k, h=self.num_heads) 
        K_heads = rearrange(K, '... sequence_length (h d_k) -> ... h sequence_length d_k', d_k=self.d_k, h=self.num_heads)
        V_heads = rearrange(V, '... sequence_length (h d_k) -> ... h sequence_length d_k', d_k=self.d_k, h=self.num_heads)

        if self.theta is not None:
            rope_module = RotaryPositionalEmbedding(self.theta, self.d_k, self.max_seq_len)
            token_positions = torch.arange(x.shape[-2])
            Q_heads = rope_module.forward(Q_heads, token_positions)
            K_heads = rope_module.forward(K_heads, token_positions)
        
        #Note causal mask shape is shape seq_len * seq_len
        causal_mask = torch.triu(torch.ones((x.shape[-2], x.shape[-2]), dtype=self.dtype, device=self.device), diagonal=1).bool()
        
        attention_matrix = scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask=~causal_mask) # ... h sequence_length queries d_v

        attention_concat = rearrange(attention_matrix, '... h sequence_length d_v -> ... sequence_length (h d_v)')

        weighted_attention = self.output_proj.forward(attention_concat)
        return weighted_attention

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int | None = None, theta: float | None = None, device= None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype

        self.attn = MultiheadSelfAttention(self.d_model, self.num_heads, self.theta, self.max_seq_len, self.device, self.dtype)
        self.ln1 = RMSNorm(self.d_model)
        self.ffn = SwiGLU(self.d_model, self.d_ff)
        self.ln2 = RMSNorm(self.d_model)
        
    def forward(self, x: torch.Tensor):
        identity = x
        ln1_out = self.ln1.forward(x)
        attn_out = self.attn.forward(ln1_out)
        attn_out += identity

        identity_2 = attn_out
        ln2_out = self.ln2.forward(attn_out)
        ffn_out = self.ffn.forward(ln2_out)
        ffn_out += identity_2
        return ffn_out

class TransformerLM(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_length: int, num_layers: int,  rope_theta: float | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.device = device
        self.dtype = dtype

        self.token_embeddings = Embedding(self.vocab_size, self.d_model, self.device, self.dtype)
        self.layers = torch.nn.ModuleList([TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.context_length, self.rope_theta) for _ in range(self.num_layers)])
        self.ln_final = RMSNorm(self.d_model)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, in_indices: torch.Tensor): 
        embeddings = self.token_embeddings.forward(in_indices)
        
        for layer in self.layers:
            embeddings = layer(embeddings)
        
        ln_final_out = self.ln_final.forward(embeddings)
        lm_head_out = self.lm_head.forward(ln_final_out)
        return lm_head_out
        


        









        


    














    
