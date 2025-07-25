import torch
import torch.nn as nn
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, 
                q: Tensor,  # (batch, heads, seq_len, d_k)
                k: Tensor,  # (batch, heads, seq_len, d_k)
                v: Tensor,  # (batch, heads, seq_len, d_v)
                mask: Tensor | None = None  # optional mask
               ) -> Tensor:
        d_k = q.size(-1)
        # 1) skorları hesapla
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        # 2) maskele (opsiyonel)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 3) softmax ile ağırlıklandır, dropout
        attn_weights = self.dropout(self.softmax(scores))
        
        # 4) çıktı = ağırlıklar * V
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention block.
    """
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int = 8, 
                 dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.d_k = self.d_v = embed_dim // num_heads
        self.num_heads = num_heads
        
        # Ortak lineer projeksiyonlar
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.attention = ScaledDotProductAttention(dropout)
        
        # Son birleştirme lineeri
        self.w_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, embed_dim = x.size()
        
        # 1) Q,K,V hesapla ve head’lere böl
        def reshape(t: Tensor) -> Tensor:
            # (batch, seq_len, embed_dim) -> (batch, heads, seq_len, d_k)
            return t.view(batch_size, seq_len, self.num_heads, self.d_k) \
                    .transpose(1, 2)
        
        q = reshape(self.w_q(x))
        k = reshape(self.w_k(x))
        v = reshape(self.w_v(x))
        
        # 2) Attention uygula
        attn_output, _ = self.attention(q, k, v, mask)
        
        # 3) Head’leri birleştir
        attn_output = attn_output.transpose(1, 2) \
                       .contiguous() \
                       .view(batch_size, seq_len, embed_dim)
        
        # 4) Çıkış lineeri ve residual + norm
        out = self.w_o(attn_output)
        out = self.dropout(out)
        return self.layer_norm(x + out)
