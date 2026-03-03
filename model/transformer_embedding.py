from torch import nn
from model.positional_encoding import PositionalEncoding

class TransformerEmbedding(nn.Module):
    def __init__(self, d_model, voc_size, dropout, pad_token_id, max_len, dtype, device):
        super().__init__()

        self.token_emb = nn.Embedding(num_embeddings=voc_size, embedding_dim=d_model, padding_idx=pad_token_id)
        self.pos_enc = PositionalEncoding(max_len, d_model, dtype, device)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)

        x = self.token_emb(x) # Embed the inputs tokens
        x = x + self.pos_enc(seq_len) # Add positional encoding

        return self.dropout(x)
        
