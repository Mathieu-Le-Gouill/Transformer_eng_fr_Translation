from torch import nn, Tensor, dtype
from model.positional_encoding import PositionalEncoding

class TransformerEmbedding(nn.Module):
    def __init__(self, d_model: int, voc_size: int, dropout: float, pad_token_id: int, max_len: int, dtype: dtype, device):
        """
        Transformer Embedding Layer

        Args:
            d_model: dimensionality of the token embeddings.
            voc_size: size of the vocabulary.
            dropout: dropout probability applied after embeddings.
            pad_token_id: index used for padding tokens.
            max_len: maximum supported sequence length.
            dtype: data type used for positional encodings.
            device: device where positional encodings are allocated.
        """
        super().__init__()

        self.token_emb = nn.Embedding(num_embeddings=voc_size, embedding_dim=d_model, padding_idx=pad_token_id)
        self.pos_enc = PositionalEncoding(max_len, d_model, dtype, device)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x: Tensor):
        """
        Forward pass of the embedding layer.

        Args:
            x: input token indices of shape (B, seq_len).

        Returns:
            embedded tensor of shape (B, seq_len, d_model).
        """
        seq_len = x.size(1)

        x = self.token_emb(x) # Embed the inputs tokens
        x = x + self.pos_enc(seq_len) # Add positional encoding

        return self.dropout(x)
