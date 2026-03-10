import torch
from torch import nn, dtype

class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int, dtype: dtype, device):
        """
        Transformer sinusoidal positional encoding

        Args:
            max_len: Maximum sequence length supported.
            d_model: Dimension of the embeddings (must be even).
            dtype: Data type for the positional encodings.
            device: Device to store the positional encodings.
        """
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("d_model must be even")

        # pos : the tokens positions in the sequence
        pos = torch.arange(max_len, dtype=dtype, device=device).unsqueeze(-1) # (max_len, 1)
        # dim : ids of the embedding dimension
        dim = torch.arange(0, d_model, 2, dtype=dtype, device=device) # (d_model)

        # div_term : 1 / 10000^(dim/d_model)
        div_term = torch.exp(dim * -torch.log( torch.tensor(10000.0, dtype=dtype, device=device) ) / d_model) 

        self.pe = torch.zeros(max_len, d_model, device=device, requires_grad=False) # (max_len, d_model)

        self.pe[:, 0::2] = torch.sin(pos * div_term)
        self.pe[:, 1::2] = torch.cos(pos * div_term)
    

    def forward(self, seq_len: int):
        """
        Returns the positional encodings for a sequence of a given length.

        Args:
            seq_len: Length of the sequence.

        Returns:
            Positional encodings of shape (seq_len, d_model).
        """

        return self.pe[:seq_len, :] # (seq_len, d_model)