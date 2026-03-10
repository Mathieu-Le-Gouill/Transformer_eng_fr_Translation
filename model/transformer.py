import torch
from torch import Tensor, nn
from model.decoder import Decoder
from model.encoder import Encoder
from model.transformer_embedding import TransformerEmbedding
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(
        self,
        
        d_model: int=512,
        num_heads: int=8,
        num_layers: int=6,
        d_ff: int =1024,
        dropout: float=0.1,

        lambda_sparse: float=0.01,

        src_pad_id: int=1,
        target_pad_id: int=1,

        target_bos_id: int=0,
        target_eos_id: int=2,

        src_voc_size: int=32000,
        target_voc_size: int=32000,

        max_len: int=64,
        dtype: torch.dtype=torch.float32,
        device=None
    ):
        """
        Transformer model implementing the encoder–decoder architecture introduced in "Attention Is All You Need".

        Args:
            d_model: Dimension of the token embeddings and hidden representations.
            num_heads: Number of attention heads in the multi-head attention layers.
            num_layers: Number of stacked encoder and decoder layers.
            d_ff: Hidden dimension of the feed-forward networks within each transformer block.
            dropout: Dropout probability used throughout the model.

            src_pad_id: Token id used for padding in the source sequence.
            target_pad_id: Token id used for padding in the target sequence.

            target_bos_id: Token id representing the beginning of a target sequence.
            target_eos_id: Token id representing the end of a target sequence.

            src_voc_size: Vocabulary size of the source tokenizer.
            target_voc_size: Vocabulary size of the target tokenizer.

            max_len: Maximum length of the sequence generated during inference.
            dtype: Data type used for the transformer computations.
            device: Device on which the model is allocated.
        """
        super().__init__()

        self.src_pad_id = src_pad_id
        self.target_pad_id = target_pad_id

        self.target_bos_id = target_bos_id
        self.target_eos_id = target_eos_id

        self.max_len = max_len
        self.device = device

        self.enc_embedding = TransformerEmbedding(d_model, src_voc_size, dropout, src_pad_id, max_len, dtype, device)
        self.dec_embedding = TransformerEmbedding(d_model, target_voc_size, dropout, target_pad_id, max_len, dtype, device)

        self.encoders = nn.ModuleList([Encoder(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)])
        self.decoders = nn.ModuleList([Decoder(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)])

        # Projection to target vocab
        self.linear = nn.Linear(d_model, target_voc_size)

        self.lambda_sparse = lambda_sparse
        

    def forward(self, src: Tensor, target: Tensor):
        """
        Performs the forward pass of the Transformer during training.

        Args:
            src: Source token ids of shape (B, src_len).
            target: Target token ids of shape (B, target_len).

        Returns:
            Logits over the target vocabulary of shape (B, target_len, target_voc_size).
        """

        total_sparsity_loss = 0.0

        # --- ENCODER ---
        src_cross_mask = self._src_cross_mask(src)

        src = self.enc_embedding(src)

        for encoder in self.encoders:
            src, loss = encoder(src, src_cross_mask)
            total_sparsity_loss += loss

        encoder_out = src

        # --- DECODER ---
        target_self_mask = self._target_self_mask(target)

        target = self.dec_embedding(target)

        for decoder in self.decoders:
            target, loss = decoder(target, encoder_out, target_self_mask, src_cross_mask)
            total_sparsity_loss += loss

        decoder_out = self.linear(target)

        return decoder_out, total_sparsity_loss
    

    def compute_loss(self, src: Tensor, target: Tensor):
        """
        Compute the transformer total loss (cross entropy + sparsity) after a forward pass

        Args:
            src: Source token ids of shape (B, src_len).
            target: Target token ids of shape (B, target_len).

        Returns:
            Total loss
        """

        logits, sparsity_loss = self.forward(src, target[:, :-1])

        V = logits.size(-1)
        logits_flat = logits.view(-1, V)
        labels_flat = target[:, 1:].contiguous().view(-1)

        ce_loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=self.target_pad_id)

        return ce_loss + self.lambda_sparse * sparsity_loss
    

    def generate(
        self, 
        src: torch.Tensor, 
        max_len=None, 
        temperature: float = 1.0, 
        top_k: int = 0
    ):
        """
        Generates a target sequence autoregressively from a source sequence.

        Args:
            src: Source token ids of shape (batch_size, src_len).
            max_len: Maximum number of tokens to generate.
            temperature: Softmax temperature for sampling (>0). 1.0 = normal softmax.
            top_k: Top-k sampling (0 = disabled).

        Returns:
            Generated token ids of shape (B, generated_len)
        """
        if max_len is None:
            max_len = self.max_len

        batch_size = src.size(0)
        src = src.to(self.device)

        # --- ENCODER ---
        src_cross_mask = self._src_cross_mask(src)
        encoder_out = self.enc_embedding(src)
        for encoder in self.encoders:
            encoder_out, _ = encoder(encoder_out, src_cross_mask)

        # --- DECODER INIT ---
        generated = torch.full(
            (batch_size, 1), self.target_bos_id, dtype=torch.long, device=self.device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # --- AUTOREGRESSIVE DECODING ---
        for _ in range(max_len):
            target_mask = self._target_self_mask(generated)
            target_emb = self.dec_embedding(generated)

            out = target_emb
            for decoder in self.decoders:
                out, _ = decoder(out, encoder_out, target_mask, src_cross_mask)

            logits = self.linear(out)[:, -1, :]  # (B, vocab_size)
            
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k)
                min_topk = topk_vals[:, -1].unsqueeze(1)
                logits[logits < min_topk] = -float("Inf")
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, self.target_pad_id), next_token)

            finished = finished | (next_token.squeeze(1) == self.target_eos_id)
            generated = torch.cat([generated, next_token], dim=1)

            if finished.all():
                break

        return generated


    def _src_cross_mask(self, src: Tensor):
        """
        Creates the source padding mask used for encoder self-attention and
        decoder cross-attention.

        The mask prevents attention from attending to padding tokens in the
        source sequence.

        Args:
            src: Source token ids of shape (B, src_len).

        Returns:
            Boolean mask of shape (B, 1, 1, src_len)
        """

        src_mask = (src != self.src_pad_id).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def _target_self_mask(self, target: Tensor):
        """
        Creates the decoder self-attention mask.

        The mask combines a padding mask and a causal mask to ensure that
        tokens can only attend to previous tokens in the sequence and not
        to future positions.

        Args:
            target: Target token ids of shape (B, target_len).

        Returns:
            Boolean mask of shape (B, 1, target_len, target_len).
        """
        target_pad_mask = (target != self.target_pad_id).unsqueeze(1).unsqueeze(2)  # (B, 1, target_len, target_len)
        target_len = target.size(1)
        target_sub_mask = torch.tril(torch.ones(target_len, target_len, dtype=torch.bool, device=self.device))
        target_sub_mask = target_sub_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, target_len, target_len)
        target_mask = target_pad_mask & target_sub_mask  # (B, 1, target_len, target_len)
        return target_mask