import torch
from torch import nn
from model.decoder import Decoder
from model.encoder import Encoder
from model.transformer_embedding import TransformerEmbedding

class Transformer(nn.Module):
    def __init__(
        self,
        
        d_model=512,            # The embedding size
        num_heads=8,            # The number of attention heads
        num_layers=6,           # The number of decoder/encoder layers in the transformer
        d_ff=1024,              # Hidden size of the ff network in the attention
        dropout=0.1,            # Probability of dropping values in the dropout layers

        src_pad_id=1,           # Id of the padding token from the source
        target_pad_id=2,        # Id of the padding token from the target

        target_bos_id=3,        # Id of the beginning of sequence token from the target
        target_eos_id=4,        # Id of the end of sequence token from the target

        src_voc_size=32000,     # The source tokenizer vocabulary size
        target_voc_size=32000,  # The target tokenizer vocabulary size

        max_len=64,             # Maximum tokens length generated 
        dtype=torch.float32,      # Type used for transformer units
        device=None             # The device used
    ):
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
        

    def forward(self, src=None, target=None):

        # --- ENCODER ---
        src_cross_mask = self._src_cross_mask(src)

        src = self.enc_embedding(src)

        for encoder in self.encoders:
            src = encoder(src, src_cross_mask)

        encoder_out = src

        # --- DECODER ---

        # Training
        if target is not None:
            target_self_mask = self._target_self_mask(target)

            target = self.dec_embedding(target)

            for decoder in self.decoders:
                target = decoder(target, encoder_out, target_self_mask, src_cross_mask)

            decoder_out = self.linear(target)

            return decoder_out
        
        # Inference
        else:
            assert self.max_len is not None, "max_len must be set for inference"
            return self._generate(encoder_out, src_cross_mask, self.max_len)


    def _src_cross_mask(self, src):
        src_mask = (src != self.src_pad_id).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def _target_self_mask(self, target):
        target_pad_mask = (target != self.target_pad_id).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, target_len, target_len]
        target_len = target.size(1)
        target_sub_mask = torch.tril(torch.ones(target_len, target_len, dtype=torch.bool, device=self.device))
        target_sub_mask = target_sub_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, target_len, target_len]
        target_mask = target_pad_mask & target_sub_mask  # [batch_size, 1, target_len, target_len]
        return target_mask
    
    def _generate(self, encoder_out, src_cross_mask, max_len):
        batch_size = encoder_out.size(0)

        # Start with BOS token
        generated = torch.full(
            (batch_size, 1), self.target_bos_id, dtype=torch.long, device=self.device
        )
        # Boolean mask to track sequences that have finished (EOS generated)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            target_self_mask = self._target_self_mask(generated)

            target_emb = self.dec_embedding(generated)

            out = target_emb
            for decoder in self.decoders:
                out = decoder(out, encoder_out, target_self_mask, src_cross_mask)

            logits = self.linear(out)  # [batch_size, seq_len, vocab_size]

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [batch_size, 1]

            finished = finished | (next_token.squeeze(1) == self.target_eos_id)

            generated = torch.cat([generated, next_token], dim=1)

            if finished.all():
                break

        # Remove initial BOS token for output
        return generated[:, 1:]