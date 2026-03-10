import torch
from torch.nn.utils.rnn import pad_sequence
import nltk
nltk.download("punkt")
from utils.tokenizing.tokenize_texts import tokenize_texts

def prepare_dataset(dataset, src_tokenizer, target_tokenizer, max_len, target_bos_id, target_eos_id):

    # Preprocess each batch
    dataset = dataset.map(
        lambda batch: preprocess_batch(batch, src_tokenizer, target_tokenizer, max_len, target_bos_id, target_eos_id),
        batched=True,
        batch_size=1000,
        num_proc=4, 
        remove_columns=dataset.column_names,
    )

    dataset.set_format(
        type="torch",
         columns=["src", "target"]
    )

    return dataset


def preprocess_batch(batch, src_tokenizer, target_tokenizer, max_len, target_bos_id, target_eos_id):
    """
    Prepares a batch: tokenizes, adds BOS/EOS, and returns padded tensors.
    Adds EOS after each sentence using nltk.sent_tokenize.
    Truncates sequences to max_len to match positional encoding size.
    """

    src_texts = [ex["en"] for ex in batch["translation"]]
    target_texts = [ex["fr"] for ex in batch["translation"]]

    src_enc = tokenize_texts(src_texts, src_tokenizer, max_len=max_len, padding=False, 
                             truncation=True, return_tensors=None, add_special_tokens=True)
    src_tensors = [torch.tensor(ids, dtype=torch.long) for ids in src_enc["input_ids"]]

    decoder_inputs = []

    for text in target_texts:
        sentences = nltk.sent_tokenize(text)

        flat_tokens = []

        for sentence in sentences:
            sent_tokens = tokenize_texts([sentence], target_tokenizer,
                                         padding=False, truncation=True,
                                         return_tensors=None, add_special_tokens=False)["input_ids"][0]
            flat_tokens.extend(sent_tokens)
            flat_tokens.append(target_eos_id)

        # Truncate to max_len - 1 (leave space for BOS)
        if len(flat_tokens) > max_len - 1:
            flat_tokens = flat_tokens[:max_len - 1]

        # Add BOS at the start
        full_sequence = [target_bos_id] + flat_tokens
        decoder_inputs.append(torch.tensor(full_sequence, dtype=torch.long))

    # Pad sequences to the longest in the batch
    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=src_tokenizer.pad_token_id)
    target_padded = pad_sequence(decoder_inputs, batch_first=True, padding_value=target_tokenizer.pad_token_id)

    return {
        "src": src_padded,
        "target": target_padded
    }