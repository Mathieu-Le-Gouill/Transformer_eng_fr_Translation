import torch
from torch.nn.utils.rnn import pad_sequence
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

def tokenize_texts(
    texts,
    tokenizer,
    max_len=64,
    padding="max_length",
    truncation=True,
    return_tensors=None,
    add_special_tokens=True
):
    """
    Tokenizes a list of texts or a single text.
    
    Returns:
        dict with "input_ids"
    """
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]
    
    enc = tokenizer(
        texts,
        padding=padding,
        truncation=truncation,
        max_length=max_len,
        return_tensors=return_tensors,
        add_special_tokens=add_special_tokens
    )
    
    return enc

def prepare_dataset(dataset, src_tokenizer, target_tokenizer, max_len):

    # Preprocess each batch
    dataset = dataset.map(
        lambda batch: preprocess_batch(batch, src_tokenizer, target_tokenizer, max_len),
        batched=True,
        batch_size=1000,
        num_proc=4, 
        remove_columns=dataset.column_names,
    )

    dataset.set_format(
        type="torch",
         columns=["src", "target", "labels"]
    )

    return dataset


def preprocess_batch(batch, src_tokenizer, target_tokenizer, max_len):
    """
    Prepares a batch: tokenizes, adds BOS/EOS, and returns padded tensors.
    """
    bos_id = target_tokenizer.cls_token_id  # use CLS as BOS
    eos_id = target_tokenizer.sep_token_id  # use SEP as EOS

    # Extract source and target texts
    src_texts = [ex["en"] for ex in batch["translation"]]
    target_texts = [ex["fr"] for ex in batch["translation"]]

    # Tokenize source (with padding)
    src_enc = tokenize_texts(src_texts, src_tokenizer, max_len=max_len, return_tensors=None, add_special_tokens=True)

    # Tokenize target (without BOS/EOS, will add manually)
    target_enc = tokenize_texts(target_texts, target_tokenizer, max_len=max_len-2, return_tensors=None, add_special_tokens=False)

    # Convert input_ids to list of tensors
    src_tensors = [torch.tensor(ids, dtype=torch.long) for ids in src_enc["input_ids"]]
    target_tensors = [torch.tensor(ids, dtype=torch.long) for ids in target_enc["input_ids"]]

    decoder_inputs = []
    labels = []

    for target in target_tensors:
        full_input = torch.cat([
            torch.tensor([bos_id], dtype=torch.long), # BOS at start
            target,
            torch.tensor([eos_id], dtype=torch.long) # EOS at the end
        ])
        decoder_inputs.append(full_input[:-1])  # decoder input excludes last token
        labels.append(full_input[1:])           # labels exclude first token

    # Pad sequences to max length in batch
    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=src_tokenizer.pad_token_id)
    target_padded = pad_sequence(decoder_inputs, batch_first=True, padding_value=target_tokenizer.pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=target_tokenizer.pad_token_id)

    return {
        "src": src_padded,
        "target": target_padded,
        "labels": labels_padded
    }


def translate_sentences(model, tokenizer_src, tokenizer_target, sentences, max_len=50, device="cuda"):
    """
    Translate multiple sentences at once using the Transformer model.
    
    Args:
        model: trained Transformer model
        tokenizer_src: tokenizer for source language
        tokenizer_target: tokenizer for target language
        sentences: list of source sentences
        max_len: maximum generated length
        device: "cuda" or "cpu"
        
    Returns:
        List of translated sentences
    """
    model.eval()

    # tokenize the input sentences
    src_enc = tokenize_texts(sentences, tokenizer_src, max_len=max_len, return_tensors="pt", add_special_tokens=True)

    # Use the transformer to translate the tokenized sentences
    src_input_ids = src_enc["input_ids"].to(device)

    with torch.no_grad():
        generated_ids = model(src=src_input_ids, target=None)

    # Decode the translated tokenized sentences
    translations = [
        tokenizer_target.decode(ids, skip_special_tokens=True)
        for ids in generated_ids
    ]

    return translations