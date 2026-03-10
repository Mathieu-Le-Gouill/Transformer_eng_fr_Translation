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