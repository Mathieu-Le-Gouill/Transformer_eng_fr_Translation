from utils.tokenizing.tokenize_texts import tokenize_texts
import torch

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

    src_enc = tokenize_texts(sentences, tokenizer_src, max_len=max_len, padding="longest", 
                             truncation=True, return_tensors="pt", add_special_tokens=True)

    src_input_ids = src_enc["input_ids"].to(device)

    with torch.no_grad():
        generated_ids = model.generate(src_input_ids, max_len=50, temperature=1.0, top_k=50)

    for ids in generated_ids:
        print(ids[-1], tokenizer_target.eos_token_id)

    # Decode the translated tokenized sentences
    translations = [
        tokenizer_target.decode(ids, skip_special_tokens=True)
        for ids in generated_ids
    ]

    return translations