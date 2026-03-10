import torch
from utils.translation.translate_sentences import translate_sentences
from transformers import MarianTokenizer
import os
import sys


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

    # Model saving path
    checkpoint_dir = "checkpoints"
    model_path = os.path.join(checkpoint_dir, "transformer.pt")

    # Check if model exists
    if not os.path.isfile(model_path):
        sys.exit(f"\nError: Transformer model not found at '{model_path}'.\n"
                "Please make sure you have trained the model or placed the checkpoint in the correct directory.\n")

    # Load model
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # Test sentences
    english_sentences = [
        "It is often said that the early bird catches the worm, but sometimes patience is more valuable.",
        "Had they followed the instructions carefully, they might have avoided the costly mistake.",
        "The book, which was written in the 19th century, still resonates with readers today.",
        "The scientist, who had spent years studying climate change, finally published her groundbreaking research.",
        "Although it was raining heavily, she decided to go for a long walk in the park.",
        "She wondered whether she would ever have the courage to confront her fears.",
        "While waiting for the train, I noticed a group of children playing happily near the station.",
        "If I had known about the meeting earlier, I would have prepared a detailed presentation."
    ]

    french_translation = translate_sentences(model, tokenizer, tokenizer, english_sentences, device=device, max_len=64)

    for eng_sentence, fr_sentence in zip(english_sentences, french_translation):
        print(f"EN: {eng_sentence}")
        print(f"FR: {fr_sentence}\n")
