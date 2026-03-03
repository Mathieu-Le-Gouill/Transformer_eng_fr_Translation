import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import prepare_dataset, save_model
from model.transformer import Transformer
from datasets import load_dataset
import os
from transformers import AutoTokenizer


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_len = 50

    # --- Preparing Dataset ---

    dataset = load_dataset("opus_books", "en-fr")

    dataset_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train = dataset_split["train"]  
    test = dataset_split["test"]

    # Load pretrained tokenizers
    eng_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    fr_tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    train = prepare_dataset(train, src_tokenizer=eng_tokenizer, target_tokenizer=fr_tokenizer, max_len=max_len)
    test  = prepare_dataset(test, src_tokenizer=eng_tokenizer, target_tokenizer=fr_tokenizer, max_len=max_len)

    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test, batch_size=32)

    # --- Loading Model ---
    model = Transformer(
        src_voc_size=eng_tokenizer.vocab_size,
        target_voc_size=fr_tokenizer.vocab_size,

        src_pad_id=eng_tokenizer.pad_token_id,
        target_pad_id=fr_tokenizer.pad_token_id,

        target_bos_id=fr_tokenizer.cls_token_id, # Use cls token for bos because the tokenizer don't thave bos by default
        target_eos_id=fr_tokenizer.sep_token_id, # use sep token for eos because the tokenizer don't thave eos by default

        d_model=256,
        num_heads=4,
        num_layers=4,
        d_ff=1024,
        dropout=0.1,
        
        max_len=max_len,
        dtype=torch.float32,
        device=device
    ).to(device)

    # Model saving path
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # --- TRAINING PARAMETERS ---
    criterion = nn.CrossEntropyLoss(ignore_index=fr_tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        threshold=1e-3,
        cooldown=1
    )

    best_test_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 3

    # --- TRAINING LOOP ---
    num_epochs = 15

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            src = batch["src"].to(device)
            target = batch["target"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(src, target)
            loss = criterion(logits.view(-1, fr_tokenizer.vocab_size), labels.view(-1))

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Train loss: {avg_train_loss:.4f}")

        # --- EVAL ---
        model.eval()
        total_test_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                src = batch["src"].to(device)
                target = batch["target"].to(device)
                labels = batch["labels"].to(device)

                logits = model(src, target)
                loss = criterion(logits.view(-1, fr_tokenizer.vocab_size), labels.view(-1))
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Test loss: {avg_test_loss:.4f}")

        scheduler.step(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            save_model(model, os.path.join(checkpoint_dir, "transformer.pt"))
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break