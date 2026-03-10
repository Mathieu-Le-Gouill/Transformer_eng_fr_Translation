import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.data_processing.data_prep import prepare_dataset
from model.transformer import Transformer
from datasets import load_dataset
from transformers import MarianTokenizer
import os

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_len = 64

    # --- Preparing Dataset ---
    dataset = load_dataset("opus_books", "en-fr")

    dataset_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train = dataset_split["train"]  
    test = dataset_split["test"]

    # Load pretrained tokenizer
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    vocab_size = tokenizer.vocab_size

    train = prepare_dataset(train, src_tokenizer=tokenizer, target_tokenizer=tokenizer,
                            max_len=max_len, target_bos_id=bos_id, target_eos_id=eos_id)
    test  = prepare_dataset(test, src_tokenizer=tokenizer, target_tokenizer=tokenizer,
                            max_len=max_len, target_bos_id=bos_id, target_eos_id=eos_id)

    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test, batch_size=32)

    # --- Loading Model ---
    model = Transformer(
        src_voc_size=vocab_size,
        target_voc_size=vocab_size,

        lambda_sparse=0.01,

        src_pad_id=pad_id,
        target_pad_id=pad_id,

        target_bos_id=bos_id,
        target_eos_id=eos_id,

        d_model=256,
        num_heads=4,
        num_layers=4,
        d_ff=1024,
        dropout=0.1,
        
        max_len=max_len,
        dtype=torch.float32,
        device=device
    ).to(device)

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # --- TRAINING PARAMETERS ---
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
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

            optimizer.zero_grad()
            loss = model.compute_loss(src, target)

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

                loss = model.compute_loss(src, target)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Test loss: {avg_test_loss:.4f}")

        scheduler.step(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            torch.save(model, os.path.join(checkpoint_dir, "transformer.pt"))
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break