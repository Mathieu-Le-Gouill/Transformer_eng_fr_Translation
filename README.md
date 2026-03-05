# Custom Transformer Project (English → French)

### Overview

This project implements a **custom transformer model** for English-to-French translation.

The model is trained on the **[OPUS Books dataset](https://huggingface.co/datasets/opus_books)** and uses **pretrained tokenizers**:

- **English:** `bert-base-uncased`  
- **French:** `camembert-base`  

The project supports **training**, **translation/inference**, and **testing** through a simple **Makefile interface**.

---

### Training Progress

The model was trained for up to 15 epochs. Below is a summary of the **training and test losses** for each epoch:

| Epoch | Train Loss | Test Loss |
|-------|------------|-----------|
| 1     | 4.4641     | 3.6554    |
| 2     | 3.4096     | 3.1144    |
| 3     | 2.9456     | 2.8349    |
| 4     | 2.6589     | 2.6873    |
| 5     | 2.4630     | 2.6027    |
| 6     | 2.3193     | 2.5435    |
| 7     | 2.2071     | 2.4975    |
| 8     | 2.1161     | 2.4791    |
| 9     | 2.0399     | 2.4536    |
| 10    | 1.9738     | 2.4551    |
| 11    | 1.9168     | 2.4517    |
| 12    | 1.8661     | 2.4304    |
| 13    | 1.8218     | 2.4322    |
| 14    | 1.7807     | 2.4324    |
| 15    | 1.7430     | 2.4389    |


Early stopping was triggered at epoch 15 to prevent overfitting.  

---

### What is a Transformer?

A **transformer** is a deep learning architecture designed for **sequence-to-sequence tasks**, such as translation, text summarization, or text generation.  

**Key characteristics:**

- **Attention Mechanism** – Learns which parts of the input sequence are important for generating each output token.  
- **Parallelizable** – Processes all input tokens simultaneously, making training faster than RNNs.  
- **Encoder–Decoder Structure:**  
  - **Encoder:** Converts the input sequence into a context-aware representation.  
  - **Decoder:** Generates the output sequence token by token, attending to the encoder's representation.  

---

### How to Run the Code

#### Show help
```python
make help 
```
To show available commands

#### Training
```python
make train
```
To launch model training on the dataset.

#### Translation / Inference
```python
make translate
```
To translate English sentences to French.

#### Cleanup
```python
make clean    
```
To remove temporary files and virtual environment.

---

### Requirements

```python
torch>=2.1.0
torchvision>=0.15.0
transformers>=5.0.0
datasets>=4.0.0
nltk>=3.9.0
```