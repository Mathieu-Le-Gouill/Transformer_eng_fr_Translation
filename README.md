# Custom Transformer Project (English → French)

### Overview

This project implements a **custom transformer model** for English-to-French translation.

The model is trained on the **[OPUS Books dataset](https://huggingface.co/datasets/opus_books)**.

Tokenization is performed using the pretrained **[Helsinki-NLP/opus-mt-en-fr](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr)** tokenizer, which is specifically optimized for English-to-French translation.

The project supports **training**, **translation/inference**, and **testing** through a simple **Makefile interface**.

---

### Training Results

The model was trained for up to 15 epochs.

Below is a summary of the **training and test losses** for each epoch:

| Epoch | Train Loss | Test Loss |
|-------|------------|-----------|
| 1     | 3.6661     | 2.8754    |
| 2     | 2.7687     | 2.4861    |
| 3     | 2.4398     | 2.2492    |
| 4     | 2.2211     | 2.1171    |
| 5     | 2.0720     | 2.0236    |
| 6     | 1.9618     | 1.9582    |
| 7     | 1.8773     | 1.9206    |
| 8     | 1.8091     | 1.8842    |
| 9     | 1.7508     | 1.8613    |
| 10    | 1.7019     | 1.8315    |
| 11    | 1.6591     | 1.8198    |
| 12    | 1.6215     | 1.7947    |
| 13    | 1.5891     | 1.7889    |
| 14    | 1.5576     | 1.7753    |
| 15    | 1.5316     | 1.7635    |

---

### Translation for custom sequences

| English (EN) | French (FR) |
|--------------|-------------|
| It is often said that the early bird catches the worm, but sometimes patience is more valuable. | Il est souvent dit que l’oiseau de bonne humeur, mais la patience est plus précieuse. |
| Had they followed the instructions carefully, they might have avoided the costly mistake. | Ils avaient suivis les instructions, ils auraient évité la tromper. |
| The book, which was written in the 19th century, still resonates with readers today. | Le registre, qui était écrit au 19, toujours des lecteurs avec des lecteurs. |
| The scientist, who had spent years studying climate change, finally published her groundbreaking research. | Le aïeur, qui avait passé des années de travail, finit par accepter ses études. |
| Although it was raining heavily, she decided to go for a long walk in the park. | Bien qu’il pleuvait lourdement, elle se décidait pour aller une longue promenade dans le parc. |
| She wondered whether she would ever have the courage to confront her fears. | Elle s’interrogea si jamais elle eût eu le courage de se nourrir de ses craintes. |
| While waiting for the train, I noticed a group of children playing happily near the station. | Pendant qu'on attendit le train, je remarquai un groupe de enfants qui jouissait heureusement près de la gare. |
| If I had known about the meeting earlier, I would have prepared a detailed presentation. | Si je savais bien quelle était la rencontre, je serais employée à un cadeau. |

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
sentencepiece>=0.2.1
sacremoses>=0.1.1
```