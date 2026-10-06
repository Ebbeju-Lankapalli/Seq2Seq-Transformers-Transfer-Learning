# Grammar Correction using Seq2Seq

A word-level **Encoder-Decoder Seq2Seq model** built with PyTorch to automatically correct grammatically incorrect English sentences.

## 📌 Overview

The model learns to transform incorrect English sentences into grammatically correct sentences using an **LSTM-based Seq2Seq architecture**.

**Example:**

    Input:  the teacher need a story
    Output: the teacher needs a story

## 🧠 Architecture

    Incorrect Sentence
            ↓
       Word Embedding
            ↓
        LSTM Encoder
            ↓
    Hidden State + Cell State
            ↓
        LSTM Decoder
            ↓
        Linear Layer
            ↓
      Greedy Decoding
            ↓
      Correct Sentence

## ⚙️ Technologies

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## 📊 Dataset

- Approximately **23,000 sentence pairs**
- Columns: `incorrect`, `correct`
- Word-level tokenization
- Shared vocabulary
- Special tokens: `<pad>`, `<start>`, `<end>`, `<unk>`

## 📈 Results

| Metric | Result |
|---|---:|
| Epochs | 20 |
| Embedding Dimension | 128 |
| Hidden Dimension | 256 |
| Batch Size | 64 |
| Best Validation Loss | **0.02077** |
| Final Training Loss | **0.0200** |
| Final Validation Loss | **0.0225** |
| Exact Match Accuracy | **~90%** |
| Device | Apple Silicon MPS |

## 🎯 Grammar Patterns Learned

The model successfully learned:

- Subject-verb agreement
- Verb conjugation
- Past-tense correction
- Article insertion
- Singular/plural agreement

## 📝 Example Predictions

| Incorrect Sentence | Predicted Correction |
|---|---|
| `the teacher need a story` | `the teacher needs a story` |
| `the dog love lunch` | `the dog loves lunch` |
| `i plays the game` | `i play the game` |
| `the child know after school` | `the child knows after school` |
| `we sleeps lunch` | `we sleep lunch` |

## 📁 Project Structure

    Grammar-Correction/
    │
    ├── Grammar Correction using Seq2Seq Model.ipynb
    ├── grammar_dataset.csv
    ├── grammar_seq2seq_best.pt
    └── README.md

## 🚀 Future Work

- Implement Transformer Encoder-Decoder
- Add Attention Mechanism
- Experiment with pretrained language models
- Apply Transfer Learning
- Experiment with LoRA/PEFT
- Support longer and more complex sentences