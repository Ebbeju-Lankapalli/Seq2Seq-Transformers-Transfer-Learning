# Seq2Seq, Transformers & Transfer Learning for NLP

A collection of three end-to-end Natural Language Processing projects exploring sequence-to-sequence learning, Transformer architectures, transfer learning, and parameter-efficient fine-tuning.

The repository demonstrates three different approaches to language transformation and understanding:

1. **Grammar Correction using Seq2Seq LSTM**
2. **Review-Based Extractive Question Answering using DistilBERT + LoRA**
3. **Shakespearean English Transformation using a Transformer Encoder-Decoder**

Each project includes its own dataset, Jupyter notebook, README, results, and visualizations where applicable.

---

## 🚀 Repository Overview

| Project | Architecture | Main Task | Key Result |
|---|---|---|---:|
| Grammar Correction | Encoder-Decoder LSTM | Grammar correction | **90.17% Exact Match** |
| Review-Based Extractive QA | DistilBERT + LoRA | Extractive Question Answering | **0.3141 Token F1** |
| Shakespearean English | Transformer Encoder-Decoder | Language style transformation | **97.49% Exact Match** |

### Technologies

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- LoRA
- Jupyter Notebook
- Apple Silicon MPS

---

# 1. Grammar Correction using Seq2Seq LSTM

## 📌 Overview

A word-level **Encoder-Decoder Seq2Seq model** built with PyTorch to automatically transform grammatically incorrect English sentences into grammatically correct sentences.

The model uses an **LSTM-based Encoder-Decoder architecture** with word embeddings, hidden states, cell states, and greedy decoding.

### Example

**Input**

```text
the teacher need a story
```

**Output**

```text
the teacher needs a story
```

## 🧠 Architecture

```text
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
```

## 📊 Dataset

- Approximately **23,281 sentence pairs**
- Training examples: **20,952**
- Validation examples: **2,329**
- Columns: `incorrect`, `correct`
- Word-level tokenization
- Shared vocabulary
- Special tokens: `<pad>`, `<start>`, `<end>`, `<unk>`

## ⚙️ Model Configuration

| Parameter | Value |
|---|---:|
| Architecture | Encoder-Decoder LSTM |
| Embedding Dimension | 128 |
| Hidden Dimension | 256 |
| Batch Size | 64 |
| Epochs | 20 |
| Best Validation Loss | **0.02095** |
| Exact Match Accuracy | **90.17%** |
| Required Accuracy | 40% |
| Device | Apple Silicon MPS |
| Total Parameters | 887,485 |
| Trainable Parameters | 887,485 |

## 📈 Results

The model achieved **90.17% exact-match accuracy**, significantly exceeding the required **40%** threshold.

```text
Model Accuracy     : 90.17%
Required Accuracy  : 40.00%
Status             : PASSED
```

### Visualization

![Grammar Correction Results](Grammar%20Correction%20using%20Seq2Seq%20Model/visualizations/grammar_correction_results_dashboard.png)

The dashboard summarizes training/validation loss, dataset statistics, exact-match evaluation, and model parameters.

## 🎯 Grammar Patterns Learned

The model learns patterns such as:

- Subject-verb agreement
- Verb conjugation
- Past-tense correction
- Article insertion
- Singular/plural agreement

## 📂 Project

[Open Grammar Correction Project](Grammar%20Correction%20using%20Seq2Seq%20Model/)

[Open Jupyter Notebook](Grammar%20Correction%20using%20Seq2Seq%20Model/Grammar%20Correction%20using%20Seq2Seq%20Model.ipynb)

[Project README](Grammar%20Correction%20using%20Seq2Seq%20Model/README.md)

---

# 2. Review-Based Extractive QA using DistilBERT + LoRA

## 📌 Overview

A Transformer-based **extractive Question Answering system** trained on review-based question-answer pairs using **DistilBERT** and **LoRA (Low-Rank Adaptation)**.

The system receives a question and a review/context passage and extracts the most relevant answer span directly from the context.

## 🧠 Architecture

```text
Question + Review Context
          ↓
     DistilBERT
          ↓
      LoRA / PEFT
          ↓
 Start / End Logits
          ↓
 Candidate Answer Span
          ↓
 Predicted Answer
```

## 📊 Dataset

- Total QA pairs: **1,500**
- Training examples: **1,200**
- Validation examples: **300**
- Training features: **1,368**
- Validation features: **340**

Six review domains are represented:

- Books
- Electronics
- Grocery
- Movies
- Restaurants
- Tripadvisor

## ⚙️ Model Configuration

| Parameter | Value |
|---|---:|
| Base Model | `distilbert-base-uncased` |
| Maximum Sequence Length | 384 |
| Document Stride | 128 |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.1 |
| Target Modules | `q_lin`, `v_lin` |
| Epochs | 3 |
| Learning Rate | 2e-4 |
| Batch Size | 8 |
| Total Parameters | 66,513,412 |
| Trainable Parameters | 148,994 |
| Trainable Percentage | **0.224%** |

## 📈 Results

The final validation Token F1 is:

```text
Validation Token F1 : 0.3141
Required Token F1   : 0.2300
Status              : PASSED
```

The project trains only **0.224% of the total model parameters** using LoRA while keeping the majority of the pretrained DistilBERT parameters frozen.

### Visualization

![Review-Based Extractive QA Results](Review-Based%20Extractive%20QA%20using%20Transfer%20Learning/images/review_qa_project_dashboard.png)

The dashboard summarizes training/validation loss, dataset and feature statistics, Token F1, and LoRA parameter efficiency.

## 🔬 Key Concepts

- Extractive Question Answering
- DistilBERT
- Transfer Learning
- Transformer architectures
- Sliding-window tokenization
- Answer span alignment
- Start/end logits
- Parameter-Efficient Fine-Tuning
- LoRA
- PEFT
- Hugging Face Trainer
- Token F1 evaluation

## 📂 Project

[Open Review-Based Extractive QA Project](Review-Based%20Extractive%20QA%20using%20Transfer%20Learning/)

[Open Jupyter Notebook](Review-Based%20Extractive%20QA%20using%20Transfer%20Learning/Review-Based%20Extractive%20QA%20using%20Transfer%20Learning.ipynb)

[Project README](Review-Based%20Extractive%20QA%20using%20Transfer%20Learning/README.md)

---

# 3. Shakespearean English using Transformers

## 📌 Overview

A **Transformer Encoder-Decoder** model that learns to transform modern English sentences into Shakespearean-style English using paired examples.

The model uses:

- Multi-Head Attention
- Positional Encoding
- Transformer Encoder
- Transformer Decoder
- Causal decoder masking
- Teacher forcing
- Greedy decoding

The model learns vocabulary and language-style transformations from approximately **15,000 paired examples** rather than relying on manually programmed translation rules.

### Example

**Modern English**

```text
you are my friend
```

**Shakespearean English**

```text
thou art my companion
```

## 🧠 Architecture

```text
Modern English Sentence
          ↓
   Word Tokenization
          ↓
    Shared Vocabulary
          ↓
    Token Embedding
          ↓
   Positional Encoding
          ↓
┌─────────────────────────┐
│   Transformer Encoder   │
│                         │
│ Multi-Head Attention    │
│ Feed Forward Network    │
└────────────┬────────────┘
             ↓
      Context Memory
             ↓
┌─────────────────────────┐
│   Transformer Decoder   │
│                         │
│ Masked Self-Attention   │
│ Cross-Attention         │
│ Feed Forward Network    │
└────────────┬────────────┘
             ↓
     Linear Projection
             ↓
       Token Prediction
             ↓
      Greedy Decoding
             ↓
 Shakespearean English
```

## ⚙️ Model Configuration

| Parameter | Value |
|---|---:|
| Architecture | Transformer Encoder-Decoder |
| Tokenization | Word-level |
| Vocabulary | Shared |
| Embedding Dimension | 256 |
| Attention Heads | 8 |
| Encoder Layers | 3 |
| Decoder Layers | 3 |
| Feedforward Dimension | 512 |
| Dropout | 0.1 |
| Batch Size | 64 |
| Epochs | 20 |
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Loss Function | Cross-Entropy |
| Decoding | Greedy |
| Device | Apple Silicon MPS |

## 📊 Dataset

- Approximately **15,000 sentence pairs**
- Source column: `modern`
- Target column: `shakespearean`
- Word-level tokenization
- Shared vocabulary
- Sentence length approximately 3–10 words

Special tokens:

| Token | Index | Purpose |
|---|---:|---|
| `<pad>` | 0 | Padding |
| `<sos>` | 1 | Start of sentence |
| `<eos>` | 2 | End of sentence |
| `<unk>` | 3 | Unknown word |

## 📈 Training Loss

The Transformer was trained for **20 epochs**.

The training loss decreased sharply from approximately **0.57** at the beginning of training to approximately **0.019** by the final epoch.

![Transformer Training Loss](Shakespearean%20English%20using%20Transformers/transformer_training_loss.png)

> **Note:** Add the provided `transformer_training_loss.png` image to the `Shakespearean English using Transformers` folder using this exact filename so GitHub can display the chart.

## 🏆 Evaluation Results

The model was evaluated on **1,514 held-out test samples**.

| Metric | Result |
|---|---:|
| Test Samples | 1,514 |
| Exact Matches | 1,476 |
| Non-Exact Predictions | 38 |
| Exact Match Accuracy | **97.49%** |
| Average Cosine Similarity | **0.9999** |
| Required Similarity | 0.60 |

### Evaluation

```text
Exact Match Accuracy : 97.49%
Cosine Similarity     : 0.9999
Required Similarity   : 0.60
```

The model significantly exceeds the required similarity threshold.

### Complete Visualization Dashboard

![Shakespearean Transformer Results](Shakespearean%20English%20using%20Transformers/shakespearean_transformer_visualizations.png)

The dashboard presents the evaluation metrics, test-set prediction results, exact-match performance, and cosine similarity.

## 🎯 Learned Transformations

Examples of patterns learned by the model include:

| Modern English | Shakespearean Style |
|---|---|
| you | thou |
| are | art |
| has | hath |
| perhaps | perchance |
| before | ere |
| here | hither |
| beautiful | fair |
| brave | valiant |
| enemy | foe |
| friend | companion |
| old | aged |
| kingdom | realm |
| listen | hark |

## 📂 Project

[Open Shakespearean English Project](Shakespearean%20English%20using%20Transformers/)

[Open Jupyter Notebook](Shakespearean%20English%20using%20Transformers/Shakespearean%20English%20using%20Transformers.ipynb)

[Project README](Shakespearean%20English%20using%20Transformers/README.md)

---

# 📊 Project Comparison

| Feature | Grammar Correction | Review QA | Shakespearean Transformer |
|---|---|---|---|
| Main Architecture | LSTM Seq2Seq | DistilBERT + LoRA | Transformer Encoder-Decoder |
| Task | Grammar Correction | Extractive QA | Style Transformation |
| Learning Type | Sequence-to-Sequence | Transfer Learning | Sequence-to-Sequence |
| Attention | No explicit attention | Self-Attention | Multi-Head Attention |
| Parameter-Efficient Fine-Tuning | No | **Yes — LoRA** | No |
| Dataset Size | ~23K pairs | 1,500 QA pairs | ~15K pairs |
| Main Metric | Exact Match | Token F1 | Exact Match + Cosine Similarity |
| Result | **90.17%** | **0.3141** | **97.49% / 0.9999** |
| Framework | PyTorch | PyTorch + Hugging Face | PyTorch |

---

# 🔬 What This Repository Demonstrates

This repository provides hands-on implementations of several important NLP and Deep Learning concepts.

### Sequence Modeling

- Encoder-Decoder architectures
- LSTM
- Hidden and cell states
- Teacher forcing
- Greedy decoding

### Transformers

- Transformer Encoder
- Transformer Decoder
- Multi-Head Self-Attention
- Cross-Attention
- Positional Encoding
- Causal masking

### Transfer Learning

- Pretrained DistilBERT
- Fine-tuning
- Hugging Face Transformers
- Hugging Face Trainer

### Parameter-Efficient Fine-Tuning

- LoRA
- PEFT
- Frozen pretrained parameters
- Low-rank adaptation

### NLP Preprocessing

- Word-level tokenization
- Vocabulary construction
- Special tokens
- Padding
- Unknown token handling
- Sliding-window tokenization
- Answer span alignment

### Evaluation

- Exact Match Accuracy
- Token F1
- Cosine Similarity
- Training/validation loss analysis
- Prediction analysis

---

# 📁 Repository Structure

```text
Seq2Seq-Transformers-Transfer-Learning/
│
├── Grammar Correction using Seq2Seq Model/
│   ├── Grammar Correction using Seq2Seq Model.ipynb
│   ├── grammar_dataset.csv
│   ├── grammar_seq2seq_best.pt
│   ├── README.md
│   └── visualizations/
│       ├── 01_sentence_length_distribution.png
│       ├── 02_top_20_words.png
│       ├── 03_training_validation_loss.png
│       ├── 04_exact_match_accuracy.png
│       ├── 05_prediction_examples.png
│       ├── grammar_correction_results_dashboard.png
│       ├── prediction_examples.csv
│       └── results_summary.csv
│
├── Review-Based Extractive QA using Transfer Learning/
│   ├── Review-Based Extractive QA using Transfer Learning.ipynb
│   ├── subjqa_dataset.csv
│   ├── README.md
│   ├── images/
│   │   ├── project_architecture.png
│   │   ├── review_qa_project_dashboard.png
│   │   ├── sample_prediction_f1.png
│   │   ├── token_f1_distribution.png
│   │   └── training_validation_loss.png
│   ├── qa_lora_results/
│   └── subjqa_distilbert_lora/
│
├── Shakespearean English using Transformers/
│   ├── Shakespearean English using Transformers.ipynb
│   ├── shakespeare_dataset.csv
│   ├── shakespearean_transformer_visualizations.png
│   ├── transformer_training_loss.png
│   └── README.md
│
└── README.md
```

---

# 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/Ebbeju-Lankapalli/Seq2Seq-Transformers-Transfer-Learning.git
```

Enter the repository:

```bash
cd Seq2Seq-Transformers-Transfer-Learning
```

Create the Conda environment:

```bash
conda create -n mlproject python=3.12
```

Activate it:

```bash
conda activate mlproject
```

Install the main dependencies:

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

For the Review-Based Extractive QA project:

```bash
pip install transformers datasets peft accelerate
```

For Jupyter:

```bash
pip install notebook
```

Launch Jupyter:

```bash
jupyter notebook
```

---

# ▶️ Running the Projects

Each project is self-contained.

### Grammar Correction

Open:

```text
Grammar Correction using Seq2Seq Model/
```

Then run:

```text
Grammar Correction using Seq2Seq Model.ipynb
```

### Review-Based Extractive QA

Open:

```text
Review-Based Extractive QA using Transfer Learning/
```

Then run:

```text
Review-Based Extractive QA using Transfer Learning.ipynb
```

### Shakespearean English

Open:

```text
Shakespearean English using Transformers/
```

Then run:

```text
Shakespearean English using Transformers.ipynb
```

Run notebook cells sequentially to reproduce preprocessing, training, evaluation, and visualizations.

---

# 🍎 Apple Silicon Support

The models were developed and tested using an Apple Silicon Mac and PyTorch's MPS backend where supported.

Device selection can be implemented with:

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

This allows supported PyTorch operations to use Apple's Metal Performance Shaders acceleration.

---

# 📥 Download / Clone the Complete Repository

### Clone with Git

```bash
git clone https://github.com/Ebbeju-Lankapalli/Seq2Seq-Transformers-Transfer-Learning.git
```

### Open the GitHub Repository

[View the complete repository on GitHub](https://github.com/Ebbeju-Lankapalli/Seq2Seq-Transformers-Transfer-Learning)

### Download as ZIP

On GitHub, select:

```text
Code → Download ZIP
```

---

# 🚀 Future Improvements

Possible extensions across the repository include:

- Transformer-based grammar correction
- Attention-based grammar correction
- Larger pretrained language models
- Beam search decoding
- Subword tokenization
- BPE / WordPiece tokenization
- Larger QA datasets
- SQuAD-style evaluation
- Exact Match and additional QA metrics
- Better answer-span post-processing
- Quantized inference
- LoRA hyperparameter experimentation
- BLEU / ROUGE / METEOR evaluation
- Attention visualization
- FastAPI deployment
- Interactive NLP web applications

---

# 🏁 Conclusion

This repository presents a progression from classical sequence-to-sequence modeling to modern Transformer-based NLP and parameter-efficient transfer learning.

The three projects demonstrate different approaches to language problems:

```text
LSTM Seq2Seq
     ↓
Grammar Correction

DistilBERT + LoRA
     ↓
Extractive Question Answering

Transformer Encoder-Decoder
     ↓
Shakespearean Language Transformation
```

Together, they provide practical experience with **sequence modeling, LSTMs, Transformers, attention mechanisms, transfer learning, LoRA, PEFT, tokenization, text generation, extractive QA, and NLP evaluation**.

---

## 👨‍💻 Author

**Ebbeju Lankapalli**

B.Tech — Computer Science and Engineering (AI & ML)

GitHub: [Ebbeju-Lankapalli](https://github.com/Ebbeju-Lankapalli)

---

## ⭐ If You Find This Repository Useful

Consider giving the repository a ⭐ on GitHub.

[⭐ Star the Repository](https://github.com/Ebbeju-Lankapalli/Seq2Seq-Transformers-Transfer-Learning)
