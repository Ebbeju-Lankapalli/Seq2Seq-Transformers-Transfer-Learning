# Shakespearean English using Transformers

A Transformer Encoder-Decoder model that learns to transform modern English sentences into Shakespearean-style English using paired examples.

The project uses a word-level Transformer with Multi-Head Attention, positional encoding, causal decoder masking, teacher forcing, and greedy decoding.

---

## 📌 Project Overview

Language changes significantly across different historical periods. Shakespearean English has distinctive vocabulary and grammatical patterns such as:

- `you` → `thou`
- `are` → `art`
- `has` → `hath`
- `before` → `ere`
- `perhaps` → `perchance`
- `friend` → `companion`

The objective of this project is to build a sequence-to-sequence Transformer that learns these transformations directly from examples.

The model does not use manually defined translation rules. Instead, it learns transformation patterns from approximately 15,000 modern-English and Shakespearean-English sentence pairs.

---

## 🎯 Objective

Build a Transformer Encoder-Decoder model that:

- Accepts a modern English sentence
- Encodes the sentence using a shared vocabulary
- Uses Multi-Head Attention to learn contextual relationships
- Generates Shakespearean-style English autoregressively
- Uses a causal mask during decoding
- Uses greedy decoding during inference
- Provides a reusable `predict()` function

---

## ✨ Example Transformations

| Modern English | Shakespearean English |
|---|---|
| you are my friend | thou art my companion |
| it is very beautiful here | 'tis most fair hither |
| she has a brave enemy | she hath a valiant foe |
| perhaps you will help | perchance thou shall aid |
| goodbye my old friend | farewell my aged companion |

---

## 🧠 Model Architecture

```text
                 Modern English Sentence
                           │
                           ▼
                    Word Tokenization
                           │
                           ▼
                    Shared Vocabulary
                           │
                           ▼
                     Token Embedding
                           │
                           ▼
                   Positional Encoding
                           │
                           ▼
              ┌─────────────────────────┐
              │   Transformer Encoder  │
              │                         │
              │   Multi-Head Attention  │
              │   Feed Forward Network  │
              └────────────┬────────────┘
                           │
                     Context Memory
                           │
                           ▼
              ┌─────────────────────────┐
              │   Transformer Decoder  │
              │                         │
              │   Masked Self-Attention │
              │   Cross-Attention       │
              │   Feed Forward Network  │
              └────────────┬────────────┘
                           │
                           ▼
                   Linear Projection
                           │
                           ▼
                    Token Prediction
                           │
                           ▼
                    Greedy Decoding
                           │
                           ▼
              Shakespearean English
```

---

## 🏗️ Architecture Details

| Component | Configuration |
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
| Training Epochs | 20 |
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Loss Function | Cross-Entropy Loss |
| Decoder Mask | Causal / Look-ahead |
| Decoding | Greedy |
| Hardware | Apple Silicon MPS |

---

## 🔤 Special Tokens

The vocabulary follows the required token mapping:

| Token | Index | Purpose |
|---|---:|---|
| `<pad>` | 0 | Padding sequences |
| `<sos>` | 1 | Start of sentence |
| `<eos>` | 2 | End of sentence |
| `<unk>` | 3 | Unknown words |

---

## 📊 Dataset

### Dataset Information

| Property | Details |
|---|---|
| Dataset | Shakespeare Dataset |
| Approximate Size | 15,000 sentence pairs |
| Source Column | `modern` |
| Target Column | `shakespearean` |
| Tokenization | Word-level |
| Sentence Length | Approximately 3–10 words |

Each row contains a modern English sentence and its corresponding Shakespearean-style transformation.

### Example

Modern:

    you are my friend

Shakespearean:

    thou art my companion

---

## 🔄 Data Processing Pipeline

```text
Raw CSV Dataset
      │
      ▼
Load with Pandas
      │
      ▼
Remove Missing / Empty Rows
      │
      ▼
Train-Test Split
      │
      ▼
Build Shared Vocabulary
      │
      ▼
Add <sos> and <eos>
      │
      ▼
Convert Words → Token IDs
      │
      ▼
Pad Sequences
      │
      ▼
PyTorch Dataset
      │
      ▼
DataLoader
```

---

## 🧮 Sentence Encoding

Every sentence is converted into token IDs.

For example:

    you are my friend

becomes:

    <sos> you are my friend <eos>

which is then converted into numerical token IDs:

    [1, word_id, word_id, word_id, word_id, 2]

Unknown words are mapped to:

    <unk> = 3

---

## 🧩 Transformer Encoder

The encoder receives the complete modern English sentence.

The input passes through:

1. Token Embedding
2. Positional Encoding
3. Multi-Head Self-Attention
4. Feed Forward Network

The encoder produces a contextual representation of the source sentence.

---

## 🧩 Transformer Decoder

The decoder generates the Shakespearean sentence one token at a time.

During training, teacher forcing is used.

For example, the target sentence is:

    <sos> thou art my companion <eos>

The decoder receives the shifted target sequence and learns to predict the next token at every position.

---

## 🔒 Causal Masking

A causal or look-ahead mask prevents the decoder from accessing future target tokens.

This ensures that each generated token depends only on previously available tokens.

    Previous Tokens
          │
          ▼
    Masked Self-Attention
          │
          ▼
    Next Token Prediction

---

## 🎯 Greedy Decoding

During inference, generation starts with the `<sos>` token.

At every step, the model predicts the next token and selects the token with the highest probability using `argmax`.

The generated token is then added to the decoder input.

Generation continues until `<eos>` is produced or the maximum sequence length is reached.

    <sos>
      ↓
    Predict next token
      ↓
    Select highest-probability token
      ↓
    Append token
      ↓
    Predict next token
      ↓
    ...
      ↓
    <eos>

---

## 💻 Prediction

The trained model provides a `predict()` function.

    prediction = predict("you are my friend")
    print(prediction)

Example output:

    thou art companion

---

## 📈 Training

The model was trained for 20 epochs using the Adam optimizer.

Training configuration:

| Parameter | Value |
|---|---:|
| Batch Size | 64 |
| Epochs | 20 |
| Learning Rate | 0.0005 |
| Optimizer | Adam |
| Loss Function | Cross-Entropy Loss |
| Padding Index | 0 |
| Device | Apple Silicon MPS |

### Training Loss

| Metric | Value |
|---|---:|
| Initial Loss | 0.5697 |
| Minimum Observed Loss | 0.0128 |
| Final Loss | 0.0187 |

The substantial reduction in training loss indicates that the model successfully learned the transformation patterns present in the training data.

---

## 🧪 Sample Predictions

The model was tested on several example sentences.

| Modern English | Model Prediction |
|---|---|
| you are my friend | thou art companion |
| it is very beautiful here | 'tis hither to bestow the foe |
| she has a brave enemy | she hath a valiant foe |
| perhaps you will help | perchance thou shall aid the soul |
| goodbye my old friend | farewell my aged companion |

The model correctly learned many characteristic Shakespearean vocabulary and grammatical transformations.

Some predictions generated additional tokens, which is a common behavior in autoregressive sequence generation.

---

## 📊 Test Set Evaluation

The model was evaluated on the complete held-out test set.

### Final Results

| Metric | Result |
|---|---:|
| Test Samples | 1,514 |
| Exact Matches | 1,476 |
| Exact Match Accuracy | **97.49%** |
| Average Cosine Similarity | **0.9999** |
| Required Minimum Similarity | 0.60 |

The model significantly exceeds the required average cosine similarity threshold.

---

## 🔍 Evaluation Method

### Exact Match Accuracy

Exact match accuracy measures the percentage of generated sentences that are identical to the expected Shakespearean output.

    Exact Match Accuracy = Exact Matches / Total Test Samples

The model achieved:

    1,476 / 1,514 = 97.49%

### Cosine Similarity

Cosine similarity measures the semantic similarity between the expected and predicted outputs.

The assignment requires:

    Minimum Similarity = 0.60

The model achieved:

    Average Cosine Similarity = 0.9999

Therefore, the model substantially exceeds the required evaluation threshold.

---

## 📚 Learned Shakespearean Transformations

The model learned recurring transformations directly from the training examples.

| Modern English | Shakespearean Style |
|---|---|
| you | thou |
| are | art |
| has | hath |
| perhaps | perchance |
| before | ere |
| here | hither |
| give | bestow |
| beautiful | fair |
| brave | valiant |
| enemy | foe |
| body | flesh |
| happy | merry |
| friend | companion |
| old | aged |
| king | sovereign |
| listen | hark |
| think | ponder |
| kingdom | realm |
| light | radiance |
| very | most |

These transformations were learned by the neural network rather than manually programmed.

---

## 💻 Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Transformer Architecture
- Multi-Head Attention
- Jupyter Notebook
- Apple Silicon MPS

---

## 📁 Project Structure

```text
Shakespearean-English-using-Transformers/
│
├── Shakespearean English using Transformers.ipynb
│
├── shakespeare_dataset.csv
│
└── README.md
```

---

## ⚙️ Installation

Create or activate a Python environment:

    conda create -n mlproject python=3.12

Activate the environment:

    conda activate mlproject

Install the required libraries:

    pip install torch pandas numpy scikit-learn matplotlib

---

## ▶️ Running the Project

Clone the repository:

    git clone <YOUR_GITHUB_REPOSITORY_URL>

Navigate into the project directory:

    cd Shakespearean-English-using-Transformers

Start Jupyter Notebook:

    jupyter notebook

Open:

    Shakespearean English using Transformers.ipynb

Run the notebook cells sequentially.

---

## 🍎 Apple Silicon Support

This project was trained and tested on an Apple Silicon Mac using the PyTorch MPS backend.

The device selection follows:

    torch.backends.mps.is_available()

When MPS is available, the model uses:

    torch.device("mps")

This allows the Transformer to use Apple's GPU acceleration through Metal Performance Shaders.

A compatibility adjustment was also applied to disable the Transformer Encoder nested-tensor optimization because the relevant nested-tensor operation is not currently supported by the installed MPS backend.

---

## ⚠️ Limitations

Although the model achieved excellent performance on the provided dataset, several limitations remain.

### Word-Level Tokenization

The model uses word-level tokenization. Rare or unseen words may therefore be mapped to `<unk>`.

### Small Dataset

The dataset contains approximately 15,000 sentence pairs, which is relatively small for a general-purpose language transformation system.

### Greedy Decoding

Greedy decoding selects the highest-probability token at every step. This can sometimes produce suboptimal sequences or unnecessary additional words.

### Dataset-Specific Vocabulary

The model primarily learns the transformations represented in the training dataset and may not generalize to arbitrary Shakespearean writing.

---

## 🚀 Future Improvements

Potential improvements include:

### 1. Beam Search

Use beam search instead of greedy decoding to explore multiple candidate sequences.

### 2. Subword Tokenization

Use BPE, WordPiece, or SentencePiece tokenization to handle rare and unseen words more effectively.

### 3. Larger Dataset

Train on a larger and more diverse Shakespearean corpus.

### 4. Pretrained Transformers

Fine-tune a pretrained Transformer language model for Shakespearean style transfer.

### 5. Learning Rate Scheduling

Use learning-rate schedulers to improve training stability and convergence.

### 6. Label Smoothing

Apply label smoothing to improve model generalization.

### 7. Advanced Evaluation

Add BLEU, ROUGE, METEOR, and embedding-based semantic evaluation.

### 8. Attention Visualization

Visualize attention weights to understand relationships between modern and Shakespearean words.

### 9. Interactive Web Application

Deploy the trained model as an interactive web application where users can enter modern English and receive Shakespearean-style output in real time.

---

## 🏆 Final Results

The Transformer successfully learned to transform modern English sentences into Shakespearean-style English.

### Final Performance

    Test Samples          : 1,514
    Exact Matches         : 1,476
    Exact Match Accuracy  : 97.49%
    Cosine Similarity     : 0.9999
    Required Similarity   : 0.60

The achieved cosine similarity is significantly higher than the required threshold, demonstrating strong performance on the held-out test set.

---

## 📝 Conclusion

This project demonstrates how a Transformer Encoder-Decoder architecture can learn a language-style transformation task directly from paired examples.

The model successfully learned grammatical and vocabulary transformations characteristic of Shakespearean-style English and achieved **97.49% exact-match accuracy** with an **average cosine similarity of 0.9999** on 1,514 unseen test samples.

The project provides a practical example of applying sequence-to-sequence Transformers to natural language style transformation.

---

## 👨‍💻 Author

**Ebbeju Lankapalli**

B.Tech Computer Science and Engineering — Artificial Intelligence & Machine Learning

GitHub: `Ebbeju-Lankapalli`
