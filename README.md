# Movie Genre Classification

**Predict a movie's genre from its plot summary — three models side by side, from TF-IDF + linear classifiers up to a fine-tuned DistilBERT.**

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Transformers](https://img.shields.io/badge/Transformers-4.44.2-FFD21E?style=flat&logo=huggingface&logoColor=black) ![NLTK](https://img.shields.io/badge/NLTK-3.9-154F5B?style=flat&logo=python&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-2.2.3-150458?style=flat&logo=pandas&logoColor=white)

## Overview

This is a text-classification project that takes a movie plot description and predicts its genre — `drama`, `comedy`, `thriller`, `sci-fi`, and so on. It was built as a GrowthLink assignment, and the point of it was to compare a classic ML approach against a transformer on the same problem and the same data.

The dataset is the FU-Berlin movie database export (the IMDb genre dataset you'll see on Kaggle): 54,214 labeled training rows in a simple `ID ::: TITLE ::: GENRE ::: DESCRIPTION` format, spread across 27 genres. The class balance is rough — `drama` (13,613) and `documentary` (13,096) together make up roughly half the data, while `war` sits at 132 examples — which is part of what makes the comparison interesting.

Three models get trained on it: Logistic Regression and a linear SVM, both on TF-IDF features, plus a fine-tuned DistilBERT for sequence classification. The repo also ships a standalone EDA script that produces seven plots describing the data before any model touches it.

## Key Features

- **End-to-end pipeline** split into three runnable stages: preprocess → train → evaluate, each in its own module under `src/`.
- **Custom parser** for the `:::`-delimited text format, handling both labeled train rows (4 fields) and unlabeled test rows (2-3 fields), with logging for malformed lines instead of crashing.
- **Text cleaning** with NLTK: lowercasing, non-alphabetic stripping, tokenization, English stopword removal, and WordNet lemmatization.
- **TF-IDF features** built with unigrams and bigrams, capped at 5,000 features.
- **Per-plot sentiment** scored with TextBlob and carried through as an extra signal / EDA dimension.
- **Three trained models** persisted to disk: `logreg_model.joblib`, `svm_model.joblib`, and a full DistilBERT directory (weights, tokenizer, label encoder).
- **DistilBERT fine-tuning** through the Hugging Face `Trainer` — 3 epochs, batch size 8, 512-token truncation, warmup and weight decay configured.
- **Batched GPU/CPU inference** for the transformer — it moves to CUDA automatically when available and falls back to CPU otherwise.
- **Seven EDA visualizations** generated and saved as 300-DPI PNGs: genre distribution, plot-length distribution, genre co-occurrence heatmap, sentiment distribution, plot length by genre, sentiment by genre, and genre frequency over time (year parsed out of the title).
- **CSV prediction exports** per model, each with `id`, `title`, `plot`, and `predicted_genre` columns.

## How It Works

The project is three scripts that hand off to each other, plus a separate EDA script. Everything keys off a `data/` folder of raw `.txt` files and writes artifacts into `models/` and `outputs/`.

### 1. Data loading and preprocessing (`src/preprocess.py`)

`load_data()` reads the train and test files line by line and splits each on the `:::` delimiter (with a regex that tolerates surrounding whitespace). Train rows expect four fields (`id`, `title`, `genre`, `plot`); test rows are treated as unlabeled and only need an id and a plot. Anything that doesn't parse cleanly is logged and skipped rather than killing the run.

`preprocess_data()` then:

- Cleans each plot with `clean_text()` — lowercase, drop everything that isn't a letter or space, tokenize with NLTK, remove English stopwords, and lemmatize with `WordNetLemmatizer`.
- Computes a TextBlob sentiment polarity per plot.
- Vectorizes the cleaned text with `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))`. The vectorizer is **fit on train and reused on test**, so the two feature spaces line up.

It returns the train/test DataFrames, the TF-IDF matrices, and the fitted vectorizer.

### 2. Training (`src/train.py`)

`main()` calls `preprocess_data()` and trains all three models on the same target column (`genre`):

- **Logistic Regression** — `LogisticRegression(max_iter=1000)` on the TF-IDF matrix.
- **SVM** — `SVC(kernel='linear', probability=True)`, also on TF-IDF.
- **DistilBERT** — labels are integer-encoded with a `LabelEncoder`, plots are tokenized by `distilbert-base-uncased` (truncation + padding, `max_length=512`), and `DistilBertForSequenceClassification` is fine-tuned with the Hugging Face `Trainer`. Training arguments: 3 epochs, batch size 8, 500 warmup steps, `weight_decay=0.01`, logging every 10 steps, per-epoch saving.

Each model is written out under `models/`. The DistilBERT directory also stores its label encoder and tokenizer so inference can reconstruct the exact label mapping. The saved config shows a standard DistilBERT backbone (6 layers, 12 attention heads, hidden dim 768) with a 27-class single-label classification head.

### 3. Evaluation / prediction (`src/evaluate.py`)

`main()` reloads the three saved models and runs them over the test set. The two scikit-learn models predict straight off the TF-IDF matrix. DistilBERT re-tokenizes the test plots, wraps them in a `DataLoader`, runs batched inference in `torch.no_grad()` on GPU if one is present, takes the argmax over the logits, and inverse-transforms the indices back to genre strings via the saved label encoder. Each model's results go to its own CSV in `outputs/` (`logreg_predictions.csv`, `svm_predictions.csv`, `distilbert_predictions.csv`).

### 4. Exploratory data analysis (`notebooks/EDA.py`)

A separate script that loads the training data and produces seven Matplotlib/Seaborn figures, all saved to `outputs/` at 300 DPI:

- Top-10 genre distribution bar chart
- Plot-length (word-count) histogram
- Genre co-occurrence heatmap (multi-label binarized, then a co-occurrence matrix)
- Sentiment-polarity histogram
- Plot length by top-10 genre (boxplot)
- Sentiment by top-10 genre (violin plot)
- Genre frequency over time — the release year is regex-extracted from the title (`Movie (2009)` → `2009`) and the top-5 genres are counted per year

## Results / Highlights

- **Dataset:** 54,214 labeled training movies across **27 genres**, with a long-tailed distribution — `drama` (13,613) and `documentary` (13,096) dominate, down to `war` at 132.
- **Three models trained and compared** on identical features/text: Logistic Regression, linear SVM, and fine-tuned DistilBERT (`distilbert-base-uncased`, 27-class head).
- **Saved artifacts** ready to load: ~1 MB LogReg model, ~39 MB SVM model, and a full DistilBERT checkpoint with tokenizer and label encoder.

> Note: this build doesn't record held-out accuracy / F1 numbers — it produces per-model prediction CSVs against the test set rather than a scored evaluation report. Adding a metrics step (the repo ships `test_data_solution.txt` with ground-truth labels) would be the obvious next addition.

## Tech Stack

- **Language:** Python 3.8+
- **Classic ML:** scikit-learn 1.5.2 (TF-IDF, Logistic Regression, linear SVM, `LabelEncoder`, `MultiLabelBinarizer`)
- **Deep learning / NLP:** PyTorch 2.4.1, Hugging Face Transformers 4.44.2 (DistilBERT), NLTK 3.9, spaCy 3.7, TextBlob
- **Data / viz:** pandas 2.2, NumPy 1.26, Matplotlib 3.9, Seaborn 0.13, WordCloud
- **Utilities:** joblib (model persistence), SHAP

## Getting Started

### Prerequisites

- Python 3.8 or higher
- The data files in `data/`: `train_data.txt` and `test_data.txt`, UTF-8 encoded, in `ID ::: TITLE ::: GENRE ::: DESCRIPTION` format (test rows omit the genre). Original source: `ftp://ftp.fu-berlin.de/pub/misc/movies/database/`
- A GPU is optional but makes DistilBERT training and inference much faster

### Installation

```bash
git clone https://github.com/DCode-v05/Movie-Genre-Classification.git
cd Movie-Genre-Classification
pip install -r requirements.txt
```

The first run downloads the NLTK resources it needs (`punkt`, `punkt_tab`, `stopwords`, `wordnet`) automatically.

## Usage

Run the three stages in order. EDA is independent and can be run any time.

```bash
# 1. Generate the EDA plots into outputs/
python notebooks/EDA.py

# 2. Train all three models into models/
cd src
python train.py

# 3. Predict on the test set -> outputs/*_predictions.csv
python evaluate.py
```

After training you'll have `models/logreg_model.joblib`, `models/svm_model.joblib`, and `models/distilbert_model/`. After evaluation, each model writes a prediction CSV (`id`, `title`, `plot`, `predicted_genre`) into `outputs/`.

Note that `train.py` and `evaluate.py` import `preprocess` directly, so they expect to be run from inside `src/`, while `EDA.py` imports `src.preprocess` and is run from the project root.

## Project Structure

```
Movie-Genre-Classification/
├── data/
│   ├── train_data.txt              # 54,214 labeled rows (ID ::: TITLE ::: GENRE ::: DESCRIPTION)
│   ├── test_data.txt               # unlabeled rows for prediction
│   ├── test_data_solution.txt      # test rows with ground-truth genre
│   └── description.txt             # data format + source notes
├── src/
│   ├── preprocess.py               # parsing, text cleaning, sentiment, TF-IDF
│   ├── train.py                    # trains LogReg, SVM, DistilBERT
│   └── evaluate.py                 # loads models, predicts, writes CSVs
├── notebooks/
│   └── EDA.py                      # seven Matplotlib/Seaborn plots
├── models/
│   ├── logreg_model.joblib         # Logistic Regression on TF-IDF
│   ├── svm_model.joblib            # linear SVM on TF-IDF
│   └── distilbert_model/           # weights, config, tokenizer, label encoder
├── outputs/                        # *_predictions.csv + EDA *.png
├── requirements.txt
└── README.md
```

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
