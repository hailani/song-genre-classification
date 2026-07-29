# Song Genre Classification Using Machine Learning

## Overview

This project uses natural language processing and machine learning to classify song lyrics into music genres. The goal was to evaluate whether lyrics alone could predict a song's genre and compare traditional machine learning models against a transformer-based model.

The project was completed in a Jupyter Notebook and includes data loading, exploratory data analysis, text cleaning, feature engineering, model training, and model evaluation.

## Dataset

This project uses the **Genius Song Lyrics with Language Information** dataset from Kaggle.

Dataset used in the notebook:

```text
carlosgdcj/genius-song-lyrics-with-language-information
```

The notebook downloads the dataset using `kagglehub`, so the full CSV file does **not** need to be uploaded to this GitHub repository.

The project focuses on English-language songs and removes the `misc` category to classify lyrics into five major genres:

- Rap
- Rock
- Pop
- Country
- R&B

> Note: Because this project uses real-world song lyrics, the dataset may contain explicit language.

## Project Workflow

### 1. Data Loading

The notebook downloads the Kaggle dataset using `kagglehub` and loads the song lyrics data into a Pandas DataFrame.

Key fields used include:

- `artist`
- `lyrics`
- `tag`
- `language`

### 2. Exploratory Data Analysis

The project explores:

- Genre distribution
- Lyric length
- Duplicate lyrics
- Class imbalance
- Outlier lyric lengths

### 3. Text Cleaning and Preprocessing

The lyrics are cleaned using several NLP preprocessing steps:

- Convert text to lowercase
- Remove lyric section annotations such as `[Chorus]`, `[Verse]`, and `[Bridge]`
- Remove punctuation and numbers
- Tokenize text
- Remove English stop words
- Lemmatize words
- Remove duplicate lyrics
- Remove extreme lyric-length outliers

### 4. Feature Engineering

The project uses `TfidfVectorizer` to convert cleaned lyrics into numerical features that machine-learning models can process.

TF-IDF settings include:

- Up to 50,000 features
- Unigrams and bigrams
- Minimum and maximum document frequency filtering
- Sublinear term frequency scaling

### 5. Model Training

The notebook trains and evaluates multiple models:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine
4. DistilBERT Transformer Model

### 6. Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- Weighted recall
- F1-score
- Classification reports
- Confusion matrices

## Results

| Model | Validation Accuracy |
|---|---:|
| DistilBERT Transformer | 89.72% |
| Linear SVM | 86.96% |
| Naive Bayes | 83.30% |
| Logistic Regression | 82.75% |

The DistilBERT transformer model produced the strongest overall accuracy, while Linear SVM performed best among the traditional machine-learning models.

## How to Run This Project

### Option 1: Run in Google Colab

1. Open the notebook in Google Colab.
2. Set the runtime to GPU for the transformer model:
   - `Runtime` → `Change runtime type` → `T4 GPU`
3. Install the main Python libraries:

```bash
pip install pandas numpy matplotlib nltk scikit-learn xgboost kagglehub transformers datasets torch wordcloud
```

4. Run the notebook cells from top to bottom.

### Option 2: Run Locally

Clone this repository:

```bash
git clone https://github.com/YOUR-USERNAME/song-genre-classification.git
cd song-genre-classification
```

Install the main Python libraries:

```bash
pip install pandas numpy matplotlib nltk scikit-learn xgboost kagglehub transformers datasets torch wordcloud
```

Open the notebook:

```bash
jupyter notebook song_genre_classification.ipynb
```
