# Rumor Detection Project

This repository contains code for detecting rumors in Chinese microblog (Weibo) texts using two different approaches:
- **Statistical Method** based on Naive Bayes and Jieba tokenizer.
- **Deep Learning Method** using PKUSeg for tokenization and models like TextCNN and LSTM.

## Folder Structure

```
├── .idea/                  # Project configuration files
├── .ipynb_checkpoints/     # Jupyter notebook checkpoint files
├── __pycache__/            # Compiled Python cache
├── data/                   # Contains original or preprocessed dataset
├── embeddings/             # Word vector files (excluded if larger than 100MB)
├── stopwords/              # Stopword lists used in tokenization
├── rumor_all_data.ipynb    # Data preprocessing and utility notebook
├── rumor_jieba_bayes.ipynb # Statistical method: Naive Bayes with Jieba
├── rumor_pkuseg_gensim.ipynb # Deep learning models: TextCNN & LSTM with PKUSeg
├── utils.py                # Helper functions
└── .gitattributes          # Git LFS settings (if large file tracking is used)
```

## Dependencies

Install the required packages using:

```
pip install -r requirements.txt
```

**Note:** You may need to install the following if not already included:
- `jieba`
- `pkuseg`
- `gensim`
- `scikit-learn`
- `tensorflow` 

## Running the Project

### 1. Statistical Method (Naive Bayes)

Run the following notebook:

```
rumor_jieba_bayes.ipynb
```

- This uses **Jieba** for tokenization.
- Classification is done using a **Naive Bayes classifier**.
- Outputs such as accuracy and classification reports are saved in the notebook.

### 2. Deep Learning Method (TextCNN & LSTM)

Run the notebook:

```
rumor_pkuseg_gensim.ipynb
```

- Two models are used: **TextCNN** and **LSTM**.
- Output results and metrics are stored in the notebook cells.
