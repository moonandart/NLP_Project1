# 🇮🇩 Sentiment Analysis (Bahasa Indonesia) — Enhanced (Per Bagian)

This project implements a **sentiment analysis pipeline** in Indonesian, enhanced with several improvements for better handling of informal text.

## ✨ Features & Enhancements
The notebook combines the *baseline pipeline* with **three main improvements**:
1. **Normalization of abbreviations** (e.g., `gk → tidak`, `tp → tapi`, etc.)
2. **Emoji-to-word conversion** (🙂 → *senyum*, 😢 → *sedih*, etc.)
3. **Filtering of rare words** using `min_df` in TF-IDF (cleaner than manual removal)

### Optional Enhancements
You can enable additional features (currently commented out):
- **Character n-grams** → helps with typos and informal spelling
- **Stopword removal + Sastrawi stemming**

## 📂 Dataset
The dataset used is `tweet.csv`, containing tweets labeled for sentiment.

- **Source**: (describe if known)  
- **Structure**: Text + sentiment labels (positive, negative, neutral)

## ⚙️ Workflow Overview
1. **Data Preprocessing**  
   - Cleaning text  
   - Abbreviation expansion  
   - Emoji translation  
   - Optional stemming/stopword removal  

2. **Feature Extraction**  
   - TF-IDF (with `min_df` to remove very rare terms)  
   - Optional: Char n-grams  

3. **Model Training & Evaluation**  
   - Traditional ML classifiers (Naive Bayes, SVM, Logistic Regression, etc.)  
   - Performance evaluation with accuracy, precision, recall, F1  

## 📊 Results
- Enhanced pipeline shows improved performance on informal/abbreviated text.
- Filtering rare words reduces noise and improves generalization.

## 🚀 Usage
1. Clone this repo  
   ```bash
   git clone https://github.com/your-username/your-repo.git
   ```
2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook  
   ```bash
   jupyter notebook sentiment_id_enhanced_per_part.ipynb
   ```

## 🔮 Next Steps
- Try deep learning (LSTM, BERT-based Indo models)  
- Expand abbreviation and emoji dictionaries  
- Apply to larger social media datasets  
