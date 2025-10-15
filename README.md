# Sentiment Analysis (Bahasa Indonesia)

Project ini adalah implementasi **Analisis Sentimen pada tweet berbahasa Indonesia** (notebook: `Sentiment Analysis (Bahasa Indonesia) — Versi Tuned (Per Bagian).ipynb`). Notebook berisi seluruh pipeline: preprocessing teks, feature extraction (word + char TF‑IDF), training (Logistic Regression & Multinomial NB), hyperparameter tuning dengan `RandomizedSearchCV`, serta evaluasi dan contoh prediksi.

---

## 🎯 Ringkasan
- Format: Jupyter Notebook (dirancang untuk Google Colab / lokal).  
- Dataset: `tweet.csv` (letakkan di folder yang sama atau set `CSV_PATH` / `DATA_PATH`).  
- Metode: TF‑IDF (word + char) → model ML klasik (LogisticRegression, MultinomialNB) → RandomizedSearchCV (scoring: `f1_macro`).

---

## 📁 Struktur penting
- `Sentiment Analysis (Bahasa Indonesia) — Versi Tuned (Per Bagian).ipynb` — notebook utama.  
- `tweet.csv` — dataset yang digunakan (harus berisi kolom teks & label).  

> Notebook menyimpan artefak (model & vectorizer) ke `OUTPUT_DIR` (default di notebook: `/content/drive/My Drive/Proyek/OutputSentimentNew`) dan membuat `artefak_model.zip`.

---

## 🔧 Persyaratan / Dependencies
Notebook berisi perintah instalasi (Colab):

```bash
# di Colab cell (notebook):
!pip -q install pandas scikit-learn matplotlib joblib nltk Sastrawi imbalanced-learn
```

Untuk lingkungan lokal, rekomendasi `requirements.txt` minimal:

```
pandas
numpy
scikit-learn
matplotlib
joblib
nltk
Sastrawi
imbalanced-learn
scipy
```

---

## 🚀 Cara menjalankan (ringkas)
### Di Google Colab
1. Buka notebook di Colab.  
2. Jalankan cell untuk mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Jalankan cell instalasi paket (atau pastikan environment Anda sudah memiliki paket yang diperlukan).  
4. Set `CSV_PATH` (folder) atau `DATA_PATH` langsung: contoh:
   ```python
   CSV_PATH = '/content/drive/My Drive/Proyek/dataset'
   DATA_PATH = os.path.join(CSV_PATH, 'tweet.csv')
   ```
5. Pastikan variabel `TEXT_COL` dan `LABEL_COL` cocok dengan nama kolom di `tweet.csv` (contoh: `TEXT_COL='tweet'`, `LABEL_COL='label'`).  
6. Jalankan sel notebook berurutan.


---

## 🔎 Isi notebook (step-by-step)
Notebook terstruktur; berikut step utama yang ada:
1. **Mount Google Drive** (opsional jika di Colab).  
2. **Instalasi paket / setup environment** (Colab cell tersedia).  
3. **Import & konfigurasi** (SEED, stopwords, stemmer Sastrawi, OUTPUT_DIR).  
4. **Load dataset** (`pd.read_csv(DATA_PATH)`) — pastikan kolom teks & label ada.  
5. **Preprocessing**: slang normalization, negation marking, hashtag extraction/penanganan, tokenisasi, stopword removal, Sastrawi stemmer.  
6. **Split**: stratified train/test split.  
7. **Vectorization**: Word TF‑IDF + Char TF‑IDF + horizontal stack (`scipy.sparse.hstack`).  
8. **Model selection**: `RandomizedSearchCV` untuk LogisticRegression & MultinomialNB (scoring `f1_macro`, CV = StratifiedKFold).  
9. **Retrain & evaluasi** di test set: `classification_report`, `confusion_matrix`, `f1_score (macro)`.  
10. **Simpan artefak**: `joblib.dump(best_model, model_path)` plus vectorizer (`tfidf_word.joblib`, `tfidf_char.joblib`), lalu zip semua artefak.  
11. **Contoh prediksi**: fungsi `predict_text(text)` untuk single sample dan contoh batch (CSV -> output CSV) di notebook.

---

