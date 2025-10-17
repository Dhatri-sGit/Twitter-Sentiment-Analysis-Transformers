# 💬 Twitter Sentiment Analysis using Transformers

### 🧠 Project Overview  
This project focuses on analyzing **Twitter sentiments** using both **traditional machine learning** and **Transformer-based deep learning** models.  
The goal is to classify tweets into **Positive**, **Negative**, and **Neutral** sentiments, highlighting how **DistilBERT**, a state-of-the-art Transformer model, outperforms classical approaches like **TF-IDF + Logistic Regression**.

---

## 🚀 Highlights
- Built an **end-to-end NLP pipeline** covering data preprocessing, feature engineering, model training, and evaluation.  
- Implemented a **baseline model** using TF-IDF vectorization and Logistic Regression.  
- Fine-tuned **DistilBERT** (Hugging Face Transformers) for contextual sentiment understanding.  
- Compared models based on **Accuracy**, **F1-Macro**, and **F1-Weighted** metrics.  
- Visualized model performance using **confusion matrices** and **classification reports**.

---

## 📈 Key Results
|               Model              | Accuracy | F1-Score | Insight  |
|:------                           |:---------|:---------|:---------|
| **TF-IDF + Logistic Regression** | ~83%     | ~0.81    | Performs well on general sentiment patterns. |
| **DistilBERT Transformer**       | ~89%     | ~0.88    | Better captures context, negation, and sarcasm. |

✅ **Outcome:** Fine-tuned Transformer achieved ~6% higher accuracy, showing the power of contextual understanding in modern NLP.

---

## 🧰 Tools & Libraries
- **Languages:** Python  
- **Libraries:** Scikit-learn · Transformers · PyTorch · Datasets · Evaluate · Pandas · Matplotlib  
- **Environment:** Jupyter Notebook  

---

## 💡 Business Impact
By leveraging deep learning for sentiment analysis, organizations can:  
- 📊 **Monitor brand reputation** through social-media insights.  
- 🤖 **Automate sentiment detection** at scale.  
- 🗣️ **Enhance customer experience** by identifying negative trends early.  

---

## 🔮 Future Scope
- Fine-tune larger models (BERT, RoBERTa, DeBERTa).  
- Deploy the model via **Flask** or **Streamlit** for real-time analysis.  
- Integrate with **Twitter API (X API)** for live data monitoring.  
- Add **explainability** using SHAP or LIME for token-level interpretation.  

---

## 📥 Dataset (Kaggle Link)
The full dataset is large and not stored in this repo. You can download it from Kaggle:

[Twitter Entity Sentiment Analysis Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)  
It includes:
- `twitter_training.csv`
- `twitter_validation.csv`

---

## 👩‍💻 Author
**Dhatri Shree Podugu**  
*M.S. in Information Systems — University of Maryland, Baltimore County*  

---

### 🌟 “Transforming tweets into actionable insights with NLP and Machine Learning.”
