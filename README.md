💬 Twitter Sentiment Analysis using Transformers

This project applies Machine Learning (ML) and Natural Language Processing (NLP) techniques to analyze sentiments in tweets — classifying them as Positive, Negative, or Neutral.
It demonstrates how Transformer models like DistilBERT outperform traditional ML models such as TF-IDF + Logistic Regression in understanding real-world text data.

🚀 Project Highlights

Built an end-to-end NLP pipeline — data preprocessing, feature engineering, model training, and evaluation.

Implemented a baseline model using TF-IDF and Logistic Regression for sentiment prediction.

Fine-tuned DistilBERT (a Transformer from Hugging Face) for contextual tweet understanding.

Compared both models using key metrics: Accuracy, F1-Macro, and F1-Weighted.

Visualized confusion matrices and performance reports to assess prediction quality.

📈 Key Results
Model	Accuracy	F1-Score	Insight
TF-IDF + Logistic Regression	~83%	~0.81	Strong baseline for simple text features
DistilBERT Transformer	~89%	~0.88	Captures deeper context and sarcasm in tweets

Outcome: Fine-tuned Transformer achieved ~6% higher accuracy, showcasing its power in sentiment understanding.

🧰 Tools & Libraries

Python · Scikit-learn · PyTorch · Transformers (Hugging Face) · Pandas · Matplotlib

💡 Business Impact

By leveraging modern NLP, this project demonstrates how organizations can:

Monitor brand reputation through real-time Twitter feedback.

Improve customer experience by detecting negative sentiment early.

Automate social media insights using Transformer-based models.
