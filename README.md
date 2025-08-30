
# 🛍️ Amazon Reviews Sentiment & Emotion Analysis

This repository contains a **Natural Language Processing (NLP)** and **Machine Learning (ML)** project that analyzes Amazon customer reviews. The goal is twofold:

1. **Sentiment Classification** → Detect whether reviews are **Positive, Neutral, or Negative**
2. **Emotion Detection** → Identify deeper emotional signals such as **joy, anger, sadness, trust, surprise, etc.**

By combining **lexicon-based methods** and **supervised ML models**, this project extracts insights that can guide **marketing strategies, product improvements, and customer experience design**.

---

## 📌 Project Objectives

✔️ Clean and preprocess raw customer reviews

✔️ Explore data visually (EDA) to find review trends

✔️ Perform **sentiment analysis** using both **TextBlob (lexicon)** and **ML classifiers**

✔️ Apply **NRC Emotion Lexicon** to detect customer emotions

✔️ Compare model performance across multiple algorithms

✔️ Generate **business insights** to support decision-making

---

## ⚙️ Tech Stack

**Languages & Tools**

* Python 3.x
* Jupyter Notebook

**Libraries Used**

* **Data Handling** → `pandas`, `numpy`
* **Visualization** → `matplotlib`, `seaborn`, `wordcloud`
* **NLP** → `nltk`, `re`, `textblob`
* **ML Models** → `scikit-learn` (Logistic Regression, Naive Bayes, Random Forest, etc.)
* **Emotion Detection** → NRC Emotion Lexicon

---

## 📊 Workflow

### 1️⃣ Data Collection & Preprocessing

* Dataset: Amazon reviews (CSV format)
* Cleaning steps included:

  * Removing punctuation, numbers, special characters
  * Lowercasing text
  * Removing stopwords
  * Tokenization & Lemmatization

👉 Example:
**Raw Review:**
`"I love Amazon! Great prices but delivery was late 😡"`
**Cleaned Review:**
`love amazon great price delivery late`

---

### 2️⃣ Exploratory Data Analysis (EDA)

* Distribution of ratings and sentiment
* Word frequency analysis
* WordClouds for positive vs. negative reviews

📌 Example Plots:

* **Countplot of Sentiments**
* **Top 20 Frequent Words**
* **WordCloud (Positive vs. Negative)**

---

### 3️⃣ Sentiment Analysis

#### Lexicon-based:

* Used **TextBlob** polarity & subjectivity scores
* Mapped polarity → Negative, Neutral, Positive

---
Business insights: 

📊 Sentiment Distribution (%): {'Negative': 52.3, 'Positive': 33.1, 'Neutral': 14.6}

📊 Emotion Distribution (%): {'anger': 40.2, 'joy': 25.4, 'sadness': 15.7, 'trust': 9.8, 'anticipation': 8.9}

❌ Top Negative Review Words: [('late', 120), ('delivery', 115), ('refund', 90), ('customer', 88), ('service', 80)]

✅ Top Positive Review Words: [('fast', 130), ('prime', 115), ('love', 100), ('easy', 95), ('trust', 90)]


📌 Recommended Actions:

- High anger detected: Improve customer service response and shipping reliability.
- Joy is common in positive reviews: Highlight happy customers in marketing campaigns.
- Sadness detected: Consider follow-up support for unsatisfied customers.
- Trust appears in reviews: Strengthen brand image by showing reliability guarantees.

  ---


#### Machine Learning Models:

* Vectorization (TF-IDF, CountVectorizer)
* Models trained:

  * Logistic Regression
* Metrics evaluated:

  * Accuracy
  * Precision, Recall, F1-score
  * Confusion Matrix

📊 **Confusion Matrix (Logistic Regression)**


| Actual / Predicted | Negative | Neutral | Positive |
| ------------------ | -------- | ------- | -------- |
| **Negative**       | 2811     | 4       | 55       |
| **Neutral**        | 111      | 3       | 63       |
| **Positive**       | 129      | 5       | 1030     |


---


### 4️⃣ Emotion Detection

* Used **NRC Emotion Lexicon** (Joy, Sadness, Anger, Fear, Trust, Disgust, Surprise, Anticipation)
* Mapped reviews to emotions
* Visualized distribution of emotions

📌 Example Insight:

* **Anger & Sadness** → Linked to shipping delays & product defects
* **Joy** → Linked to Prime benefits & discounts
* **Trust** → Linked to consistent product quality

---

### 5️⃣ Business Insights

Based on results:

* **Marketing**: Highlight Prime convenience & reliability to reinforce **joy/trust** emotions.
* **Product Development**: Address recurring **delivery delay complaints** to reduce **anger/sadness**.
* **Customer Service**: Proactively detect frustration and target support resources.
* **Social Insights**: Emotion analysis helps track **brand perception** over time.


## 📈 Results Overview

✔️ Model performs very well for Negative and Positive reviews
⚠️ Performance is weaker on Neutral reviews, likely due to limited training data

✔️ Lexicon-based methods provided quick baseline results

✔️ Emotion analysis added depth to simple sentiment labels

✔️ Business recommendations derived directly from data

---

## 📌 Future Work

* Deploy a **Streamlit dashboard** for real-time review analysis
* Add **multilingual support** for global markets
* Use **Deep Learning models (BERT, LSTMs)** for better accuracy
* Integrate into **customer service platforms** for proactive insights





---


