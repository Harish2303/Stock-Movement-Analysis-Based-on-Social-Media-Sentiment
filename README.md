
# Twitter Sentiment and Stock Movement Analysis

This project evaluates the potential correlation between public sentiment on Twitter and fluctuations in the stock market. It fetches live Twitter data, performs sentiment analysis, and utilizes machine learning models to predict stock movements based on sentiment trends.

---

## Project Overview

### Objectives:
1. **Sentiment Analysis**: Classify tweets into positive, neutral, and negative sentiment categories.
2. **Stock Movement Prediction**: Use sentiment data to train a machine learning model predicting stock movements.
3. **Visualization**: Generate charts for sentiment trends over time and distribution among sentiment classes.

---

## Project Workflow

### 1. **Fetching Twitter Data**
   - Use the Tweepy library to fetch recent tweets with specific keywords.
   - Syntax: 
     ```python
     client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=["created_at", "text"])
     ```

### 2. **Data Cleaning**
   - Clean raw tweets by removing:
     - URLs
     - Mentions
     - Non-alphabet characters
   - Syntax:
     ```python
     re.sub(r"http\S+", "", text)  # Remove URLs
     re.sub(r"@\w+", "", text)    # Remove mentions
     ```

### 3. **Sentiment Analysis**
   - Apply VADER SentimentIntensityAnalyzer for compound sentiment scoring.
   - Syntax:
     ```python
     analyzer.polarity_scores(text)["compound"]
     ```

### 4. **Machine Learning Model**
   - Train a Random Forest Classifier for predicting stock movement.
   - Syntax:
     ```python
     model = RandomForestClassifier(random_state=42)
     model.fit(X_train, y_train)
     ```

### 5. **Visualization**
   - Generate pie charts for sentiment distribution and line plots for sentiment trends over time.
   - Syntax:
     ```python
     plt.figure(figsize=(6, 4))
     tweets_df["sentiment_label"].value_counts().plot(kind="pie", autopct='%1.1f%%')
     ```

---

## Setup Instructions

### Prerequisites
1. **Python Environment**: Ensure Python (3.8 or higher) is installed.
2. **Twitter API Credentials**: Obtain a Bearer Token by creating a developer account on the [Twitter Developer Platform](https://developer.twitter.com/).


## Outputs
- **Visuals**:
  - Sentiment Distribution: `sentiment_distribution.png`
  - Sentiment Trend Over Time: `sentiment_trend.png`
- **Reports**:
  - Classification performance metrics for stock movement prediction.

## Dependencies

### Mapped in `requirements.txt`
- **Tweepy**: For interacting with the Twitter API.
  ```bash
  pip install tweepy==4.14.0
  ```
- **Pandas**: Data manipulation and analysis.
  ```bash
  pip install pandas==2.0.3
  ```
- **NLTK**: Sentiment analysis tools.
  ```bash
  pip install nltk==3.8.1
  ```
- **Matplotlib & Seaborn**: Data visualization libraries.
  ```bash
  pip install matplotlib==3.7.2 seaborn==0.12.2
  ```
- **Scikit-learn**: Machine learning algorithms.
  ```bash
  pip install scikit-learn==1.3.0
  ```
- **Python-dotenv**: Manage environment variables.
  ```bash
  pip install python-dotenv==1.0.0
  ```
