# review_sentiment_analyzer.py

import requests
from bs4 import BeautifulSoup
import joblib
import streamlit as st

# -------------------------------
# STEP 1: Load your trained sentiment model and vectorizer
# -------------------------------

# Load the Logistic Regression (or similar) model
model = joblib.load("twitter_sentiment_model_3class.sav")



# Load the vectorizer (e.g., TF-IDF)
vectorizer = joblib.load("tfidf_vectorizer_3class.sav")

# -------------------------------
# STEP 2: Function to scrape reviews from Amazon page
# -------------------------------

def extract_reviews_amazon(url):
    """
    Extracts reviews from an Amazon product review page.
    Each Amazon product page has a section with reviews (if visible in HTML).
    """
    # Automatically convert product page link to review page link
    if "/dp/" in url:
        url = url.replace("/dp/", "/product-reviews/")

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/142.0.0.0 Safari/537.36'
        )
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error("Failed to fetch the page. Check the URL or your internet connection.")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract review texts (Amazon’s HTML pattern may vary)
    review_spans = soup.select('.review-text-content span')
    reviews = [span.get_text(strip=True) for span in review_spans if span.get_text(strip=True)]

    return reviews

# -------------------------------
# STEP 3: Prediction function using your .sav model
# -------------------------------

def predict_sentiment(text):
    """Predict the sentiment of a single text input using the trained model."""
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return prediction

# -------------------------------
# STEP 4: Analyze all reviews from a URL
# -------------------------------

def analyze_reviews(url):
    reviews = extract_reviews_amazon(url)
    if not reviews:
        return None, None

    sentiments = [predict_sentiment(r) for r in reviews]

    # Count the number of each sentiment
    positive = sum(1 for s in sentiments if s.lower() == "positive")
    negative = sum(1 for s in sentiments if s.lower() == "negative")
    neutral = sum(1 for s in sentiments if s.lower() == "neutral")

    total = len(reviews)
    overall = max(
        [('positive', positive), ('negative', negative), ('neutral', neutral)],
        key=lambda x: x[1]
    )[0]

    summary = {
        "total_reviews": total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "overall_sentiment": overall
    }

    return summary, reviews

# -------------------------------
# STEP 5: Streamlit Web Interface
# -------------------------------

def main():
    st.title("🛍️ Product Review Sentiment Analyzer")
    st.markdown("Paste an **Amazon product review page link** below to analyze the overall sentiment using your trained model.")

    url = st.text_input("Enter Amazon review page URL:")

    if st.button("Analyze"):
        with st.spinner("Fetching and analyzing reviews..."):
            summary, reviews = analyze_reviews(url)

            if summary:
                st.success(f"✅ Overall Sentiment: **{summary['overall_sentiment'].capitalize()}**")
                st.write(f"**Total Reviews Analyzed:** {summary['total_reviews']}")
                st.write(f"**Positive:** {summary['positive']}")
                st.write(f"**Negative:** {summary['negative']}")
                st.write(f"**Neutral:** {summary['neutral']}")

                with st.expander("📋 View Sample Reviews"):
                    for r in reviews[:10]:
                        st.write("-", r)
            else:
                st.error("Could not extract any reviews from the provided link.")

if __name__ == "__main__":
    main()
