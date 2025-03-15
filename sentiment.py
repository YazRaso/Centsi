from transformers import pipeline
import os
from dotenv import load_dotenv
from google import genai


def public_sentiment():
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    prompt = "Summarize the current state of the global economy and general public sentiment in one to two sentences."
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )
    return response


# Analyzes the current market sentiments with NLP
def sentiment_analysis():
    sentiment = pipeline("sentiment-analysis")
    result = sentiment(public_sentiment())
    return result[0]
