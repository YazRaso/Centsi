import os
from pathlib import Path
import sys
from streamlit import secrets

# Try to import dependencies with better error handling
try:
    from dotenv import load_dotenv
except ImportError:
    print("dotenv not installed. Please install with: pip install python-dotenv")


    # Define a simple fallback
    def load_dotenv():
        print("Warning: python-dotenv not available, using environment variables only")
        return False

try:
    import google.generativeai as genai
except ImportError:
    print("Google Generative AI package not installed. Install with: pip install google-generativeai")


def get_sentiment_pipeline():
    """Initialize and return the sentiment analysis pipeline from transformers"""
    try:
        from transformers import pipeline
        # Explicitly set a model for consistent results
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=False
        )
        print("Successfully loaded sentiment analysis pipeline")
        return sentiment_analyzer
    except ImportError:
        print("Transformers library not installed. Install with: pip install transformers")
        return None
    except Exception as e:
        print(f"Error loading transformers pipeline: {str(e)}")
        return None



def get_api_key():
    """Attempt multiple methods to get the API key"""
    # First, try direct environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if "GOOGLE_API_KEY" in secrets:
        return secrets["GOOGLE_API_KEY"]
    elif api_key:
        print("Found API key in environment variables")
        return api_key

    # Next, try loading from .env file in current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, '.env')

    if os.path.exists(env_path):
        print(f"Found .env file at {env_path}")
        load_dotenv(env_path)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            print("Successfully loaded API key from .env file")
            return api_key
    else:
        print(f"No .env file found at {env_path}")

    # Try parent directory
    parent_dir = os.path.dirname(current_dir)
    parent_env_path = os.path.join(parent_dir, '.env')

    if os.path.exists(parent_env_path):
        print(f"Found .env file in parent directory: {parent_env_path}")
        load_dotenv(parent_env_path)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            print("Successfully loaded API key from parent directory .env file")
            return api_key

    # Final attempt - check project root (where streamlit is run from)
    try:
        import streamlit as st
        streamlit_dir = os.path.dirname(os.path.abspath(st.__file__))
        streamlit_env_path = os.path.join(streamlit_dir, '.env')
        if os.path.exists(streamlit_env_path):
            print(f"Found .env file in Streamlit directory: {streamlit_env_path}")
            load_dotenv(streamlit_env_path)
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key:
                return api_key
    except:
        pass

    return None


def public_sentiment():
    """Fetch current economic sentiment from Google's Gemini model"""
    api_key = get_api_key()

    if not api_key:
        print("No API key found after trying multiple locations")
        return "No API key found. Economic sentiment unavailable."

    prompt = "Summarize the current state of the global economy and general public sentiment in one to two sentences."

    try:
        # Configure the generative AI client
        genai.configure(api_key=api_key)

        # Create a model object
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Generate the content
        response = model.generate_content(prompt)

        # Return the text content
        return response.text
    except Exception as e:
        print(f"Error in public_sentiment(): {e}")
        return f"Error fetching sentiment: {str(e)}"


def sentiment_analysis():
    """Analyzes the current market sentiments with NLP"""
    try:
        # Get economic sentiment text
        text = public_sentiment()

        # If we're using a string error message, return a default sentiment
        if text.startswith("Error") or text.startswith("No API key"):
            return {"label": "NEUTRAL", "score": 0.5, "message": text}

        # Try to use transformers, but have a fallback
        sentiment_pipeline = get_sentiment_pipeline()

        if sentiment_pipeline is None:
            # Fallback: Simple rule-based sentiment analysis
            positive_words = ["growth", "positive", "increasing", "recovery", "optimistic",
                              "bullish", "confident", "strong", "robust", "improving"]
            negative_words = ["recession", "decline", "crisis", "negative", "bearish",
                              "downturn", "pessimistic", "weak", "struggling", "inflation"]

            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)

            if pos_count > neg_count:
                return {"label": "POSITIVE", "score": 0.5 + (pos_count / (pos_count + neg_count + 1)) * 0.5,
                        "message": text}
            elif neg_count > pos_count:
                return {"label": "NEGATIVE", "score": 0.5 + (neg_count / (pos_count + neg_count + 1)) * 0.5,
                        "message": text}
            return {"label": "NEUTRAL", "score": 0.5, "message": text}

        # If transformers is available, use it
        result = sentiment_pipeline(text)

        # Return the first result with the original text
        result_with_text = result[0].copy() if result else {"label": "NEUTRAL", "score": 0.5}
        result_with_text["message"] = text
        return result_with_text

    except Exception as e:
        return {"label": "NEUTRAL", "score": 0.5, "message": f"Sentiment analysis failed: {str(e)}"}
