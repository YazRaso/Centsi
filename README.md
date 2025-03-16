# CentSeek Documentation

## Overview

CentSeek is a financial analytics application designed to track and predict payment default risks based on customer payment history. The application utilizes XGBoost machine learning model to analyze patterns in payment behavior and determine the likelihood of default.

## Purpose

CentSeek helps financial institutions and creditors make informed decisions about credit issuance by:
- Analyzing historical payment patterns
- Providing risk assessments for potential defaults
- Visualizing customer payment behaviors
- Incorporating market sentiment analysis for broader economic context

## Technical Stack

- **Frontend**: Streamlit for interactive UI
- **Machine Learning**: XGBoost for default prediction
- **Data Visualization**: Plotly for graphical representation
- **Natural Language Processing**: Transformers for sentiment analysis
- **External API**: Google's Gemini model for economic sentiment analysis data

## Features

### 1. Payment Default Risk Assessment

CentSeek's core functionality is to assess the probability that a customer will default on payments. The application calculates this probability and categorizes risk levels:
- **Very likely to default**: 48% or higher probability
- **Likely to default**: 45-48% probability 
- **Moderate risk**: 20-45% probability
- **Very unlikely to default**: Below 20% probability

### 2. Interactive Visualizations

#### Risk Profile Radar Chart
Visualizes key risk factors for a customer in a radar chart format:
- Credit limit
- Average bill amount
- Average payment amount
- Maximum payment delay

The chart also provides a summary with detailed statistics and calculates a bill-to-payment ratio, which serves as an additional risk indicator.

#### Feature Importance Analysis
Displays the top 10 features that most significantly impact the default prediction model, helping users understand which factors drive the risk assessment.

### 3. Market Sentiment Analysis

Integrates economic sentiment analysis to provide context for default risk:
- Retrieves current economic sentiment using Google's Gemini model
- Analyzes sentiment with NLP techniques (positive, negative, or neutral)
- Displays sentiment score and analysis alongside prediction results

## How to Use

### Input Data

Users must provide the following information:

#### 1. Credit Limit
- The maximum amount of credit available to the customer

#### 2. Payment Delays (for past 7 months)
- Select 0 if payment was made on time
- Select 1-8 for the number of months payment was delayed
- Select 9 for delays of 9 or more months

#### 3. Bill Amounts (for past 6 months)
- The amount billed to the customer each month

#### 4. Payment Amounts (for past 6 months)
- The amount the customer paid each month

### Understanding Results

After submitting data, users can view:

1. **Default Probability**: Numeric probability with risk assessment
2. **Risk Profile**: Radar chart showing risk metrics and summary statistics
3. **Feature Importance**: Bar chart showing which factors are most important
4. **Market Sentiment**: Current economic sentiment analysis

## Technical Implementation Details

### Model

CentSeek uses an XGBoost model trained on payment history data. The model:
- Is optimized with hyperparameters for binary classification
- Uses GPU acceleration for performance
- Loads from the file "centseek_model.json"

### Sentiment Analysis

CentSeek offers two methods for sentiment analysis:
1. **Primary Method**: Uses Google's Gemini model through their Generative AI API
2. **Fallback Method**: Uses a rule-based approach when API access is unavailable

The sentiment analysis requires:
- Google API key (stored in environment variables or .env file)
- Transformers library (optional, with fallback implementation)

### Error Handling

The application includes robust error handling for:
- Model loading failures
- API connection issues
- Data processing errors
- Missing dependencies

## Installation Requirements

To run CentSeek, you need:

1. **Python Libraries**:
   - Please check requirements.txt for required libaries

2. **Files**:
   - centseek_model.json (in house developed model)
   - .env file (recommended, for API keys)

3. **API Keys**:
   - Google API key (for sentiment analysis data)

## Troubleshooting

Common issues and solutions:

1. **Model Loading Error**
   - Ensure "centseek_model.json" is in the correct directory
   - Check file permissions

2. **Sentiment Analysis Unavailable**
   - Verify Google API key is correctly set
   - Check internet connectivity
   - Ensure required libraries are installed

3. **Visualization Issues**
   - Ensure input data is in the correct format
   - Check for missing values in financial data

## Development

CentSeek was developed in house by the developer, made with expertise in financial risk modeling. The GitHub repository is available for further development and contributions:

[https://github.com/YazRaso/Centsi.git](https://github.com/YazRaso/Centsi.git)

## License
Software is licensed under the MIT license