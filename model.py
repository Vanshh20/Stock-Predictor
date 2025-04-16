import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime
import requests
from newsdataapi import NewsDataApiClient

ticker_df = pd.read_csv('Yahoo Ticker Symbols.csv')

ticker_mapping = dict(zip(ticker_df['Ticker'], ticker_df['Name']))
# Define candlestick patterns with the required parameters
patterns = {
    'Head and Shoulders': lambda x: (x['High'] > x['High'].shift(1)) & (x['High'] > x['High'].shift(-1)) &
                                    (x['High'].shift(-1) > x['High'].shift(2)) &
                                    (x['High'] > x['High'].shift(2)),
    'Inverse Head and Shoulders': lambda x: (x['Low'] < x['Low'].shift(1)) & (x['Low'] < x['Low'].shift(-1)) &
                                           (x['Low'].shift(-1) < x['Low'].shift(2)) &
                                           (x['Low'] < x['Low'].shift(2)),
    'Double Top': lambda x: (x['High'] > x['High'].shift(1)) & (x['High'] > x['High'].shift(-1)) &
                            (x['High'].shift(-1) > x['High'].shift(2)) &
                            (x['High'] > x['High'].shift(2)),
    'Double Bottom': lambda x: (x['Low'] < x['Low'].shift(1)) & (x['Low'] < x['Low'].shift(-1)) &
                               (x['Low'].shift(-1) < x['Low'].shift(2)) &
                               (x['Low'] < x['Low'].shift(2)),
    'Rising Wedge': lambda x: (x['High'] > x['High'].shift(1)) & (x['Low'] > x['Low'].shift(1)) &
                              (x['High'] < x['High'].shift(-1)) & (x['Low'] < x['Low'].shift(-1)),
    'Falling Wedge': lambda x: (x['High'] < x['High'].shift(1)) & (x['Low'] < x['Low'].shift(1)) &
                               (x['High'] > x['High'].shift(-1)) & (x['Low'] > x['Low'].shift(-1)),
    'Bullish Pennant': lambda x: (x['High'] > x['High'].shift(1)) & (x['Low'] > x['Low'].shift(1)) &
                                 (x['High'] < x['High'].shift(-1)) & (x['Low'] < x['Low'].shift(-1)),
    'Bearish Pennant': lambda x: (x['High'] < x['High'].shift(1)) & (x['Low'] < x['Low'].shift(1)) &
                                 (x['High'] > x['High'].shift(-1)) & (x['Low'] > x['Low'].shift(-1)),
    'Bullish Flag': lambda x: (x['Close'] > x['Open']) & (x['Close'].shift(-1) > x['Open'].shift(-1)),
    'Bearish Flag': lambda x: (x['Close'] < x['Open']) & (x['Close'].shift(-1) < x['Open'].shift(-1)),
    'Cup and Handle': lambda x: (x['Close'] > x['Close'].shift(1)) & (x['Close'] > x['Close'].shift(-1)) &
                                (x['Low'].shift(1) > x['Low'].shift(2)),
    'Descending Triangle': lambda x: (x['Low'] < x['Low'].shift(1)) & (x['Low'] < x['Low'].shift(-1)) &
                                     (x['High'].shift(1) < x['High'].shift(2))
}

def detect_candlestick_patterns(data):
    for pattern, func in patterns.items():
        data[pattern] = func(data)
    return data

def filter_significant_patterns(data, pattern, threshold=7):
    window_size = 20
    filtered = []
    for i in range(len(data)):
        if i < window_size:
            filtered.append(False)
        else:
            count = sum(data[pattern][i-window_size:i])
            if count >= threshold:
                filtered.append(True)
            else:
                filtered.append(False)
    return filtered

def add_technical_indicators(data):
    data['20_MA'] = data['Close'].rolling(window=20).mean()
    data['50_MA'] = data['Close'].rolling(window=50).mean()

    data = detect_candlestick_patterns(data)

    for pattern in patterns.keys():
        data[pattern] = filter_significant_patterns(data, pattern)

    support, resistance = compute_support_resistance(data)
    data['Support'] = support
    data['Resistance'] = resistance

    return data


def compute_support_resistance(data, window=100):
    support = []
    resistance = []

    for i in range(len(data)):
        if i < window:
            support.append(np.nan)
            resistance.append(np.nan)
        else:
            low = data['Low'][i - window:i].min()  # Already a float, no .iloc needed
            high = data['High'][i - window:i].max()  # Already a float, no .iloc needed
            support.append(low)
            resistance.append(high)

    return support, resistance


def get_news_sentiment(ticker, start_date, end_date):
    analyzer = SentimentIntensityAnalyzer()
    api_key = "your NewsDataApi key"

    # Initialize the newsdataapi client
    api = NewsDataApiClient(apikey=api_key)

    # Get the company name for the given ticker symbol
    company_name = ticker_mapping.get(ticker, None)

    # Check if the company name is found
    if not company_name:
        print(f"Company name not found for ticker {ticker}. Using ticker symbol instead.")
        company_name = ticker

    print(f"Fetching news for {company_name}...")  # Debug print to ensure correct company name is used

    # Fetch news data
    response = api.news_api(q=company_name, language='en')

    if response.get('status') != 'success':
        print(f"Failed to retrieve data: {response}")
        return pd.DataFrame(columns=['Date', 'Sentiment'])

    sentiments = []
    if 'results' in response:
        for article in response['results']:
            if 'title' in article and 'pubDate' in article:
                title = article['title']
                pub_date = article['pubDate']
                sentiment_score = analyzer.polarity_scores(title)['compound']
                date = pub_date.split(' ')[0]
                sentiments.append((date, sentiment_score))

                # Print the news title and its publication date
                print(f"Date: {date}, Title: {title}, Sentiment Score: {sentiment_score}")
    else:
        print("No results found in the news data")

    sentiment_df = pd.DataFrame(sentiments, columns=['Date', 'Sentiment'])
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    if sentiments:
        sentiment_df = pd.DataFrame(sentiments, columns=['Date', 'Sentiment'])
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    else:
        # Return empty DataFrame with correct columns
        sentiment_df = pd.DataFrame(columns=['Date', 'Sentiment'])

    return sentiment_df



# Step 3: Preprocessing the Data
def flatten_columns(df):
    # For each column tuple, join the non-empty parts with an underscore.
    df.columns = [
        '_'.join([str(item) for item in col if item])
        if isinstance(col, tuple) else col
        for col in df.columns.values
    ]
    return df


def preprocess_data(data, sentiment_df):
    # Reset index (keeps Date as a column)
    data = data.reset_index()
    sentiment_df = sentiment_df.reset_index(drop=True)

    # Flatten the MultiIndex columns to single-level columns
    data = flatten_columns(data)


    # Convert 'Date' to datetime in both DataFrames
    data['Date'] = pd.to_datetime(data['Date'])
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

    if sentiment_df.empty:
        data['Sentiment'] = 0
    else:
        data = pd.merge(
            data,
            sentiment_df,
            on='Date',
            how='left'
        ).fillna(0)

    # Now define your feature columns
    # Note: Use the new flattened column names.
    feature_columns = [
        'Close', '20_MA', '50_MA',
        'Sentiment', 'Support', 'Resistance'
    ]

    # Add technical indicator pattern columns if they exist (these were created by your functions)
    pattern_columns = [col for col in data.columns if col in patterns.keys()]
    feature_columns += pattern_columns

    # For debugging, you could print the data types:
    print(data[feature_columns].dtypes)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature_columns])
    return scaled_data, scaler


# Step 4: Creating the LSTM Model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 5: Training the Model
def train_model(model, train_data):
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], train_data.shape[1]))
    print("Training started...")
    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)
    print("Training finished!")

# Step 6: Making Predictions
def predict_stock_price(model, data, scaler, prediction_days):
    predictions = []
    for i in range(60, len(data)):
        test_data = data[i-60:i]
        test_data = np.array(test_data)
        test_data = np.reshape(test_data, (1, test_data.shape[0], test_data.shape[1]))
        prediction = model.predict(test_data)
        prediction_full = np.concatenate((prediction, np.zeros((prediction.shape[0], test_data.shape[2] - 1))), axis=1)
        prediction_full_rescaled = scaler.inverse_transform(prediction_full)
        predictions.append(prediction_full_rescaled[0, 0])

    extended_predictions = list(data[:60, 0]) + predictions

    # Forecast future prices
    for _ in range(prediction_days):
        test_data = data[-60:]
        test_data = np.array(test_data)
        test_data = np.reshape(test_data, (1, test_data.shape[0], test_data.shape[1]))
        prediction = model.predict(test_data)
        prediction_full = np.concatenate((prediction, np.zeros((prediction.shape[0], test_data.shape[2] - 1))), axis=1)
        prediction_full_rescaled = scaler.inverse_transform(prediction_full)
        next_predicted_price = prediction_full_rescaled[0, 0]
        extended_predictions.append(next_predicted_price)
        new_entry = np.append(prediction, test_data[0, -1, 1:])
        data = np.append(data, [new_entry], axis=0)

    return extended_predictions
