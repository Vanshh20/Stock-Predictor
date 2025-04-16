from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import re
import csv
import os
import bcrypt
import pandas as pd
import datetime
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import model
from newsdataapi import NewsDataApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf


app = Flask(__name__)
app.secret_key = 'your_secret_key'
user_csv = 'users.csv'
ticker_df = pd.read_csv('Yahoo Ticker Symbols.csv')

ticker_mapping = dict(zip(ticker_df['Ticker'], ticker_df['Name']))

# Function to read users from CSV file
def read_users():
    users = {}
    if os.path.exists(user_csv):
        with open(user_csv, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:
                    users[row[0]] = row[1]
    return users

# Function to write a new user to the CSV file
def write_user(username, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    with open(user_csv, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([username, hashed_password.decode('utf-8')])

@app.route('/',methods=['GET', 'POST'])
def index():
    if 'username' in session:
        if request.method == 'POST':
            ticker = request.form['ticker']
            prediction_days = int(request.form['prediction_days'])
            return redirect(url_for('predict', ticker=ticker, prediction_days=prediction_days))
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/form_login', methods=['POST'])
def form_login():
    username = request.form['username']
    password = request.form['password']
    users = read_users()

    if username in users and bcrypt.checkpw(password.encode('utf-8'), users[username].encode('utf-8')):
        session['username'] = username
        return redirect(url_for('index'))
    else:
        flash('Invalid username or password.', 'danger')
        return redirect(url_for('login'))

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/form_register', methods=['POST'])
def form_register():
    username = request.form['username']
    password = request.form['password']
    confirm_password = request.form['confirm_password']
    users = read_users()

    if username in users:
        flash('Username already exists.', 'danger')
        return redirect(url_for('register_page'))

    if not re.match(r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\W).{8,}$', password):
        flash('Password must be at least 8 characters long, include an uppercase letter, a lowercase letter, and a special character.', 'danger')
        return redirect(url_for('register_page'))

    if password != confirm_password:
        flash('Passwords do not match.', 'danger')
        return redirect(url_for('register_page'))

    write_user(username, password)
    flash('Registration successful. Please log in.', 'success')
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/check_username', methods=['POST'])
def check_username():
    username = request.form['username']
    users = read_users()
    return jsonify({'exists': username in users})


@app.route('/ticker_search')
def ticker_search():
    query = request.args.get('query')
    matches = model.ticker_df[model.ticker_df['Ticker'].str.contains(query, case=False, na=False)]
    tickers = matches['Ticker'].tolist()
    return jsonify(tickers)

def fetch_recent_news(ticker):
    api_key = "your NewsDataApi key"
    api = NewsDataApiClient(apikey=api_key)
    company_name = ticker_mapping.get(ticker, ticker)

    response = api.news_api(q=company_name, language='en')

    if response.get('status') != 'success':
        print(f"Failed to retrieve data: {response}")
        return []

    news_items = []
    if 'results' in response:
        for article in response['results']:
            if 'title' in article and 'pubDate' in article:
                title = article['title']
                pub_date = article['pubDate']
                sentiment_score = SentimentIntensityAnalyzer().polarity_scores(title)['compound']
                sentiment = 'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'
                news_items.append({
                    'title': title,
                    'date': pub_date.split(' ')[0],
                    'url': article['link'],
                    'sentiment': sentiment,
                    'sentiment_score': sentiment_score
                })
    return news_items



@app.route('/predict/<ticker>/<int:prediction_days>')
def predict(ticker, prediction_days):
    start_date = '2021-01-01'
    end_date = datetime.date.today().isoformat()

    # Fetch stock data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Check if data was successfully fetched
    if data is None or data.empty:
        flash("No stock data returned. Please check the ticker symbol or date range.", "danger")
        return redirect(url_for('index'))

    # Reset index so Date becomes a column and check for MultiIndex
    data = data.reset_index()
    print("Downloaded data:\n", data.head())  # For debugging
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)



    # Add technical indicators
    data = model.add_technical_indicators(data)

    # Get sentiment data
    sentiment_df = model.get_news_sentiment(ticker, start_date, end_date)

    # Preprocess the data
    scaled_data, scaler = model.preprocess_data(data, sentiment_df)

    # Create and train the model
    lstm_model = model.create_lstm_model((60, scaled_data.shape[1]))
    model.train_model(lstm_model, scaled_data)

    # Make predictions
    predictions = model.predict_stock_price(lstm_model, scaled_data, scaler, prediction_days)

    # Plotting the results
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.8, 0.2])
    fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Candlestick'), row=1, col=1)

    # Add prediction line
    fig.add_trace(go.Scatter(x=data['Date'].tolist() + [data['Date'].iloc[-1] + datetime.timedelta(days=i) for i in range(1, prediction_days + 1)], y=predictions, mode='lines', name='Predicted'), row=1, col=1)

    # Add moving averages
    fig.add_trace(go.Scatter(x=data['Date'], y=data['20_MA'], mode='lines', name='20 Day MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['50_MA'], mode='lines', name='50 Day MA'), row=1, col=1)

    # Add support and resistance
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Support'], mode='lines', name='Support', line=dict(dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Resistance'], mode='lines', name='Resistance', line=dict(dash='dot')), row=1, col=1)

    # Add candlestick patterns
    colors = {'Head and Shoulders': 'pink', 'Inverse Head and Shoulders': 'lightgreen', 'Double Top': 'yellow', 'Double Bottom': 'cyan',
              'Rising Wedge': 'magenta', 'Falling Wedge': 'purple', 'Bullish Pennant': 'blue', 'Bearish Pennant': 'red',
              'Bullish Flag': 'orange', 'Bearish Flag': 'brown', 'Cup and Handle': 'green', 'Descending Triangle': 'lightblue'}

    for pattern in model.patterns.keys():
        pattern_dates = data['Date'][data[pattern]]
        pattern_prices = data['Close'][data[pattern]]
        fig.add_trace(go.Scatter(x=pattern_dates, y=pattern_prices, mode='markers', name=pattern, marker=dict(color=colors[pattern], size=8, symbol='circle')), row=1, col=1)

    fig.update_layout(title=f'{ticker} Stock Price and Short-Term Prediction', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')

    graph_html = fig.to_html(full_html=False)
    
    # Fetch recent news
    recent_news = fetch_recent_news(ticker)

    return render_template('prediction.html', ticker=ticker, graph_html=graph_html, news=recent_news)


if __name__ == '__main__':
    app.run(debug=True)
