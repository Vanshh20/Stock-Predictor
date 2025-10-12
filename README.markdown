# Predictive-Equity-Market-Analytics-and-Recommendation-System

This Flask application allows users to register, log in, and utilize a stock prediction feature. The stock prediction model leverages an LSTM (Long Short-Term Memory) neural network to forecast future stock prices based on historical data and sentiment analysis. The application includes a recommendation system to guide users with actionable "Buy," "Hold," or "Sell" recommendations based on technical indicators, sentiment analysis, and predictive models.

### Disclaimer: This project is for educational purposes only. Use it at your own risk. The data and predictions may not be accurate.

## Features

- **User Authentication**: Secure registration and login system with password hashing using bcrypt.
- **Stock Prediction**: Predict future stock prices using LSTM models trained on historical stock data and sentiment analysis.
- **Technical Indicators**: Calculation and visualization of key indicators such as Moving Averages, Relative Strength Index (RSI), Support, and Resistance levels.
- **Candlestick Patterns**: Identification of patterns like Head and Shoulders, Double Top/Bottom, and more to inform trading decisions.
- **Recommendation System**: Provides "Buy," "Hold," or "Sell" recommendations based on a combination of technical indicators, sentiment analysis, and predictive trends.
- **Interactive Graphs**: Visualization of stock data, predictions, and technical indicators using Plotly.
- **Sentiment Analysis**: Integrates news sentiment analysis to enhance prediction accuracy.
- **Real-Time Data**: Fetches real-time stock data and news articles using Yahoo Finance and NewsData API.

## Recommendation System

The recommendation system generates "Buy," "Hold," or "Sell" signals based on a weighted analysis of multiple factors:

- **Relative Strength Index (RSI)**: Measures the stock's momentum to determine if it is overbought (RSI > 70) or oversold (RSI < 30). An oversold condition may suggest a "Buy," while an overbought condition may indicate a "Sell."
- **Sentiment Analysis**: Uses VADER sentiment analysis to evaluate the sentiment of recent news articles. Positive sentiment may support a "Buy" or "Hold" recommendation, while negative sentiment may lean toward "Sell."
- **Moving Average Comparison**: Compares the current stock price to its 50-day moving average to identify bullish (price > 50-day MA) or bearish (price < 50-day MA) trends.
- **LSTM Predictions**: Incorporates the forecasted price trend from the LSTM model. A predicted upward trend supports a "Buy" or "Hold," while a downward trend suggests a "Sell."
- **Candlestick Patterns**: Detects patterns like Head and Shoulders or Double Bottom to identify potential reversals or continuations, influencing the recommendation.

The system combines these factors using a weighted scoring mechanism to produce a final recommendation. For example, strong positive sentiment and a bullish trend may result in a "Buy" signal, while a bearish trend with negative sentiment may trigger a "Sell."

## Technologies Used

- **Flask**: Web framework for Python.
- **Keras & TensorFlow**: For building and training the LSTM model.
- **Pandas & NumPy**: For data manipulation and analysis.
- **Scikit-learn**: For data preprocessing.
- **yfinance**: To fetch historical stock data.
- **VADER Sentiment Analysis**: For analyzing sentiment from news articles.
- **Plotly**: For interactive data visualization.
- **bcrypt**: For secure password hashing.
- **NewsData API**: To fetch real-time news articles related to the stock.

## Getting Started

### Prerequisites

- Python 3.x
- Virtual environment (recommended)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/stock-prediction-flask.git
    cd stock-prediction-flask
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in the root directory.
    - Add your secret key and NewsData API key:
        ```
        SECRET_KEY=your_secret_key
        NEWS_API_KEY=your_news_api_key
        ```

### Folder Structure

```
stock-prediction-flask/
│
├── app.py                   # Main application file
├── templates/               # HTML templates
│   ├── register.html
│   ├── login.html
│   ├── index.html
│   └── prediction.html
├── static/                  # Static files (CSS, JS, images)
│   ├── css/
│       └── styles.css
│ 
├── model.py                 # LSTM models and recommendation logic
├── users.csv
├── Yahoo Ticker Symbols.csv # Data files (CSV, etc.)
├── requirements.txt         # Python dependencies
└── README.md                # Project description
```

# Usage Example

### Login

- Users can log in by entering their email and password on the login screen. If they don't have an account, they can click on the "Register" button to create one.
  <img width="1440" alt="Screenshot 2024-07-04 at 4 26 41 PM" src="https://github.com/Vanshh20/Stock-Predictor/assets/142145536/a200bc72-1fe9-4092-b4d5-ec05dd28f355">

### Register

- On the registration page, users need to provide their email, set a password, and confirm it. After successful registration, they can log in using their new credentials.
  <img width="1440" alt="Screenshot 2024-07-04 at 4 26 47 PM" src="https://github.com/Vanshh20/Stock-Predictor/assets/142145536/b68215ea-990c-46f3-9225-c63a9db8bb15">

### Choosing Stock from Dropdown List

- After logging in, users can select a stock from the dropdown list available on the main dashboard. This list includes a wide range of stocks for users to choose from.
  <img width="1440" alt="Screenshot 2024-07-04 at 4 28 29 PM" src="https://github.com/Vanshh20/Stock-Predictor/assets/142145536/05ad07db-9d7d-4c6b-9f19-e38e2a986112">

### Setting Prediction Days and Buy Price

- Users can choose the number of days for the prediction. If the selected number of days is more than 7, a warning message will appear indicating that predictions might not be accurate for longer periods.
- The maximum number of days that can be selected is 14.
- Users must enter a buy price for the stock. The buy price should be within 10% up or down of the current stock price. If the buy price is out of this range, an error message will prompt the user to enter a valid price.
  <img width="1440" alt="Screenshot 2024-07-04 at 4 29 17 PM" src="https://github.com/Vanshh20/Stock-Predictor/assets/142145536/2ea13f1f-c88d-4e13-9b69-40b810e91d4e">

### Recommendations (Buy, Hold, or Sell)

- The recommendation system provides "Buy," "Hold," or "Sell" advice based on:
  - **RSI**: Indicates overbought or oversold conditions.
  - **Sentiment Analysis**: Evaluates news sentiment for positive or negative outlook.
  - **Actual Price vs 50-day Moving Average**: Identifies bullish or bearish trends.
  - **LSTM Predictions**: Considers forecasted price movements.
  - **Candlestick Patterns**: Detects patterns signaling potential price reversals or continuations.
- The system aggregates these factors to produce a clear recommendation displayed on the dashboard.
  <img width="1440" alt="Screenshot 2024-07-04 at 4 30 43 PM" src="https://github.com/Vanshh20/Stock-Predictor/assets/142145536/7c3255af-dacf-4ede-adef-2f86abb66abd">

### Stock Chart Drawing

- The stock chart includes the following features:
  - **Pattern Generation**: Identifies and highlights patterns in the stock price movements.
  - **20-day and 50-day Moving Average Lines**: Displays these lines to show short-term and medium-term trends.
  - **Dynamic Support and Resistance Lines**: Identifies key price levels for potential reversals or continuations.
  - **Comparison between Actual Price and Predicted Price**: Shows historical and predicted prices on the same chart.
  - **Recent News and Sentiment**: Lists recent news articles with sentiment analysis for each.
  <img width="1440" alt="Screenshot 2024-07-04 at 4 35 40 PM" src="https://github.com/Vanshh20/Stock-Predictor/assets/142145536/a165b6e1-9a3a-4f81-8902-0f921f12b985">

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## Contact

If you have any questions or suggestions, feel free to open an issue or contact me at [vansh.vjain20@gmail.com](mailto:vansh.vjain20@gmail.com).

---

This project aims to provide a robust and interactive platform for stock price prediction using advanced machine learning techniques, sentiment analysis, and a sophisticated recommendation system. Happy coding!
