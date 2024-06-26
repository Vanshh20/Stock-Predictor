# Stock-Predictor
 This Flask application allows users to register, log in, and utilize a stock prediction feature. The stock prediction model leverages an LSTM (Long Short-Term Memory) neural network to forecast future stock prices based on historical data and sentiment analysis.
 # Stock Prediction Flask Application

## Description

This Flask application allows users to register, log in, and utilize a stock prediction feature. The stock prediction model leverages an LSTM (Long Short-Term Memory) neural network to forecast future stock prices based on historical data and sentiment analysis. The application also integrates various technical indicators and candlestick patterns to enhance prediction accuracy. Users can visualize the predictions and technical analysis using interactive Plotly graphs.

## Features

- **User Authentication**: Secure registration and login system with password hashing using bcrypt.
- **Stock Prediction**: Predict future stock prices using LSTM models trained on historical stock data and sentiment analysis.
- **Technical Indicators**: Calculation and visualization of common technical indicators such as Moving Averages, Support, and Resistance levels.
- **Candlestick Patterns**: Identification of key candlestick patterns including Head and Shoulders, Double Top/Bottom, and more.
- **Interactive Graphs**: Visualization of stock data, predictions, and technical indicators using Plotly.
- **Sentiment Analysis**: Integration of news sentiment analysis to enhance prediction accuracy.
- **Real-Time Data**: Fetches real-time stock data and news articles using Yahoo Finance and NewsData API.

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
│   ├── 
│   ├── register.html
│   ├── login.html
│   ├── index.html
│   └── prediction.html
├── static/                  # Static files (CSS, JS, images)
│   ├── css/
│ 
├── model.py                  # LSTM models
├── users.csv
├── Yahoo Ticker Symbols - September 2017.csv                # Data files (CSV, etc.)
├── requirements.txt         # Python dependencies
└── README.md                # Project description
```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## Contact

If you have any questions or suggestions, feel free to open an issue or contact me at [vansh.vjain20@gmail.com](mailto:vansh.vjain20@gmail.com).

---

This project aims to provide a robust and interactive platform for stock price prediction using advanced machine learning techniques and sentiment analysis. Happy coding!
