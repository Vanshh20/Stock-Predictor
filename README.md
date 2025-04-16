# Stock-Predictor
 This Flask application allows users to register, log in, and utilize a stock prediction feature. The stock prediction model leverages an LSTM (Long Short-Term Memory) neural network to forecast future stock prices based on historical data and sentiment analysis.
 
### Disclaimer: This project is for educational purposes only. Use it at your own risk. The data and predictions may not be accurate.

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





### Folder Structure

```
stock-prediction-flask/
│
├── app.py                   # Main application file
├── templates/               # HTML templates
│   ├── register.html
│   ├── login.html
│   ├── index.html
│   └── prediction.html├
├── static/                  # Static files (CSS, JS, images)
│   ├── css/
        └── styles.css
│ 
├── model.py                  # LSTM models
├── users.csv
├── Yahoo Ticker Symbols.csv                # Data files (CSV, etc.)
├── requirements.txt         # Python dependencies
└── README.md                # Project description
```

# Usage Example:


### Login:

- Users can log in by entering their email and password on the login screen. If they don't have an account, they can click on the "Register" button to create one.
  <img width="1440" alt="Screenshot 2024-07-04 at 4 26 41 PM" src="https://github.com/Vanshh20/Stock-Predictor/assets/142145536/a200bc72-1fe9-4092-b4d5-ec05dd28f355">


### Register:

- On the registration page, users need to provide their email, set a password, and confirm it. After successful registration, they can log in using their new credentials.
  <img width="1440" alt="Screenshot 2024-07-04 at 4 26 47 PM" src="https://github.com/Vanshh20/Stock-Predictor/assets/142145536/b68215ea-990c-46f3-9225-c63a9db8bb15">

### Choosing Stock from Dropdown List

- After logging in, users can select a stock from the dropdown list available on the main dashboard. This list includes a wide range of stocks for users to choose from.
  <img width="1440" alt="Screenshot 2024-07-04 at 4 28 29 PM" src="https://github.com/Vanshh20/Stock-Predictor/assets/142145536/05ad07db-9d7d-4c6b-9f19-e38e2a986112">

  
### Setting Prediction Days and Buy Price

- Users can choose the number of days for the prediction. If the selected number of days is more than 7, a warning message will appear indicating that predictions might not     be accurate for longer periods.
- The maximum number of days that can be selected is 14.
- Users must also enter a buy price for the stock. The buy price should be within 10% up or down of the current stock price. If the buy price is out of this range, an error     message will prompt the user to enter a valid price.
  <img width="1440" alt="Screenshot 2024-07-04 at 4 29 17 PM" src="https://github.com/Vanshh20/Stock-Predictor/assets/142145536/2ea13f1f-c88d-4e13-9b69-40b810e91d4e">


  
### Recommendations (Hold or Sell):

- Recommendations are generated based on various factors:
  - RSI (Relative Strength Index): Indicates whether the stock is overbought or oversold.
  - Sentiment Analysis: Analyzes recent news and sentiment about the stock.
  - Actual Price vs 50-day Moving Average: Compares the current stock price to its 50-day moving average to determine trends.
  - The system will recommend whether to hold or sell the stock based on these analyses.
    ![Screenshot 2024-07-04 at 4 30 43 PM](https://github.com/Vanshh20/Stock-Predictor/assets/142145536/7c3255af-dacf-4ede-adef-2f86abb66abd)


  
### Stock Chart Drawing:

- The stock chart includes the following features:
  - Pattern Generation: Identifies and highlights patterns in the stock price movements.
  - 20-day and 50-day Moving Average Lines: Displays these lines to show short-term and medium-term trends.
  - Dynamic Support and Resistance Lines: These lines help identify key price levels where the stock might reverse or continue its trend.
  - Comparison between Actual Price and Predicted Price: Shows both the actual historical prices and the predicted future prices on the same chart.
  - Recent News and Sentiment: Lists all recent news articles related to the stock and provides sentiment analysis for each article.
    <img width="1440" alt="Screenshot 2024-07-04 at 4 35 40 PM" src="https://github.com/Vanshh20/Stock-Predictor/assets/142145536/a165b6e1-9a3a-4f81-8902-0f921f12b985">


## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## Contact

If you have any questions or suggestions, feel free to open an issue or contact me at [vansh.vjain20@gmail.com](mailto:vansh.vjain20@gmail.com).

---

This project aims to provide a robust and interactive platform for stock price prediction using advanced machine learning techniques and sentiment analysis. Happy coding!
