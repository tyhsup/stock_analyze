# Stock Analysis & AI Valuation Platform

A comprehensive Django-based stock analysis platform featuring institutional trend tracking, AI price predictions, and automated fair value valuation.

## 🚀 Key Features

- **Interactive Dashboard**: Modern UI for searching and analyzing stocks with Plotly interactive charts.
- **AI Price Prediction**: Integrated LSTM-based price forecasting (K-Line).
- **Institutional Trend Analysis (Chips)**: Visualizing Foreign, Investment Trust, and Dealer buy/sell dynamics.
- **Fair Value Calculator**: Blended valuation using Discounted Cash Flow (DCF) and Market Approach (Relative Valuation).
- **News Intelligence**: Keyword-based news filtering with sentiment analysis results.
- **Database Integrated**: Seamlessly connects to existing MySQL financial databases.

## 🛠️ Technology Stack

- **Backend**: Django 5.x
- **Database**: MySQL (SQLAlchemy for legacy logic integration)
- **Analysis**: TA-Lib, Pandas, NumPy, Scikit-learn, TensorFlow/Keras
- **Visualization**: Plotly.js, Matplotlib (fallback)
- **Frontend**: Bootstrap 5, HTMX, Google Fonts (Inter)

## 📋 Prerequisites

1.  **MySQL Database**: Ensure a local MySQL server is running with the `stock_tw_analyse` database.
2.  **TA-Lib**: Must be installed on your system.
    -   *Windows*: [Download pre-built binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) or use the included `.whl` if available.
3.  **Python 3.11+**

## ⚙️ Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone <repo-url>
    cd demo
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Database Configuration**:
    Update `demo/settings.py` or the `mySQL_OP.py` file with your local MySQL credentials.

4.  **Run Migrations**:
    ```bash
    python manage.py migrate
    ```

5.  **Start the Development Server**:
    ```bash
    python manage.py run_server
    ```

6.  **Access the Platform**:
    Open `http://127.0.0.1:8000/` in your browser.

## 📂 Project Structure

- `valuation/`: Valuation service and fair value logic.
- `stock_Django/`: Core analysis views and AI model logic.
- `stock_utils.py`: Refactored data processing and indicator utilities.
- `templates/`: Modern Bootstrap 5 templates.

## 📝 License
Proprietary / Collaborative Work.
