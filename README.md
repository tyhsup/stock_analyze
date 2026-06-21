# Smart Stock Analysis & AI Valuation Platform

## Project Overview

This project is a comprehensive student portfolio website designed to provide end-to-end research, valuation, and AI-driven insights for the Taiwan and US stock markets. 

Built with the Django backend framework and a local MySQL database, the platform automates stock data synchronization and integrates financial engineering tools (TA-Lib), institutional chip tracking, LSTM-based price forecasting, and Large Language Model (LLM) sentiment analysis. It aims to solve the problem of fragmented investment information by offering retail investors a unified, professional-grade research station.

---

## Core Features

1. **Dual-Theme Interactive Dashboard**
   * Features Light and Dark modes with optimized CSS contrast for readability.
   * Integrates up to 12 interactive ApexCharts on the homepage, including K-line charts, trading volume, technical indicators (RSI, MACD, HT_PHASOR trajectory), institutional holding trends, holding concentration, and a Gemini-powered gauge chart.

2. **Institutional Chip Tracking**
   * Visualizes buy, sell, and net position dynamics of Foreign Investors, Investment Trusts, and Dealers.
   * Plots institutional holding ratios and concentration donut charts to help users identify major capital flows.

3. **Fair Value Calculator**
   * Uses a blended valuation model combining the Discounted Cash Flow (DCF) model and the Market Approach (Relative Valuation / Multipliers).
   * Automatically calculates fair value and potential upside percentages, displaying dynamic assumption tables.

4. **AI News Insights & Sentiment Analysis**
   * Connects to financial news APIs (e.g., CNYES) and leverages LLMs (Gemini / Llama 3) for text summarization and sentiment analysis.
   * Generates structured "AI Insights" cards detailing short, medium, and long-term market impacts, paired with news sentiment distribution charts.

5. **Automated Data Scheduler**
   * A dedicated management interface to schedule or manually trigger data synchronization tasks.
   * Integrates TWSE, TPEX, and US stock collectors. All ingested price, chip, and financial report data are persisted locally in MySQL to prevent external API rate limits.

---

## Technology Stack

* **Frontend**
  * **Core Structure**: HTML5, JavaScript, Bootstrap 5, Vanilla CSS.
  * **Data Visualization**: ApexCharts.js (for high-performance, interactive financial charts).
  * **Icon Libraries**: FontAwesome, Bootstrap Icons.

* **Backend**
  * **Web Framework**: Django 5.x (Python 3.11+).
  * **Background Processing**: Custom task scheduler utilizing multi-threading and a web-based control panel.

* **Database Management**
  * **Database**: MySQL (for persistent storage of historical prices, institutional chips, and quarterly financial statements).
  * **Optimization**: Django ORM with mysql-connector-pooling.

* **Data Scraping & Automation**
  * **APIs**: yfinance (US market data), CNYES News API.
  * **Crawlers**: aiohttp (asynchronous HTTP requests with proxy and User-Agent rotation to handle rate limits) and custom TWSE/TPEX CLIs.

* **Data Science & Machine Learning**
  * **Financial Indicators**: TA-Lib (RSI, MACD, HT_PHASOR calculation).
  * **Data Processing**: Pandas (vectorized operations), NumPy.
  * **Predictive Modeling**: TensorFlow/Keras (LSTM for time-series stock price forecasting) and Scikit-learn.
  * **Generative AI**: LLM APIs for automated sentiment analysis and structural text summary generation.

---

## Database Schema Overview

The database contains the following key tables:
* `stocks_tw` / `stocks_us`: Metadata and listing status of TWSE/TPEX and US companies.
* `stock_cost` / `stock_cost_us`: Daily historical market data (Open, High, Low, Close, Volume).
* `stock_investor` / `stock_investor_us`: Daily institutional trading details and holding ratios.
* `financial_raw_tw` / `financial_raw_us`: Raw quarterly and annual financial statements.
* `valuation_valuationresult`: Stored intrinsic value results and model assumptions.

---

## Installation & Setup Guide

### Prerequisites
* Python 3.11+
* MySQL Server (Create a database named `stock_tw_analyse`)
* TA-Lib C++ Library (Windows users are recommended to download and install pre-built `.whl` files)

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd mydjango
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Create a `.env` file under `demo/stock_Django/` and fill in your MySQL credentials and API keys:
   ```env
   DB_HOST=localhost
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   DB_NAME=stock_tw_analyse
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Run Database Migrations**
   ```bash
   python manage.py migrate
   ```

5. **Start Django Development Server**
   ```bash
   python manage.py runserver
   ```
   Once started, visit `http://127.0.0.1:8000/` in your web browser.
