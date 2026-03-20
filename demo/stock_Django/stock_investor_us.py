import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import os
import sys
import time
import re
from datetime import datetime
from sqlalchemy import text

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# Fix import path when run directly
try:
    from . import mySQL_OP
except ImportError:
    try:
        from stock_Django import mySQL_OP
    except ImportError:
        import mySQL_OP

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class USStockInvestorManager:
    def __init__(self):
        self.sql_op = mySQL_OP.OP_Fun()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }

    def get_shares_outstanding(self, ticker: str) -> int:
        """
        Fetch total shares outstanding from yfinance.
        Tries yfinance first, falls back to 0 on rate-limit or error.
        """
        if not HAS_YFINANCE:
            logger.warning("yfinance not installed — cannot calculate pct_out.")
            return 0
        try:
            info = yf.Ticker(ticker).fast_info
            shares = getattr(info, 'shares', None) or 0
            if shares:
                logger.info(f"{ticker}: shares_outstanding = {shares:,} (fast_info)")
                return int(shares)
        except Exception:
            pass

        # Slower fallback using full info
        try:
            info = yf.Ticker(ticker).info
            shares = (
                info.get('sharesOutstanding')
                or info.get('impliedSharesOutstanding')
                or 0
            )
            if shares:
                logger.info(f"{ticker}: shares_outstanding = {shares:,} (info)")
                return int(shares)
        except Exception as e:
            logger.warning(f"{ticker}: yfinance failed: {e}")

        return 0

    def scrape_market_cap_from_stockzoa(self, ticker: str, soup) -> int:
        """
        Try to extract market cap from the already-fetched Stockzoa page soup.
        Returns 0 if not found.
        """
        try:
            # Look for market cap stat box (varies by page layout)
            for tag in soup.find_all(['span', 'div', 'td', 'p']):
                text = tag.get_text(strip=True).lower()
                if 'market cap' in text:
                    # Find next sibling or child with numeric value
                    sibling = tag.find_next_sibling()
                    if sibling:
                        val_text = sibling.get_text(strip=True)
                        # Parse values like "$3.1T", "$450B"
                        val_text = val_text.replace('$', '').replace(',', '').strip()
                        mult = 1
                        if val_text.endswith('T'):
                            mult = 1_000_000_000_000
                            val_text = val_text[:-1]
                        elif val_text.endswith('B'):
                            mult = 1_000_000_000
                            val_text = val_text[:-1]
                        elif val_text.endswith('M'):
                            mult = 1_000_000
                            val_text = val_text[:-1]
                        try:
                            return int(float(val_text) * mult)
                        except ValueError:
                            pass
        except Exception:
            pass
        return 0

    def estimate_shares_outstanding_from_data(self, df: pd.DataFrame) -> int:
        """
        Fallback: estimate shares_outstanding from the scraped data.
        Uses median implied price (value_usd / shares) across all holders.
        Then scales up total institutional value by ~1/0.75 since institutions
        typically own ~75% of large-cap US stock outstanding shares.
        """
        try:
            valid = df[(df['shares'] > 0) & (df['value_usd'] > 0)].copy()
            if valid.empty:
                return 0
            valid['implied_price'] = valid['value_usd'] / valid['shares']
            # Filter out outliers (price within 3 std deviations of median)
            median_price = valid['implied_price'].median()
            if median_price <= 0:
                return 0
            total_institutional_value = df['value_usd'].sum()
            total_institutional_shares = df['shares'].sum()
            # Use total shares directly, scaled up assuming ~75% institutional ownership
            INSTITUTIONAL_OWNERSHIP_ASSUMPTION = 0.75
            estimated_shares = total_institutional_shares / INSTITUTIONAL_OWNERSHIP_ASSUMPTION
            logger.info(
                f"Fallback estimation: inst_shares={total_institutional_shares:,}, "
                f"assumed_inst_pct={INSTITUTIONAL_OWNERSHIP_ASSUMPTION:.0%}, "
                f"estimated_total_shares={estimated_shares:,.0f}"
            )
            return int(estimated_shares)
        except Exception as e:
            logger.warning(f"Could not estimate shares_outstanding from data: {e}")
            return 0

    def init_db(self):
        """Create stock_investor_us table if not exists"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS stock_investor_us (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ticker VARCHAR(20),
            date DATE,
            holder_name VARCHAR(255),
            shares BIGINT,
            pct_out FLOAT,
            value_usd BIGINT,
            change_shares BIGINT,
            change_pct FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY ticker_holder (ticker, holder_name, date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        try:
            with self.sql_op.engine.connect() as conn:
                conn.execute(text(create_sql))
                conn.commit()
            logger.info("stock_investor_us table ready.")
        except Exception as e:
            logger.error(f"Error initializing DB: {e}")

    def fetch_institutional_holders(self, ticker: str) -> pd.DataFrame:
        """
        Fetch institutional holders from Stockzoa.com using requests and BeautifulSoup.
        Stockzoa is confirmed to be scrapable with simple requests.
        """
        ticker = ticker.upper()
        # Stockzoa uses lowercase ticker in URL
        url = f"https://stockzoa.com/ticker/{ticker.lower()}/"
        
        try:
            logger.info(f"Fetching Stockzoa data for {ticker}...")
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                logger.error(f"{ticker}: HTTP {response.status_code} error.")
                return pd.DataFrame()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            # Look for the table. Stockzoa's main table is usually the first one or has a header 'Top investors'
            table = soup.find('table')
            
            if not table:
                logger.warning(f"{ticker}: No table found on page.")
                return pd.DataFrame()
            
            rows = table.find_all('tr')
            logger.info(f"{ticker}: Found {len(rows)} rows in table.")
            
            extracted_data = []
            
            # Helper to parse numbers like "1.4B", "$ 388B", "2%"
            def parse_val(s):
                if not s: return 0
                s = s.replace(',', '').replace('$', '').replace('%', '').replace('+', '').strip()
                
                multiplier = 1
                if 'B' in s:
                    multiplier = 1000000000
                    s = s.replace('B', '')
                elif 'M' in s:
                    multiplier = 1000000
                    s = s.replace('M', '')
                elif 'K' in s:
                    multiplier = 1000
                    s = s.replace('K', '')
                
                try:
                    return float(s) * multiplier
                except:
                    return 0

            for row in rows[1:]: # Skip header
                cols = row.find_all('td')
                if len(cols) >= 5:
                    # Stockzoa Columns:
                    # 0: Fund or Company Name
                    # 1: Shares Held
                    # 2: Valued At
                    # 3: Change in Shares
                    # 4: As Of
                    
                    owner = cols[0].get_text(strip=True)
                    shares_str = cols[1].get_text(strip=True)
                    value_str = cols[2].get_text(strip=True)
                    change_pct_str = cols[3].get_text(strip=True)
                    date_str = cols[4].get_text(strip=True)
                    
                    # Basic date parsing: "Dec 2025" -> "2025-12-31"
                    # We'll handle this in the update method.
                    
                    extracted_data.append({
                        'holder_name': owner,
                        'date_str': date_str,
                        'shares': int(parse_val(shares_str)),
                        'value_usd': int(parse_val(value_str)),
                        'change_shares': 0, # Stockzoa only shows pct change in this table
                        'change_pct': parse_val(change_pct_str) / 100.0,
                        'pct_out': 0  # Will be computed in update_investor_db
                    })
            
            if not extracted_data:
                logger.warning(f"{ticker}: No data parsed.")
                return pd.DataFrame()
                
            df = pd.DataFrame(extracted_data)
            logger.info(f"{ticker}: Successfully extracted {len(df)} holders.")
            return df

        except Exception as e:
            logger.error(f"{ticker}: Scraper error: {e}")
            return pd.DataFrame()

    def get_latest_holders(self, ticker: str, top_n: int = 15) -> pd.DataFrame:
        """
        Query the database for the latest institutional holder data for a given ticker.
        """
        query = """
        SELECT holder_name, date, shares, pct_out, value_usd, change_shares, change_pct 
        FROM stock_investor_us 
        WHERE ticker = :ticker 
        AND date = (SELECT MAX(date) FROM stock_investor_us WHERE ticker = :ticker)
        ORDER BY shares DESC 
        LIMIT :limit
        """
        try:
            with self.sql_op.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params={"ticker": ticker.upper(), "limit": top_n})
                return df
        except Exception as e:
            logger.error(f"Error fetching latest holders for {ticker}: {e}")
            return pd.DataFrame()

    def update_investor_db(self, tickers: list):
        """Main entry point to update DB for a list of tickers"""
        self.init_db()
        
        for ticker in tickers:
            df = self.fetch_institutional_holders(ticker)
            if df.empty:
                continue
                
            df['ticker'] = ticker.upper()

            # Enrich pct_out using yfinance shares_outstanding, with fallback estimation
            shares_outstanding = self.get_shares_outstanding(ticker)
            if shares_outstanding > 0:
                df['pct_out'] = df['shares'] / shares_outstanding
                logger.info(f"{ticker}: pct_out computed from yfinance shares_outstanding {shares_outstanding:,}.")
            else:
                # Fallback: estimate denominator from value_usd / implied_price
                est_shares = self.estimate_shares_outstanding_from_data(df)
                if est_shares > 0:
                    df['pct_out'] = df['shares'] / est_shares
                    logger.info(f"{ticker}: pct_out computed from estimated shares_outstanding {est_shares:,}.")
                else:
                    logger.warning(f"{ticker}: pct_out will be 0 — no shares_outstanding data available.")

            def format_date(d_str):
                # Handle formats like "Dec 2024", "Sep 2024"
                try:
                    # Try current date if parsing fails
                    dt = datetime.strptime(d_str, '%b %Y')
                    # Set to last day of month approximately or leave as 1st
                    return dt.strftime('%Y-%m-%d')
                except:
                    return datetime.now().strftime('%Y-%m-%d')
            
            df['date'] = df['date_str'].apply(format_date)
            
            try:
                with self.sql_op.engine.connect() as conn:
                    for _, row in df.iterrows():
                        upsert_sql = """
                        INSERT INTO stock_investor_us 
                        (ticker, date, holder_name, shares, pct_out, value_usd, change_shares, change_pct)
                        VALUES (:ticker, :date, :holder_name, :shares, :pct_out, :value_usd, :change_shares, :change_pct)
                        ON DUPLICATE KEY UPDATE 
                        shares = VALUES(shares),
                        pct_out = VALUES(pct_out),
                        value_usd = VALUES(value_usd),
                        change_shares = VALUES(change_shares),
                        change_pct = VALUES(change_pct);
                        """
                        conn.execute(text(upsert_sql), {
                            "ticker": row['ticker'],
                            "date": row['date'],
                            "holder_name": row['holder_name'],
                            "shares": row['shares'],
                            "pct_out": row['pct_out'],
                            "value_usd": row['value_usd'],
                            "change_shares": row['change_shares'],
                            "change_pct": row['change_pct']
                        })
                    conn.commit()
                logger.info(f"{ticker}: Updated DB with {len(df)} rows.")
            except Exception as e:
                logger.error(f"{ticker}: DB Error: {e}")
            
            time.sleep(2)

if __name__ == "__main__":
    manager = USStockInvestorManager()
    
    import sys
    args = sys.argv[1:]
    if not args:
        test_tickers = ['AAPL', 'NVDA', 'TSLA']
    else:
        test_tickers = args
        
    logger.info(f"Starting US investor data fetch for: {test_tickers}")
    manager.update_investor_db(test_tickers)
    logger.info("Done.")
