from django.core.management.base import BaseCommand
from market_data.models import StockTW, DailyPriceTW
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import random

class Command(BaseCommand):
    help = 'Update Taiwan Stock Prices (Daily)'
    def _save_prices(self, df, yf_symbol):
        df_copy = df.copy()
        # Flatten MultiIndex columns if present
        if isinstance(df_copy.columns, pd.MultiIndex):
            df_copy.columns = df_copy.columns.get_level_values(0)
            
        df_copy.dropna(inplace=True)
        if df_copy.empty: return
        
        df_copy.reset_index(inplace=True)
        # Rename columns to match model
        df_copy.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 
                                'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        
        for _, row in df_copy.iterrows():
            date_val = row['date']
            # Safely convert date to python datetime
            if hasattr(date_val, 'to_pydatetime'):
                py_date = date_val.to_pydatetime()
            else:
                py_date = pd.to_datetime(date_val).to_pydatetime()
                
            DailyPriceTW.objects.update_or_create(
                number=yf_symbol, # Write symbol with suffix (e.g. '2330.TW') to match DB
                date=py_date,
                defaults={
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
            )
    def handle(self, *args, **options):
        self.stdout.write("Starting TW Stock Price Update...")
        
        # Get all stocks from DB
        stocks = list(StockTW.objects.all())
        if not stocks:
            self.stdout.write("No stocks found. Run 'update_tw_stocks' first.")
            return

        # Query existing symbols from DailyPriceTW
        existing_symbols = set(DailyPriceTW.objects.values_list('number', flat=True).distinct())
        
        new_stocks = []
        existing_stocks = []
        
        for s in stocks:
            suffix = ".TW" if s.market == "sii" else ".TWO"
            full_symbol = f"{s.symbol}{suffix}"
            if full_symbol not in existing_symbols:
                new_stocks.append(s)
            else:
                existing_stocks.append(s)
                
        self.stdout.write(f"Classified stocks: {len(new_stocks)} new stocks, {len(existing_stocks)} existing stocks.")

        # 1. Process New Stocks: Download history with period="max"
        if new_stocks:
            self.stdout.write(f"Processing {len(new_stocks)} new stocks. Fetching max history...")
            for s in new_stocks:
                suffix = ".TW" if s.market == "sii" else ".TWO"
                yf_symbol = f"{s.symbol}{suffix}"
                self.stdout.write(f"  Downloading max history for {yf_symbol}...")
                try:
                    data = yf.download([yf_symbol], period="max", auto_adjust=True, progress=False)
                    if data.empty:
                        self.stdout.write(f"    No data found for {yf_symbol}")
                        continue
                    
                    self._save_prices(data, yf_symbol)
                    time.sleep(1) # Be polite
                except Exception as ex:
                    self.stderr.write(f"    Error processing new stock {yf_symbol}: {ex}")

        # 2. Process Existing Stocks: Download recent data with period="5d"
        if existing_stocks:
            self.stdout.write(f"Processing {len(existing_stocks)} existing stocks in batches...")
            batch_size = 50
            total = len(existing_stocks)
            
            for i in range(0, total, batch_size):
                batch = existing_stocks[i : i + batch_size]
                symbols_map = {}
                for s in batch:
                    suffix = ".TW" if s.market == "sii" else ".TWO"
                    symbols_map[f"{s.symbol}{suffix}"] = s.symbol
                    
                files_to_download = list(symbols_map.keys())
                self.stdout.write(f"Processing batch {i} to {i+batch_size}...")
                
                try:
                    data = yf.download(files_to_download, period="5d", group_by='ticker', auto_adjust=True, progress=False)
                    
                    if data.empty:
                        self.stdout.write(f"  No data for batch {i}")
                        continue

                    for yf_symbol in symbols_map.keys():
                        try:
                            if len(files_to_download) > 1:
                                if yf_symbol not in data.columns.levels[0]: continue
                                df = data[yf_symbol].copy()
                            else:
                                df = data.copy()
                            
                            self._save_prices(df, yf_symbol)
                        except Exception as ex:
                            self.stderr.write(f"Error processing {yf_symbol}: {ex}")
                    
                    time.sleep(1)
                except Exception as e:
                    self.stderr.write(f"Batch failed: {e}")

        self.stdout.write(self.style.SUCCESS("Price update completed."))
