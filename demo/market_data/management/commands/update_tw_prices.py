from django.core.management.base import BaseCommand
from market_data.models import StockTW, DailyPriceTW
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import random

class Command(BaseCommand):
    help = 'Update Taiwan Stock Prices (Daily)'

    def handle(self, *args, **options):
        self.stdout.write("Starting TW Stock Price Update...")
        
        # Get all stocks
        stocks = StockTW.objects.all()
        stock_list = [s.symbol for s in stocks]
        
        if not stock_list:
            self.stdout.write("No stocks found. Run 'update_tw_stocks' first.")
            return

        batch_size = 50
        total = len(stock_list)
        
        for i in range(0, total, batch_size):
            batch = stock_list[i : i + batch_size]
            symbols_map = {f"{s}.TW": s for s in batch} # yfinance needs .TW suffix
            files_to_download = list(symbols_map.keys())
            
            self.stdout.write(f"Processing batch {i} to {i+batch_size}...")
            
            try:
                # Optimized: Always try to fetch recent data first
                # In production, we should check last_date per stock to decide start_date
                # For simplicity here, we fetch last 5 days by default, or max if needed.
                
                data = yf.download(files_to_download, period="5d", group_by='ticker', auto_adjust=True, progress=False)
                
                if data.empty:
                    self.stdout.write(f"  No data for batch {i}")
                    continue

                for yf_symbol, db_symbol in symbols_map.items():
                    try:
                        if len(files_to_download) > 1:
                             # Multi-level columns: (Symbol, OHLCV)
                            if yf_symbol not in data.columns.levels[0]: continue
                            df = data[yf_symbol].copy()
                        else:
                            df = data.copy()
                        
                        df.dropna(inplace=True)
                        if df.empty: continue
                        
                        df.reset_index(inplace=True)
                        # Rename columns to match model
                        df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 
                                           'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                        
                        price_objects = []
                        for _, row in df.iterrows():
                             # Use ignore_conflicts=True in bulk_create (Django 4.1+) or manual optimized insert
                            
                            # Note: We are using managed=False models, so strictly speaking we should use raw SQL 
                            # or be very careful. But since we defined PK, Django ORM works for basic operations.
                            # Upsert is tricky.
                            
                            DailyPriceTW.objects.update_or_create(
                                number=db_symbol,
                                date=row['date'],
                                defaults={
                                    'open': row['open'],
                                    'high': row['high'],
                                    'low': row['low'],
                                    'close': row['close'],
                                    'volume': row['volume']
                                }
                            )
                    except Exception as ex:
                        self.stderr.write(f"Error processing {yf_symbol}: {ex}")
                
                time.sleep(1)
            except Exception as e:
                self.stderr.write(f"Batch failed: {e}")

        self.stdout.write(self.style.SUCCESS("Price update completed."))
