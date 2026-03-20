from django.core.management.base import BaseCommand
from market_data.models import StockUS, DailyPriceUS
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import random

class Command(BaseCommand):
    help = 'Update US Stock Prices (Daily)'

    def handle(self, *args, **options):
        self.stdout.write("Starting US Stock Price Update...")
        
        # Get all stocks
        stocks = StockUS.objects.all()
        stock_list = [s.symbol for s in stocks]
        
        if not stock_list:
            self.stdout.write("No US stocks found. (Need to implement update_us_stocks first)")
            return

        batch_size = 50
        total = len(stock_list)
        
        for i in range(0, total, batch_size):
            batch = stock_list[i : i + batch_size]
            
            self.stdout.write(f"Processing batch {i} to {i+batch_size}...")
            
            try:
                data = yf.download(batch, period="5d", group_by='ticker', auto_adjust=True, progress=False)
                
                if data.empty:
                    self.stdout.write(f"  No data for batch {i}")
                    continue

                for symbol in batch:
                    try:
                        if len(batch) > 1:
                            if symbol not in data.columns.levels[0]: continue
                            df = data[symbol].copy()
                        else:
                            df = data.copy()
                        
                        df.dropna(inplace=True)
                        if df.empty: continue
                        
                        df.reset_index(inplace=True)
                        df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 
                                           'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                        
                        for _, row in df.iterrows():
                            DailyPriceUS.objects.update_or_create(
                                number=symbol,
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
                        self.stderr.write(f"Error processing {symbol}: {ex}")
                
                time.sleep(1)
            except Exception as e:
                self.stderr.write(f"Batch failed: {e}")

        self.stdout.write(self.style.SUCCESS("US Price update completed."))
