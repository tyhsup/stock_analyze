from django.core.management.base import BaseCommand
from market_data.models import StockTW
import requests
import pandas as pd
from datetime import datetime
import time

class Command(BaseCommand):
    help = 'Update Taiwan Stock List from TWSE/TPEX'

    def handle(self, *args, **options):
        self.stdout.write("Fetching TW stock list...")
        
        urls = [
            "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", # 上市
            "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"  # 上櫃
        ]
        
        created_count = 0
        updated_count = 0
        
        for url in urls:
            try:
                res = requests.get(url)
                res.encoding = 'big5'
                dfs = pd.read_html(res.text)
                if not dfs: continue
                df = dfs[0]
                
                # Cleanup
                df.columns = df.iloc[0]
                df = df.iloc[1:]
                
                current_type = ""
                for _, row in df.iterrows():
                    val = str(row.get('有價證券代號及名稱', ''))
                    if "　" not in val and len(val) > 0:
                        current_type = val
                    elif current_type == "股票" and "　" in val:
                        parts = val.split("　")
                        symbol = parts[0].strip()
                        name = parts[1].strip()
                        market = "sii" if "strMode=2" in url else "otc"
                        
                        obj, created = StockTW.objects.get_or_create(
                            symbol=symbol,
                            defaults={'name': name, 'market': market, 'last_updated': datetime.now()}
                        )
                        
                        if created:
                            created_count += 1
                        else:
                            if obj.name != name or obj.market != market:
                                obj.name = name
                                obj.market = market
                                obj.last_updated = datetime.now()
                                obj.save()
                                updated_count += 1
                
                time.sleep(2) # Be polite
            except Exception as e:
                self.stderr.write(f"Error fetching from {url}: {e}")

        self.stdout.write(self.style.SUCCESS(f"Successfully updated stocks. Created: {created_count}, Updated: {updated_count}"))
