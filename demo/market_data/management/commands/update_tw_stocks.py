from django.core.management.base import BaseCommand
from market_data.models import StockTW
from stock_Django.stock_cost import StockCostManager
from datetime import datetime

class Command(BaseCommand):
    help = 'Update Taiwan Stock List from TWSE/TPEX'

    def handle(self, *args, **options):
        self.stdout.write("Fetching TW stock list via StockCostManager...")
        
        created_count = 0
        updated_count = 0
        
        try:
            manager = StockCostManager()
            stocks = manager.fetch_all_tw_stock_list()
            
            for s in stocks:
                symbol = s['code']
                name = s['name']
                market = s['market']
                
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
        except Exception as e:
            self.stderr.write(f"Error updating stock list: {e}")

        self.stdout.write(self.style.SUCCESS(f"Successfully updated stocks. Created: {created_count}, Updated: {updated_count}"))
