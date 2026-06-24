from django.core.management.base import BaseCommand, CommandError
from sec_edgar.services.edgar_13f_service import Edgar13FService

class Command(BaseCommand):
    help = 'Sync SEC 13F Institutional Holdings for a specific US stock (Stock-centric)'

    def add_arguments(self, parser):
        parser.add_argument('--ticker', type=str, required=True, help='US Stock Ticker (e.g. AAPL)')
        parser.add_argument('--quarters', type=int, default=1, help='Number of quarters to sync (default: 1)')

    def handle(self, *args, **options):
        ticker = options['ticker'].upper()
        quarters = options['quarters']

        self.stdout.write(self.style.NOTICE(f"Starting 13F holdings sync for {ticker} ({quarters} quarters)..."))
        
        try:
            service = Edgar13FService()
            result = service.sync_holdings_by_ticker(ticker, quarters_limit=quarters)
            
            if result.get("status") == "success":
                self.stdout.write(self.style.SUCCESS(
                    f"Successfully completed: {result.get('message')}. Synced {result.get('count')} holding records."
                ))
            else:
                self.stdout.write(self.style.ERROR(f"Sync failed: {result.get('message')}"))
                
        except Exception as e:
            raise CommandError(f"Error executing sync_13f: {str(e)}")
