from django.core.management.base import BaseCommand, CommandError
from sec_edgar.services.edgar_insider_service import EdgarInsiderService

class Command(BaseCommand):
    help = 'Sync SEC Form 4 Insider Trades for a specific US stock'

    def add_arguments(self, parser):
        parser.add_argument('--ticker', type=str, required=True, help='US Stock Ticker (e.g. AAPL)')
        parser.add_argument('--since', type=str, default='12mo', choices=['3mo', '6mo', '12mo'], help='Timeframe to sync (default: 12mo)')
        parser.add_argument('--limit', type=int, default=50, help='Max filings to process (default: 50)')

    def handle(self, *args, **options):
        ticker = options['ticker'].upper()
        since = options['since']
        limit = options['limit']

        self.stdout.write(self.style.NOTICE(f"Starting insider trades sync for {ticker} (since: {since}, limit: {limit})..."))
        
        try:
            service = EdgarInsiderService()
            result = service.sync_insiders_by_ticker(ticker, since=since, limit=limit)
            
            if result.get("status") == "success":
                self.stdout.write(self.style.SUCCESS(
                    f"Successfully completed: {result.get('message')}. Synced {result.get('count')} trade records."
                ))
            else:
                self.stdout.write(self.style.ERROR(f"Sync failed: {result.get('message')}"))
                
        except Exception as e:
            raise CommandError(f"Error executing sync_insider: {str(e)}")
