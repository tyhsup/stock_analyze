from django.core.management.base import BaseCommand, CommandError
from sec_edgar.services.edgar_financial_service import EdgarFinancialService

class Command(BaseCommand):
    help = 'Sync SEC standardized XBRL Financials for a specific US stock'

    def add_arguments(self, parser):
        parser.add_argument('--ticker', type=str, required=True, help='US Stock Ticker (e.g. AAPL)')

    def handle(self, *args, **options):
        ticker = options['ticker'].upper()

        self.stdout.write(self.style.NOTICE(f"Starting XBRL financials sync for {ticker}..."))
        
        try:
            service = EdgarFinancialService()
            result = service.sync_financials_by_ticker(ticker)
            
            if result.get("status") == "success":
                self.stdout.write(self.style.SUCCESS(
                    f"Successfully completed: {result.get('message')}. Synced {result.get('count')} financial data points."
                ))
            else:
                self.stdout.write(self.style.ERROR(f"Sync failed: {result.get('message')}"))
                
        except Exception as e:
            raise CommandError(f"Error executing sync_financials: {str(e)}")
