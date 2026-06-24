from django.db import models

class SecFilingIndex(models.Model):
    """
    SEC Filings Index Tracker
    """
    cik = models.CharField(max_length=20)
    ticker = models.CharField(max_length=20, null=True, blank=True)
    company_name = models.CharField(max_length=255, null=True, blank=True)
    form_type = models.CharField(max_length=20)  # e.g., '13F-HR', '10-K', '10-Q', '4'
    accession_no = models.CharField(max_length=30, unique=True)
    filing_date = models.DateField()
    period_of_report = models.DateField(null=True, blank=True)
    primary_doc_url = models.TextField(null=True, blank=True)
    synced_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'sec_filing_index'
        indexes = [
            models.Index(fields=['cik', 'form_type']),
            models.Index(fields=['ticker', 'form_type']),
            models.Index(fields=['filing_date']),
        ]

    def __str__(self):
        return f"{self.company_name} ({self.form_type} - {self.filing_date})"


class Sec13fHoldings(models.Model):
    """
    SEC 13F Institutional Holdings Details
    """
    filing = models.ForeignKey(SecFilingIndex, on_delete=models.CASCADE, related_name='holdings')
    cik = models.CharField(max_length=20)
    ticker = models.CharField(max_length=20, null=True, blank=True)
    cusip = models.CharField(max_length=9, null=True, blank=True)
    security_name = models.CharField(max_length=255, null=True, blank=True)
    shares = models.BigIntegerField(default=0)
    value_usd = models.BigIntegerField(default=0)  # Stored in thousands as SEC original format
    put_call = models.CharField(max_length=10, default='', blank=True)  # Empty string for common stock, 'PUT'/'CALL' for options
    investment_discretion = models.CharField(max_length=20, null=True, blank=True)
    period_of_report = models.DateField()
    
    # Quarterly changes
    prev_shares = models.BigIntegerField(null=True, blank=True)
    shares_change = models.BigIntegerField(null=True, blank=True)
    change_pct = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    action_type = models.CharField(max_length=20, null=True, blank=True)  # 'NEW', 'INCREASED', 'DECREASED', 'SOLD_ALL', 'UNCHANGED'
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'sec_13f_holdings'
        unique_together = ('filing', 'cusip', 'put_call')
        indexes = [
            models.Index(fields=['cik', 'period_of_report']),
            models.Index(fields=['ticker', 'period_of_report']),
            models.Index(fields=['cusip']),
        ]

    def __str__(self):
        return f"{self.cik} holds {self.shares} of {self.ticker or self.cusip}"


class SecInsiderTrades(models.Model):
    """
    SEC Form 4 Insider Trades Details
    """
    cik = models.CharField(max_length=20)
    ticker = models.CharField(max_length=20)
    insider_name = models.CharField(max_length=255)
    insider_title = models.CharField(max_length=255, null=True, blank=True)
    is_senior_officer = models.BooleanField(default=False)
    transaction_date = models.DateField()
    transaction_code = models.CharField(max_length=5)  # e.g., 'S', 'P', 'F'
    transaction_type = models.CharField(max_length=50, null=True, blank=True)  # e.g., 'DISCRETIONARY_SALE'
    shares = models.BigIntegerField(default=0)
    price_per_share = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True)
    total_value = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True)
    shares_owned_after = models.BigIntegerField(null=True, blank=True)
    filing_date = models.DateField(null=True, blank=True)
    source = models.CharField(max_length=20, default='pp-edgar')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'sec_insider_trades'
        unique_together = ('ticker', 'insider_name', 'transaction_date', 'transaction_code', 'shares')
        indexes = [
            models.Index(fields=['ticker', 'transaction_date']),
            models.Index(fields=['transaction_code']),
            models.Index(fields=['is_senior_officer', 'ticker']),
        ]

    def __str__(self):
        return f"{self.insider_name} {self.transaction_code} {self.shares} shares of {self.ticker}"


class SecFinancialXbrl(models.Model):
    """
    SEC Standardized XBRL Financials
    """
    ticker = models.CharField(max_length=20)
    cik = models.CharField(max_length=20, null=True, blank=True)
    period_end = models.DateField()
    fiscal_year = models.IntegerField()
    fiscal_quarter = models.IntegerField()  # 0 = annual
    concept = models.CharField(max_length=255)  # e.g., 'Revenues', 'NetIncomeLoss'
    value = models.DecimalField(max_digits=32, decimal_places=4)
    unit = models.CharField(max_length=50, default='USD')
    form_type = models.CharField(max_length=10, null=True, blank=True)  # '10-K' or '10-Q'
    accession_no = models.CharField(max_length=30, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'sec_financial_xbrl'
        unique_together = ('ticker', 'period_end', 'concept', 'form_type')
        indexes = [
            models.Index(fields=['ticker', 'fiscal_year', 'fiscal_quarter']),
            models.Index(fields=['concept']),
        ]

    def __str__(self):
        return f"{self.ticker} {self.concept} in FY{self.fiscal_year}Q{self.fiscal_quarter}: {self.value}"
