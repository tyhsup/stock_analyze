from django.db import models

class ValuationResult(models.Model):
    # Store valuation results for any stock (TW or US)
    symbol = models.CharField(max_length=20)
    market = models.CharField(max_length=10, choices=[('TW', 'Taiwan'), ('US', 'USA')])
    date = models.DateField(auto_now_add=True)
    
    # Valuation Method
    METHOD_CHOICES = [
        ('DCF', 'Discounted Cash Flow'),
        ('PE', 'P/E Ratio Model'),
        ('PB', 'P/B Ratio Model'),
        ('GRAHAM', 'Graham Number'),
    ]
    method = models.CharField(max_length=20, choices=METHOD_CHOICES)
    
    # Results
    fair_value = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    current_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    upside = models.DecimalField(max_digits=10, decimal_places=2, help_text="Percentage upside (e.g. 0.20 for 20%)", null=True, blank=True)
    
    # Store parameters used for calculation (WACC, Growth Rate, etc.)
    # Note: Requires MySQL 5.7+ for JSONField
    assumptions = models.JSONField(default=dict, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-date', 'symbol']
        indexes = [
            models.Index(fields=['symbol', 'market', 'method']),
        ]

    def __str__(self):
        return f"{self.symbol} ({self.market}) - {self.method}: {self.fair_value}"


class StockMetrics(models.Model):
    market = models.CharField(max_length=10)  # 'tw' 或 'us'
    symbol = models.CharField(max_length=20)
    pe = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True)
    pb = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True)
    dividend_yield = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'stock_metrics'
        unique_together = ('market', 'symbol')
        indexes = [
            models.Index(fields=['market', 'symbol']),
        ]

    def __str__(self):
        return f"{self.market} - {self.symbol}: PE={self.pe}, PB={self.pb}"


class MasterSelection(models.Model):
    market = models.CharField(max_length=10)  # 'tw' 或 'us'
    symbol = models.CharField(max_length=20)
    name = models.CharField(max_length=100, null=True, blank=True)
    master_name = models.CharField(max_length=50)  # 'buffett'
    rank = models.IntegerField()
    close_price = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True)
    roe = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True)
    gross_margin = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True)
    debt_ratio = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True)
    net_income_growth = models.DecimalField(max_digits=18, decimal_places=4, null=True, blank=True)
    score = models.DecimalField(max_digits=6, decimal_places=2, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'master_selection'
        unique_together = ('market', 'master_name', 'symbol')
        ordering = ['market', 'master_name', 'rank']

    def __str__(self):
        return f"{self.market} - {self.symbol} ({self.master_name}) Rank {self.rank}"


