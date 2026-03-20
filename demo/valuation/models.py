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
