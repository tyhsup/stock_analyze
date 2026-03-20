from django.db import models

class StockTW(models.Model):
    symbol = models.CharField(max_length=20, primary_key=True)
    name = models.CharField(max_length=100, blank=True, null=True)
    market = models.CharField(max_length=20, blank=True, null=True)
    last_updated = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'stocks_tw'
        verbose_name = 'Taiwan Stock'
        verbose_name_plural = 'Taiwan Stocks'

    def __str__(self):
        return f"{self.symbol} {self.name}"

class StockUS(models.Model):
    symbol = models.CharField(max_length=20, primary_key=True)
    name = models.CharField(max_length=100, blank=True, null=True)
    market = models.CharField(max_length=20, blank=True, null=True)
    cik = models.CharField(max_length=20, blank=True, null=True)
    last_updated = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'stocks_us'
        verbose_name = 'US Stock'
        verbose_name_plural = 'US Stocks'

    def __str__(self):
        return f"{self.symbol} {self.name}"

class DailyPriceTW(models.Model):
    # Note: The underlying table 'stock_cost' likely has no single primary key.
    # We set 'number' as PK to satisfy Django, but be careful with queries.
    number = models.CharField(max_length=20, primary_key=True) 
    date = models.DateTimeField(db_column='Date')  # Field name made lowercase for Python style
    open = models.FloatField(db_column='Open', blank=True, null=True)
    high = models.FloatField(db_column='High', blank=True, null=True)
    low = models.FloatField(db_column='Low', blank=True, null=True)
    close = models.FloatField(db_column='Close', blank=True, null=True)
    volume = models.BigIntegerField(db_column='Volume', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'stock_cost'
        verbose_name = 'TW Daily Price'
        verbose_name_plural = 'TW Daily Prices'
        unique_together = (('number', 'date'),)

class DailyPriceUS(models.Model):
    # Same Note as TW
    number = models.CharField(max_length=20, primary_key=True)
    date = models.DateTimeField(db_column='Date')
    open = models.FloatField(db_column='Open', blank=True, null=True)
    high = models.FloatField(db_column='High', blank=True, null=True)
    low = models.FloatField(db_column='Low', blank=True, null=True)
    close = models.FloatField(db_column='Close', blank=True, null=True)
    volume = models.BigIntegerField(db_column='Volume', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'stock_cost_us'
        verbose_name = 'US Daily Price'
        verbose_name_plural = 'US Daily Prices'
        unique_together = (('number', 'date'),)
