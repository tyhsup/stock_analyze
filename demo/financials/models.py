from django.db import models

class FinancialRawTW(models.Model):
    id = models.AutoField(primary_key=True)
    symbol = models.CharField(max_length=20)
    year = models.IntegerField()
    quarter = models.IntegerField()
    statement_type = models.CharField(max_length=10)  # IS, BS, CF
    item_name = models.CharField(max_length=255)
    amount = models.DecimalField(max_digits=32, decimal_places=4)

    class Meta:
        managed = False
        db_table = 'financial_raw_tw'
        verbose_name = 'TW Financial Raw'
        verbose_name_plural = 'TW Financial Raw Data'
        unique_together = (('symbol', 'year', 'quarter', 'statement_type', 'item_name'),)

class FinancialRawUS(models.Model):
    id = models.AutoField(primary_key=True)
    symbol = models.CharField(max_length=20)
    year = models.IntegerField()
    quarter = models.IntegerField()
    statement_type = models.CharField(max_length=10)
    item_name = models.CharField(max_length=255)
    amount = models.DecimalField(max_digits=32, decimal_places=4)

    class Meta:
        managed = False
        db_table = 'financial_raw_us'
        verbose_name = 'US Financial Raw'
        verbose_name_plural = 'US Financial Raw Data'
        unique_together = (('symbol', 'year', 'quarter', 'statement_type', 'item_name'),)
