from django.db import models

class ChipData(models.Model):
    # Composite PK: (date, number). define number as PK.
    date = models.DateField(db_column='日期')
    number = models.CharField(db_column='number', max_length=20, primary_key=True)
    
    # Common Institutional Investor Columns (Data source: TWSE)
    # Note: Column names must match EXACTLY with MySQL table columns (Chinese).
    # If the table uses different names, these queries will fail. 
    # Based on stock_investor.py, columns are dynamic but generally follow TWSE format.
    
    foreign_buy = models.BigIntegerField(db_column='外陸資買進股數(不含外資自營商)', blank=True, null=True)
    foreign_sell = models.BigIntegerField(db_column='外陸資賣出股數(不含外資自營商)', blank=True, null=True)
    foreign_net = models.BigIntegerField(db_column='外陸資買賣超股數(不含外資自營商)', blank=True, null=True)
    
    trust_buy = models.BigIntegerField(db_column='投信買進股數', blank=True, null=True)
    trust_sell = models.BigIntegerField(db_column='投信賣出股數', blank=True, null=True)
    trust_net = models.BigIntegerField(db_column='投信買賣超股數', blank=True, null=True)
    
    dealer_net = models.BigIntegerField(db_column='自營商買賣超股數', blank=True, null=True)
    # dealers usually have sub-categories (proprietary/hedging), add if needed.
    
    total_net = models.BigIntegerField(db_column='三大法人買賣超股數', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'stock_investor'
        verbose_name = 'Chip Data'
        verbose_name_plural = 'Chip Data'
        unique_together = (('number', 'date'),)
