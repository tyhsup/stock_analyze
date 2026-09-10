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
    
    # Phase 6: WACC 異常值防護閥標記
    is_flagged = models.BooleanField(default=False, db_index=True, help_text="是否觸發異常防護閥 (如 WACC 超出 [3%, 20%] 等)")
    flag_reason = models.TextField(blank=True, default='', help_text="異常原因或防護備註")
    
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


from django.core.serializers.json import DjangoJSONEncoder

class ValuationAssumptionHistory(models.Model):
    """
    SCD Type 2 (Slowly Changing Dimension Type 2) 估值假設歷史版本控制表
    維護估值假設快照 (wacc, g, roic, tax_rate, margins) 與估值結果歷史可追溯性。
    """
    symbol = models.CharField(max_length=20, db_index=True)
    market = models.CharField(max_length=10, choices=[('TW', 'Taiwan'), ('US', 'USA')])
    version = models.PositiveIntegerField(default=1)
    effective_date = models.DateField(help_text="SCD2 生效起始日期")
    end_date = models.DateField(null=True, blank=True, help_text="SCD2 生效結束日期（最新版本為 NULL）")
    is_current = models.BooleanField(default=True, db_index=True, help_text="是否為當前最新有效版本")
    
    # 假設與估值快照 (使用 DjangoJSONEncoder 支援 Decimal/Date 精度)
    assumptions = models.JSONField(default=dict, encoder=DjangoJSONEncoder, help_text="估值假設快照 (wacc, g, roic, tax_rate, margin, etc.)")
    valuation_snapshot = models.JSONField(default=dict, blank=True, encoder=DjangoJSONEncoder, help_text="當期估值結果快照 (dcf_per_share, fair_value, ev, net_debt, etc.)")
    change_reason = models.CharField(max_length=255, blank=True, default='', help_text="版本變更原因 (例如：季報更新、分析師調校、批量重算)")
    
    # Phase 6: WACC 異常值防護閥歷史追蹤
    is_flagged = models.BooleanField(default=False, db_index=True, help_text="是否包含異常參數標記")
    flag_reasons = models.JSONField(default=list, blank=True, encoder=DjangoJSONEncoder, help_text="異常原因標籤列表")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'valuation_assumption_history'
        ordering = ['-version', '-effective_date']
        indexes = [
            models.Index(fields=['symbol', 'market', 'is_current']),
            models.Index(fields=['symbol', 'market', 'version']),
            models.Index(fields=['effective_date']),
        ]

    def __str__(self):
        status = "Current" if self.is_current else f"Archived ({self.end_date})"
        return f"{self.symbol} ({self.market}) v{self.version} [{status}]"
