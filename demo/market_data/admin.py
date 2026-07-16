from django.contrib import admin
from market_data.models import MacroTW, MacroUS

@admin.register(MacroTW)
class MacroTWAdmin(admin.ModelAdmin):
    list_display = ('date', 'metric', 'value')
    list_filter = ('metric',)
    search_fields = ('metric',)

@admin.register(MacroUS)
class MacroUSAdmin(admin.ModelAdmin):
    list_display = ('date', 'metric', 'value')
    list_filter = ('metric',)
    search_fields = ('metric',)

