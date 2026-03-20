from django import template
from django.utils.safestring import mark_safe
import math

register = template.Library()

@register.filter(name='sci_notation')
def sci_notation(value):
    """
    Format numbers using financial abbreviations (K/M/B/T), common in financial reports.
    e.g.  1,234 → 1,234
         123,456 → 123.5K
     1,234,567 → 1.23M
    12,345,678,901 → 12.35B
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(num) or math.isinf(num):
        return str(value)

    abs_num = abs(num)
    sign = '-' if num < 0 else ''

    if abs_num >= 1_000_000_000_000:
        return mark_safe(f"{sign}{abs_num / 1_000_000_000_000:.2f}兆" if sign == '-' else f"{abs_num / 1_000_000_000_000:.2f}兆") # Actually sci_notation is for USD usually, but let's keep it consistent with format_large_number if possible.
    elif abs_num >= 1_000_000_000:
        return mark_safe(f"{sign}{abs_num / 1_000_000_000:.2f}B")
    elif abs_num >= 1_000_000:
        return mark_safe(f"{sign}{abs_num / 1_000_000:.2f}M")
    elif abs_num >= 1_000:
        return mark_safe(f"{sign}{abs_num / 1_000:.1f}K")
    else:
        return mark_safe(f"{num:,.0f}")

@register.filter(name='to_percentage')
def to_percentage(value, decimals=2):
    """
    Format a decimal value (e.g. 0.15) as a percentage string (e.g. 15.00%).
    """
    try:
        num = float(value)
        return f"{num * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return value
@register.filter(name='multiply')
def multiply(value, arg):
    """Multiplies the value by the argument."""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return value

@register.filter(name='divide')
def divide(value, arg):
    """Divides the value by the argument."""
    try:
        return float(value) / float(arg) if float(arg) != 0 else 0
    except (ValueError, TypeError):
        return value
