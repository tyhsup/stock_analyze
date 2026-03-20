
import os

file_path = r'e:\Infinity\mydjango\demo\templates\home.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the ROE split
target1 = '''                                {% if financial_summary.roe %}{{ financial_summary.roe|floatformat:2 }}%{% else %}--{%
                                endif %}'''
replacement1 = '                                {% if financial_summary.roe %}{{ financial_summary.roe|floatformat:2 }}%{% else %}--{% endif %}'

# Fix the Gross Margin split
target2 = '''                                {% if financial_summary.gross_margin %}{{ financial_summary.gross_margin|floatformat:2
                                }}%{% else %}--{% endif %}'''
replacement2 = '                                {% if financial_summary.gross_margin %}{{ financial_summary.gross_margin|floatformat:2 }}%{% else %}--{% endif %}'

# Fix the Revenue Growth split
target3 = '''                                {% if financial_summary.revenue_growth %}{{
                                financial_summary.revenue_growth|floatformat:2 }}%{% else %}--{% endif %}'''
replacement3 = '                                {% if financial_summary.revenue_growth %}{{ financial_summary.revenue_growth|floatformat:2 }}%{% else %}--{% endif %}'

# Perform replacements
new_content = content.replace(target1, replacement1)
new_content = new_content.replace(target2, replacement2)
new_content = new_content.replace(target3, replacement3)

if new_content != content:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Success: Fixed split tags in home.html")
else:
    # Try more flexible matching if exact match fails
    print("Exact match failed, trying flexible match...")
    import re
    new_content = re.sub(r'\{%\s+if financial_summary\.roe %\}\{\{ financial_summary\.roe\|floatformat:2 %\}\{% else %\}--\{%\s+endif %\}', 
                         '{% if financial_summary.roe %}{{ financial_summary.roe|floatformat:2 }}%{% else %}--{% endif %}', content, flags=re.MULTILINE | re.DOTALL)
    
    # Just merge all broken tags manually if needed
    new_content = content.replace('{% else %}--{% \n                                endif %}', '{% else %}--{% endif %}')
    new_content = new_content.replace('{% else %}--{%\n                                endif %}', '{% else %}--{% endif %}')
    new_content = new_content.replace('{% \n                                endif %}', '{% endif %}')
    new_content = new_content.replace('{%\n                                endif %}', '{% endif %}')
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Success: Fixed split tags using fallback replacement")
    else:
        print("Error: Could not find split tags even with fallback")
