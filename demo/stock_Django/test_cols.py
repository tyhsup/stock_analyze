import os
import django
import sys

sys.path.append(r"e:\Infinity\mydjango")
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mydjango.settings')
django.setup()

from demo.stock_Django.mySQL_OP import OP_Fun

op = OP_Fun()
try:
    with op.engine.connect() as conn:
        result = conn.execute("SHOW COLUMNS FROM stock_investor")
        cols = [row[0] for row in result]
        print("Columns in stock_investor:")
        for c in cols:
            print(c)
except Exception as e:
    print("Error:", e)
