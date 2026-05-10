import pandas as pd
import os

path = "E:/Infinity/mydjango/demo/newsapp/news_data/2303.xlsx"
df = pd.read_excel(path)
print(f"Row count: {len(df)}")
print("Column Names (Final):")
for col in df.columns:
    print(f"  repr={repr(col)}")
