import pandas as pd
files = [
    r'c:\Users\yashv_p8oa5q\Desktop\chainguard\Global_Aviation_Route_Database.xlsx',
    r'c:\Users\yashv_p8oa5q\Desktop\chainguard\India_Road_Network_Database.xlsx',
    r'c:\Users\yashv_p8oa5q\Desktop\chainguard\Maritime_vessel_database.xlsx'
]
import sys
with open('explore_output_utf8.txt', 'w', encoding='utf-8') as f:
    for file in files:
        try:
            df = pd.read_excel(file)
            f.write(f'\n=== FILE: {file} ===\n')
            f.write(f'Columns: {list(df.columns)}\n')
            f.write(f'Shape: {df.shape}\n')
            f.write(f'Top 2 rows:\n')
            for i, row in enumerate(df.head(2).to_dict(orient='records')):
                f.write(f'  {i}: {row}\n')
        except Exception as e:
            f.write(f'Error reading {file}: {e}\n')
