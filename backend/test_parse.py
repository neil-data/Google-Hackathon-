import pandas as pd
import io
import csv

def parse_aviation(file):
    df = pd.read_excel(file)
    # The whole thing is in the first column
    col = df.columns[0]
    # We can reconstruct a CSV and read it back
    data = [col] + df[col].tolist()
    csv_str = '\n'.join(str(x) for x in data)
    parsed_df = pd.read_csv(io.StringIO(csv_str))
    print("Aviation parsed shape:", parsed_df.shape)
    print("Aviation parsed cols:", list(parsed_df.columns)[:5])

def parse_road(file):
    df = pd.read_excel(file)
    print("Road parsed shape:", df.shape)
    print("Road parsed cols:", list(df.columns)[:5])

def parse_maritime(file):
    df = pd.read_excel(file)
    # Row 0 has the actual headers as comma-separated string in col 0
    header_str = df.iloc[0, 0]
    # some other columns might be captured due to formatting
    # So we combine all rows and read as CSV
    lines = [header_str]
    for idx, row in df.iloc[1:].iterrows():
        # if row[0] is string, just take it
        lines.append(str(row.iloc[0]))
    csv_str = '\n'.join(lines)
    parsed_df = pd.read_csv(io.StringIO(csv_str))
    print("Maritime parsed shape:", parsed_df.shape)
    print("Maritime parsed cols:", list(parsed_df.columns)[:5])

aviation = r'c:\Users\yashv_p8oa5q\Desktop\chainguard\Global_Aviation_Route_Database.xlsx'
road = r'c:\Users\yashv_p8oa5q\Desktop\chainguard\India_Road_Network_Database.xlsx'
maritime = r'c:\Users\yashv_p8oa5q\Desktop\chainguard\Maritime_vessel_database.xlsx'

parse_aviation(aviation)
parse_road(road)
parse_maritime(maritime)
