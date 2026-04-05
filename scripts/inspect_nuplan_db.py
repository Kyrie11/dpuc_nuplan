from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpuc.data.nuplan_sqlite import connect, list_tables, fetch_log_meta
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--db', type=str, required=True)
args = parser.parse_args()
conn = connect(args.db)
print('tables:', list_tables(conn))
print('log_meta:', fetch_log_meta(conn))
for table in list_tables(conn):
    row = conn.execute(f'SELECT * FROM {table} LIMIT 1').fetchone()
    if row is not None:
        print('\n#', table)
        print(dict(row).keys())
conn.close()
