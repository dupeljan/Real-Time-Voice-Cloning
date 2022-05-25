import sqlite3
import pandas as pd

from collections import Counter
from app import get_db_connection


def main():
    conn = get_db_connection()
    data = conn.execute('SELECT mode, gen_type, avg(score) FROM records'
                        ' r where r.user = "dupeljan" group by mode, gen_type').fetchall() 
    conn.close()
    result = [dict(x) for x in data]
    df = pd.DataFrame(result)
    print(df)

if __name__ == '__main__':
    main()