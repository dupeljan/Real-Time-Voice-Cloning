import sqlite3
from app import get_db_connection


def main():
    conn = get_db_connection()
    data = conn.execute('SELECT * FROM records').fetchall() 
    conn.close()
    for d in data:
        print(dict(d))


if __name__ == '__main__':
    main()