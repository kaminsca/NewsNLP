import pandas as pd
import csv
import sqlite3


def import_csv(filepath):
    print("importing")
    with open(filepath, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names: {", ".join(row)}')
                line_count += 1
            else:
                if len(row) > 0:
                    print(row)
                    # print(f'\tID: {row[0]}, date: {row[1]}, source: {row[2]}, title: {row[3]}, location: {row[11]}')
                line_count += 1
        print(f'Processed {line_count} lines')

def import_db(db):
     conn = sqlite3.connect(db)
     cursor = conn.cursor()
     cursor.execute("SELECT * FROM articles")
     rows = cursor.fetchall()
     count = 0
     for row in rows:
         if count > 20:
             break
         print(row)
         count += 1
     cursor.close()
     conn.close()

def show_db_tables(db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = cursor.fetchall()
    for name in table_names:
        print(name[0])
    cursor.close()
    conn.close()

if __name__ == '__main__':
    # NELA- pink slime data comes from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YHWTFC
    # import_csv('data/nela_ps_newsdata_sample.csv')

    # NELA-Local data from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GFE66K
    show_db_tables('data/database_compressed/nela_local_icwsm.db')
    import_db('data/database_compressed/nela_local_icwsm.db')
