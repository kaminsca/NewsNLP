import sqlite3
import pandas as pd
import csv



DB_CONNECTION = "./nela_local_icwsm.db"

class dbClient:
    def __init__(self):
        self._connection = sqlite3.connect(DB_CONNECTION)

    def query(self, query, fetch_all=False, fetch_size=10):
        cursor = self._connection.execute(query)
        header = [column[0] for column in cursor.description]
        if fetch_all:
            data = cursor.fetchall()
        else:
            data = cursor.fetchmany(fetch_size)
        
        return (header, data)

    def export_to_csv(self, filename, query_results):
        #https://www.geeksforgeeks.org/writing-csv-files-in-python/
        with open(filename, 'w', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(query_results[0])
            csv_writer.writerows(query_results[1])

if __name__ == "__main__":
    db_client = dbClient()
    master_data = """
    SELECT 
        a.article_id AS article_id,
        a.title AS title,
        a.content AS content,
        o.county AS county,
        o.state AS state,
        o.source AS source,
        d.total_population AS total_population,
        d.white_pct AS white_pct,
        d.median_hh_inc AS median_household_income,
        d.lesscollege_pct AS non_college_pct
    FROM 
        articles AS a
    JOIN 
        outlets AS o USING(sourcedomain_id)
    LEFT JOIN 
        demographics AS d ON o.fips = d.fips
    """
    master_data_no_content = """
    SELECT 
        a.article_id AS article_id,
        a.title AS title,
        o.county AS county,
        o.state AS state,
        o.source AS source,
        d.total_population AS total_population,
        d.white_pct AS white_pct,
        d.median_hh_inc AS median_household_income,
        d.lesscollege_pct AS non_college_pct
    FROM 
        articles AS a
    JOIN 
        outlets AS o USING(sourcedomain_id)
    LEFT JOIN 
        demographics AS d ON o.fips = d.fips
    """
    query_result = db_client.query(    
    query = master_data_no_content,
    fetch_all = False,
    fetch_size=1
    )
    db_client.export_to_csv("./output/master_data_no_article_content.csv", query_result)

