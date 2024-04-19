import sqlite3
import pandas as pd
import csv
from constants import EXTRACTED_DATA_PATH



DB_CONNECTION = "./nela_local_icwsm.db"

class DbClient:
    def __init__(self):
        self._connection = sqlite3.connect(DB_CONNECTION)

    def _filter_row_forbidden_chars(self, row):
        ret_items = []
        for word in row:
            w = str(word)
            clean_string = w.strip()
            clean_string = clean_string.replace('\u2028',' ').replace('\u2029', ' ')
            ret_items.append(clean_string)
        return tuple(ret_items)

    def query(self, query, fetch_all=False, fetch_size=10,params=()):
        cursor = self._connection.execute(query, params)
        header = [column[0] for column in cursor.description]
        if fetch_all:
            data = cursor.fetchall()
        else:
            data = cursor.fetchmany(fetch_size)
        
        return (header, data)

    def export_to_csv(self, filename, query_results):
        #https://www.geeksforgeeks.org/writing-csv-files-in-python/
        with open(filename, 'w', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file,delimiter='|')
            csv_writer.writerow(query_results[0])
            for row in query_results[1]:
                filter_res = self._filter_row_forbidden_chars(row)
                csv_writer.writerow(row)

if __name__ == "__main__":
    aggregates = """
    SELECT
        ROUND(AVG(d.white_pct),2) AS avg_white_pop_pct,
        ROUND(AVG(d.median_hh_inc),2) AS avg_median_hh_inc,
        ROUND(AVG(d.lesscollege_pct),2) AS avg_non_college_pct
    FROM
        demographics AS d
    """
    master_data = f"""
    SELECT 
        a.title AS title,
        a.content AS content,
        o.county AS county,
        o.state AS state,
        o.source AS source,
        CASE d.white_pct < ? THEN 0 ELSE 1 END AS avg_white_pop_pct,
        CASE d.median_hh_inc < ? THEN 0 ELSE 1 END AS avg_median_hh_inc,
        CASE d.lesscollege_pct < ? THEN 0 ELSE 1 END AS avg_non_college_pct,
    FROM 
        articles AS a
    JOIN 
        outlets AS o USING(sourcedomain_id)
    LEFT JOIN 
        demographics AS d ON o.fips = d.fips
    """
    master_data_no_content = f"""
    SELECT 
        a.title AS title,
        o.county AS county,
        o.state AS state,
        o.source AS source,
        d.total_population AS total_population,
        CASE WHEN d.white_pct < ? THEN 0 ELSE 1 END AS avg_white_pop_pct,
        CASE WHEN d.median_hh_inc < ? THEN 0 ELSE 1 END AS avg_median_hh_inc,
        CASE WHEN d.lesscollege_pct < ? THEN 0 ELSE 1 END AS avg_non_college_pct
    FROM 
        articles AS a
    JOIN 
        outlets AS o USING(sourcedomain_id)
    LEFT JOIN 
        demographics AS d ON o.fips = d.fips
    WHERE
        d.white_pct IS NOT "None"
        AND d.median_hh_inc IS NOT "None"
        AND d.lesscollege_pct IS NOT "None"
    """


