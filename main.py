import os
from extract_training_data import DbClient
from preprocess_data import DataProcessor

import pandas as pd
import csv
import sqlite3
from transformers import AutoModel, AutoTokenizer, AutoConfig
from constants import EXTRACTED_DATA_PATH, PROCESSED_DATA_PATH
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_raw_data(db_client, fetch_all = False, fetch_size= 5000):
    aggregates = """
    SELECT
        ROUND(AVG(d.white_pct),2) AS avg_white_pop_pct,
        ROUND(AVG(d.median_hh_inc),2) AS avg_median_hh_inc,
        ROUND(AVG(d.lesscollege_pct),2) AS avg_non_college_pct
    FROM
        demographics AS d
    """
    agg_query = db_client.query(
        query=aggregates,
        fetch_all=True
    )
    agg_res_tup = agg_query[1][0]
    master_data_no_content = f"""
    SELECT 
        (a.title || " " || REPLACE(REPLACE(a.content, '\r\n', ' '), '\n', ' ')) AS content,
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
    query_result = db_client.query(    
    query = master_data_no_content,
    fetch_all = fetch_all,
    fetch_size= fetch_size,
    params=agg_res_tup
    )
    db_client.export_to_csv(EXTRACTED_DATA_PATH, query_result)
    

def pre_process_export():
    data_processor = DataProcessor()
    data_processor.save_processed_data()


if __name__ == '__main__':
    #extract data from sqlite3 DB
    if not os.path.exists(EXTRACTED_DATA_PATH):
        db_client = DbClient()
        extract_raw_data(db_client,fetch_all=False, fetch_size=10000)
    #process data for output()
    if not os.path.exists(PROCESSED_DATA_PATH):
        pre_process_export()

    #load and get evaulation on ovr-svm
    # Load the model from disk
    clf = load('ovr-svm.joblib')
    vectorizer = load('tfidf.joblib')
    # Now you can use clf to make predictions
    test_text = ["Trump goes to Zoo","sleepy Joe has a bad knee","Biden goes to Zoo"]
    test_vector = vectorizer.transform(test_text)
    prediction = clf.predict(test_vector)
    print(prediction)