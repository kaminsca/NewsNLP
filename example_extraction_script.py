import sqlite3

'''
This script provides several simple examples of extracting data from the SQLite3 database of NELA-Local.
'''

connection = sqlite3.connect("./nela_local_icwsm.db")  # provide path to database file


def extract_article_data():
    cursor = connection.execute("SELECT * FROM articles;")  # get all articles in database
    for row in cursor.fetchall():  # go through row by row
        print(row)  # print row, which by default is a tuple.
        # At this point you can write it out or pass to analysis script


def extract_number_of_articles_per_FIPS():
    cursor = connection.execute("""SELECT COUNT(articles.content), outlets.fips 
                                FROM articles JOIN outlets ON articles.sourcedomain_id = outlets.sourcedomain_id
                                GROUP BY outlets.fips;""")  # get article content and matching FIPS

    for num_content, fips in cursor.fetchall():  # go through row by row
        print(f"{fips} has {num_content} articles")

def count_fips_per_article():
    cursor = connection.execute("""SELECT outlets.county, count(articles.content)
                                FROM articles JOIN outlets ON articles.sourcedomain_id = outlets.sourcedomain_id
                                GROUP BY outlets.county;""")  # get article content and matching FIPS


    for i in range(10):
        record = cursor.fetchone()
        article_id, fips_count = record
        print(f"{article_id} has {fips_count} associated fips")

def extract_politics_of_outlet():
    cursor = connection.execute("""SELECT outlets.sourcedomain_id, politics.logodds_Trump20 
                                FROM outlets JOIN politics ON outlets.fips = politics.fips;""")  #get outlet and matching log-odds of voting for Trump in 2020

    for sourcedomain_id, logodds_Trump20 in cursor.fetchall():
        print(f"The audience of {sourcedomain_id} had {logodds_Trump20} log-odds of voting for Trump.")

# Main
#extract_article_data()

#extract_number_of_articles_per_FIPS()

#extract_politics_of_outlet()
count_fips_per_article()