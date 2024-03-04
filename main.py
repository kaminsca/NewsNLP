import pandas as pd
import csv


def import_data(filepath):
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


if __name__ == '__main__':
    # NELA-Local data comes from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YHWTFC
    import_data('data/nela_ps_newsdata_sample.csv')