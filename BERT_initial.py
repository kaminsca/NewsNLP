import transformers
import numpy as np
from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset('./master_data_no_article_content.csv')