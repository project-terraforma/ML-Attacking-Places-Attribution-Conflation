import pandas as pd 
import json

def get_data():
    read_data_omf = pd.read_parquet('project_b_samples_2k.parquet')
    read_data_yelp = pd.read_json('yelp_academic_dataset_business.json', lines=True)


def main():
    get_data()

if __name__ == "__main__":
    main()