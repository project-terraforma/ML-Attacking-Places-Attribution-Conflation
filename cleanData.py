import pandas as pd 

read_data = pd.read_parquet('project_b_samples_2k.parquet')

print(read_data.info())
print(read_data.head(100))