import pandas as pd
import sys
import os

PARQUET_FILE = 'project_b_samples_2k.parquet'

def read_dataset(parquet_path: str):
	csv_fallback = os.path.splitext(parquet_path)[0] + '.csv'
	try:
		df = pd.read_parquet(parquet_path)
		return df
	except ImportError as e:
		# pandas raises ImportError when no parquet engine is available
		print('Parquet support is not available in this environment.')
		print('Install a parquet engine like pyarrow (recommended) or fastparquet:')
		print('  pip3 install pyarrow')
		print('  or')
		print('  pip3 install fastparquet')
		print("Or install both via 'pip3 install -r requirements.txt'.")
		if os.path.exists(csv_fallback):
			print(f"Falling back to CSV file: {csv_fallback}")
			return pd.read_csv(csv_fallback)
		else:
			print('No CSV fallback found. Exiting.')
			sys.exit(1)
	except Exception as e:
		print(f'Error reading {parquet_path}: {e}')
		# try CSV fallback if available
		if os.path.exists(csv_fallback):
			print(f"Attempting CSV fallback: {csv_fallback}")
			return pd.read_csv(csv_fallback)
		raise


if __name__ == '__main__':
	read_data = read_dataset(PARQUET_FILE)

	print(read_data.info())
	print(read_data.head(100))