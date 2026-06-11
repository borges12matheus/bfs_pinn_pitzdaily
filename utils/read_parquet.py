import pandas as pd

df = pd.read_parquet("dataset_pitzDaily_with_vorticity.parquet")
print(df.columns)