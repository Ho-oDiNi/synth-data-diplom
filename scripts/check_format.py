import pandas as pd
df = pd.read_parquet("./real_data/step_0192_real.parquet")
print(df.dtypes)