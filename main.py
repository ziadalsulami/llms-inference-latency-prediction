from datasets import load_dataset
import pandas as pd
ds = load_dataset('rbgo/llm-inference-benchmark')


split = ds['train'].train_test_split(train_size=0.2, seed=42)

df_train = split['train'].to_pandas()
df_test = split['test'].to_pandas()

print(df_train.head())