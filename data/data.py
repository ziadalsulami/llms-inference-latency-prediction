from datasets import load_dataset

def load():
    ds = load_dataset('rbgo/llm-inference-benchmark')
    split = ds['train'].train_test_split(test_size=0.2, seed=42)
    df_train = split['train'].to_pandas()
    df_test = split['test'].to_pandas()

    return df_train, df_test


