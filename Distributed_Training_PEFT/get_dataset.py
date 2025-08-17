import pandas as pd
from datasets import Dataset
import kagglehub
import os
import glob

def load_data():
    """Load Jigsaw ACRC dataset from Kaggle or local files"""
    # Check if running on Kaggle
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        # Running on Kaggle
        base_path = "/kaggle/input/jigsaw-agile-community-rules/"
        df_train = pd.read_csv(f"{base_path}*train*.csv")
        df_test = pd.read_csv(f"{base_path}*test*.csv")
    else:
        # Running locally
        base_path = "../data/tmp/"
        
        # Find all train files
        train_files = glob.glob(f"{base_path}*train*.csv")
        if train_files:
            train_dfs = [pd.read_csv(file) for file in train_files]
            df_train = pd.concat(train_dfs, ignore_index=True)
            print(f"Concatenated {len(train_files)} train files: {train_files}")
        else:
            raise FileNotFoundError(f"No train files found in {base_path}")
        
        # Find all test files
        test_files = glob.glob(f"{base_path}*test*.csv")
        if test_files:
            test_dfs = [pd.read_csv(file) for file in test_files]
            df_test = pd.concat(test_dfs, ignore_index=True)
            print(f"Concatenated {len(test_files)} test files: {test_files}")
        else:
            raise FileNotFoundError(f"No test files found in {base_path}")

    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    print(df_train.columns)
            
    req_cols=['subreddit', 'rule', 'positive_example_1', 'negative_example_1', 'positive_example_2',
           'negative_example_2', 'test_comment', 'violates_rule']

    df_train=df_train[req_cols]
    df_test=df_test[req_cols]

    for col in req_cols:
        dropped_rows = df_train[df_train[col].isna()].shape[0]
        print(f"{col}: {dropped_rows} rows would be dropped")
        
    df_train = df_train[req_cols].dropna()
    df_test = df_test[req_cols].dropna()

    print(f"Using path: {base_path}")
    print("\n After dropping:")
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")

    df_train["violates_rule"] = df_train["violates_rule"].astype(str)
    df_test["violates_rule"] = df_test["violates_rule"].astype(str)

    valid_values = {"True", "False"}
    df_train = df_train[df_train["violates_rule"].isin(valid_values)]
    df_test  = df_test[df_test["violates_rule"].isin(valid_values)]
    print("\n After checking True/False:")
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    
    return df_train, df_test

def formatting_prompts_func(examples):
    """
    Format Reddit moderation dataset for Alpaca training - matches inference format exactly
    """
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""
    
    texts = []
    
    for i in range(len(examples['subreddit'])):
        # Create instruction - exactly as in inference
        instruction = f"""You are a really experienced moderator for the subreddit /r/{examples['subreddit'][i]}. 
Your job is to determine if the following reported comment violates the given rule.
Answer with only "True" or "False"."""
        
        # Create input - exactly as in inference
        input_text = f"""Rule: {examples['rule'][i]}
Example 1:
{examples['positive_example_1'][i]}
Rule violation: True
Example 2:
{examples['negative_example_1'][i]}
Rule violation: False
Example 3:
{examples['positive_example_2'][i]}
Rule violation: True
Example 4:
{examples['negative_example_2'][i]}
Rule violation: False
Test sentence:
{examples['test_comment'][i]}"""
        
        # Response is already "True" or "False" string
        response = examples['violates_rule'][i]
                
        # Format the complete prompt
        text = alpaca_prompt.format(instruction, input_text, response)
        texts.append(text)
    
    return {"text": texts}

def build_dataset():
    """
    Build both train and test datasets using the new Alpaca format
    """
    df_train, df_test = load_data()
    
    train_dataset = Dataset.from_pandas(df_train)
    train_dataset = train_dataset.map(
        lambda examples: formatting_prompts_func(examples), 
        batched=True
    )
    
    test_dataset = Dataset.from_pandas(df_test)
    test_dataset = test_dataset.map(
        lambda examples: formatting_prompts_func(examples), 
        batched=True
    )
    
    return train_dataset, test_dataset