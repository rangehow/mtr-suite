# -------------------------------------------------------------------
# Step 0: Import necessary libraries
# -------------------------------------------------------------------
import pandas as pd
import datasets
import numpy as np
import pingouin as pg # pingouin 

# -------------------------------------------------------------------
# Step 1: Load the dataset and convert to a Pandas DataFrame
# -------------------------------------------------------------------
print("--- Step 1: Loading Dataset ---")
DATASET_PATH = "/path/to/mtr/tempfile/chatrag/Command-a_qrecc"
# DATASET_PATH = ""
dataset = datasets.load_from_disk(DATASET_PATH)
df = dataset.to_pandas()
print(f"Dataset '{DATASET_PATH}' loaded successfully!")
print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
print("-" * 50)

# -------------------------------------------------------------------
# Step 2: Extract question and response lengths
# -------------------------------------------------------------------
print("\n--- Step 2: Extracting Question and Response Lengths ---")
def get_content_length(messages, role):
 try:
 for message in messages:
 if isinstance(message, dict) and message.get('role') == role:
 content = message.get('content')
 return len(str(content)) if content else 0
 return None
 except TypeError:
 return None
df['question_length'] = df['messages'].apply(lambda x: get_content_length(x, 'user'))
df['response_length'] = df['messages'].apply(lambda x: get_content_length(x, 'assistant'))
print("Successfully created 'question_length' and 'response_length' columns.")
print("-" * 50)

# -------------------------------------------------------------------
# Step 3: Data inspection and cleaning
# -------------------------------------------------------------------
print("\n--- Step 3: Data Inspection and Cleaning ---")
columns_of_interest = [
 'question_length', 
 'response_length', 
 'faithful_score', 
 'quality_score', 
 'answerable_score', 
 'tag_score'
]
analysis_df = df[columns_of_interest].dropna()
print(f"Cleaned dataset size for analysis: {analysis_df.shape[0]} rows")
print("\nDescriptive statistics:")
print(analysis_df.describe())
print("-" * 50)

# -------------------------------------------------------------------
# Step 4: Calculate Correlation Matrix with P-values ()
# -------------------------------------------------------------------
print("\n--- Step 4: Calculating Correlation with P-values ---")

#: pg.rcorr(dataframe) dataframe.rcorr()
# MultiIndex DataFrame, ('r',...), ('p-val',...)
corr_results = pg.rcorr(analysis_df, method='pearson')

print("Raw output from pg.rcorr():")
print(corr_results)
print("-" * 50)

