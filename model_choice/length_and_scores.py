# analyze_lengths.py

import datasets
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 
#: 'output/GLM-4-32B-0414-Gemma-3-27b-it'
OUTPUT_DIR = 'output/your_judge_model-your_tested_model'
TARGET_SCORE = 'answerable_score' #, 'answerable_score', 'faithful_score' 

print(f" '{OUTPUT_DIR}'...")
try:
 final_dataset = datasets.load_from_disk(OUTPUT_DIR)
 print("!")
 print(":", final_dataset)
 print(":", final_dataset.column_names)
except FileNotFoundError:
 print(f": '{OUTPUT_DIR}'..")
 exit()

# Pandas DataFrame
df = final_dataset.to_pandas()

# 
df[TARGET_SCORE] = pd.to_numeric(df[TARGET_SCORE], errors='coerce')
df['query_length'] = pd.to_numeric(df['query_length'], errors='coerce')
df['document_length'] = pd.to_numeric(df['document_length'], errors='coerce')
df['answer_length'] = pd.to_numeric(df['answer_length'], errors='coerce')

df.dropna(subset=[TARGET_SCORE, 'query_length', 'document_length', 'answer_length'], inplace=True)

print(f"\n--- '{TARGET_SCORE}' ---")

# --- 1:, ---
print("\n:")
# ( 1, 2, 3, 4, 5)
length_stats = df.groupby(TARGET_SCORE)[['query_length', 'document_length', 'answer_length']].agg(['mean', 'median', 'std', 'count'])
print(length_stats)


# --- 2: ---
# (Box Plot) 
plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(f'Token vs. {TARGET_SCORE}', fontsize=16)

# Query Length
sns.boxplot(data=df, x=TARGET_SCORE, y='query_length', ax=axes[0])
axes[0].set_title('Query Length vs. Score')
axes[0].set_xlabel(f'{TARGET_SCORE}')
axes[0].set_ylabel('Query Token Length')

# Document Length
sns.boxplot(data=df, x=TARGET_SCORE, y='document_length', ax=axes[1])
axes[1].set_title('Document Length vs. Score')
axes[1].set_xlabel(f'{TARGET_SCORE}')
axes[1].set_ylabel('Total Document Token Length')
# axes[1].set_yscale('log') 

# Answer Length
sns.boxplot(data=df, x=TARGET_SCORE, y='answer_length', ax=axes[2])
axes[2].set_title('Answer Length vs. Score')
axes[2].set_xlabel(f'{TARGET_SCORE}')
axes[2].set_ylabel('Answer Token Length')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# 
analysis_img_path = os.path.join(OUTPUT_DIR, f'analysis_{TARGET_SCORE}_vs_lengths.png')
plt.savefig(analysis_img_path)
print(f"\n: {analysis_img_path}")
plt.show()

# --- 3: ---
print("\n:")
correlation_matrix = df[[TARGET_SCORE, 'query_length', 'document_length', 'answer_length']].corr()
print(correlation_matrix)

# 
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Scores and Lengths')
plt.show()