import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../Updated_Cleaned_COT_Results.csv')
df_base = pd.read_csv('../Cleaned_Baseline_Experiment_Results.csv')

cosine_baseline = df_base['Cosine'].mean()
bert_baseline   = df_base['BERT'].mean()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.boxplot(x='Solution', y='Cosine', data=df, ax=axes[0])
axes[0].axhline(cosine_baseline, color='red', linestyle='--', label='Baseline')
axes[0].set_title("Cosine Scores by Parameter Settings")
axes[0].tick_params(axis='x', rotation=30)   

sns.boxplot(x='Solution', y='BERT', data=df, ax=axes[1])
axes[1].axhline(bert_baseline, color='red', linestyle='--', label='Baseline')
axes[1].set_title("BERT Scores by Parameter Settings")
axes[1].tick_params(axis='x', rotation=30)

axes[0].legend()
axes[1].legend()
plt.tight_layout()

os.makedirs("images", exist_ok=True)
plt.savefig("images/cot_boxplots.png", dpi=300, bbox_inches='tight')


