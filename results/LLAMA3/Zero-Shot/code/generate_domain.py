import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../Updated_Cleaned-Zero-Shot_Results.csv')
df_base = pd.read_csv('../Cleaned_Zero_Shot_BaseLine.csv')

cosine_baseline = df_base['Cosine'].mean()
bert_baseline   = df_base['BERT'].mean()

domain_models = df["File"].unique()

for domain in domain_models:
    df_domain = df[df["File"] == domain]
    df_base_domain = df_base[df_base["File"] == domain]
    
    cos_baseline = df_base_domain["Cosine"].mean()
    bert_baseline = df_base_domain["BERT"].mean()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.boxplot(x="Solution", y="Cosine", data=df_domain, ax=axes[0])
    axes[0].axhline(cos_baseline, color="red", linestyle="--", label="Baseline")
    axes[0].set_title(f"Cosine for {domain}")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend()
    
    sns.boxplot(x="Solution", y="BERT", data=df_domain, ax=axes[1])
    axes[1].axhline(bert_baseline, color="red", linestyle="--", label="Baseline")
    axes[1].set_title(f"BERT for {domain}")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend()
    
    plt.tight_layout()
    
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/zero_shot_boxplots_{domain}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)