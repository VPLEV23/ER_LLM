import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from matplotlib.ticker import MaxNLocator

# Fixed list of experiment result CSVs
csv_files = [
    "cot_experiment_results.csv",
    "few_shot_experiment_results.csv",
]

# Main output directory
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

for csv_file in csv_files:
    # Derive experiment folder name
    base_name = os.path.splitext(csv_file)[0].replace('_experiment_results', '')
    out_path = os.path.join(output_dir, base_name)
    os.makedirs(out_path, exist_ok=True)

    print(f"Processing: {csv_file}")
    df = pd.read_csv(csv_file)

    # Convert parameter strings to dicts
    df['Parameters'] = df['Parameters'].apply(ast.literal_eval)
    params_df = df['Parameters'].apply(pd.Series)
    df = pd.concat([df.drop(['Parameters'], axis=1), params_df], axis=1)

    # Remove all-zero results
    df = df[(df['Text Cosine Similarity'] != 0) | (df['BERT F1'] != 0)]

    # Create label combining file and parameter string
    df['File_Params'] = df['File'] + '\n' + df[['temperature', 'top_k', 'top_p', 'max_new_tokens']].astype(str).agg(', '.join, axis=1)

    # --- Combined Boxplots ---
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='File_Params', y='Text Cosine Similarity')
    plt.title('Text Cosine Similarity by File and Parameters')
    plt.xticks(rotation=90)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=len(df['File_Params'].unique())))

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='File_Params', y='BERT F1')
    plt.title('BERT F1 Score by File and Parameters')
    plt.xticks(rotation=90)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=len(df['File_Params'].unique())))

    plt.subplots_adjust(bottom=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f'{base_name}_combined_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Individual Metric Plots by DOCX File ---
    for metric in ['Text Cosine Similarity', 'BERT F1']:
        plt.figure(figsize=(15, 8))
        ax = sns.boxplot(data=df, x='File', y=metric)
        plt.title(f'{metric} by DOCX File')
        plt.xticks(rotation=45)

        # Create parameter range labels for each DOCX
        param_labels = []
        for file in df['File'].unique():
            params = df[df['File'] == file][['temperature', 'top_k', 'top_p', 'max_new_tokens']].drop_duplicates()
            param_str = '\n'.join([
                f"temp={row['temperature']}, top_k={row['top_k']}, top_p={row['top_p']}, tokens={row['max_new_tokens']}"
                for _, row in params.iterrows()
            ])
            param_labels.append(f"{file}\n{param_str}")

        ax.set_xticklabels(param_labels)
        plt.tight_layout()
        filename = f'{base_name}_{metric.lower().replace(" ", "_")}_boxplot.png'
        plt.savefig(os.path.join(out_path, filename), dpi=300, bbox_inches='tight')
        plt.close()

    # --- Summary Table ---
    summary_table = df.groupby(['File']).agg({
        'Text Cosine Similarity': ['mean', 'std', 'count'],
        'BERT F1': ['mean', 'std', 'count'],
        'temperature': 'unique',
        'top_k': 'unique',
        'top_p': 'unique',
        'max_new_tokens': 'unique'
    })

    # Save summary as .txt
    with open(os.path.join(out_path, 'summary.txt'), 'w') as f:
        f.write(summary_table.to_string())

    print(f"✅ Finished: {csv_file}")
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from matplotlib.ticker import MaxNLocator

# Fixed list of experiment result CSVs
csv_files = [
    "cot_experiment_results.csv",
    "few_shot_experiment_results.csv",
    "zero_shot_experiment_results.csv"
]

# Main output directory
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

for csv_file in csv_files:
    # Derive experiment folder name
    base_name = os.path.splitext(csv_file)[0].replace('_experiment_results', '')
    out_path = os.path.join(output_dir, base_name)
    os.makedirs(out_path, exist_ok=True)

    print(f"Processing: {csv_file}")
    df = pd.read_csv(csv_file)

    # Convert parameter strings to dicts
    df['Parameters'] = df['Parameters'].apply(ast.literal_eval)
    params_df = df['Parameters'].apply(pd.Series)
    df = pd.concat([df.drop(['Parameters'], axis=1), params_df], axis=1)

    # Remove all-zero results
    df = df[(df['Text Cosine Similarity'] != 0) | (df['BERT F1'] != 0)]

    # Create label combining file and parameter string
    df['File_Params'] = df['File'] + '\n' + df[['temperature', 'top_k', 'top_p', 'max_new_tokens']].astype(str).agg(', '.join, axis=1)

    # --- Combined Boxplots ---
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='File_Params', y='Text Cosine Similarity')
    plt.title('Text Cosine Similarity by File and Parameters')
    plt.xticks(rotation=90)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=len(df['File_Params'].unique())))

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='File_Params', y='BERT F1')
    plt.title('BERT F1 Score by File and Parameters')
    plt.xticks(rotation=90)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=len(df['File_Params'].unique())))

    plt.subplots_adjust(bottom=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f'{base_name}_combined_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Individual Metric Plots by DOCX File ---
    for metric in ['Text Cosine Similarity', 'BERT F1']:
        plt.figure(figsize=(15, 8))
        ax = sns.boxplot(data=df, x='File', y=metric)
        plt.title(f'{metric} by DOCX File')
        plt.xticks(rotation=45)

        # Create parameter range labels for each DOCX
        param_labels = []
        for file in df['File'].unique():
            params = df[df['File'] == file][['temperature', 'top_k', 'top_p', 'max_new_tokens']].drop_duplicates()
            param_str = '\n'.join([
                f"temp={row['temperature']}, top_k={row['top_k']}, top_p={row['top_p']}, tokens={row['max_new_tokens']}"
                for _, row in params.iterrows()
            ])
            param_labels.append(f"{file}\n{param_str}")

        ax.set_xticklabels(param_labels)
        plt.tight_layout()
        filename = f'{base_name}_{metric.lower().replace(" ", "_")}_boxplot.png'
        plt.savefig(os.path.join(out_path, filename), dpi=300, bbox_inches='tight')
        plt.close()

    # --- Summary Table ---
    summary_table = df.groupby(['File']).agg({
        'Text Cosine Similarity': ['mean', 'std', 'count'],
        'BERT F1': ['mean', 'std', 'count'],
        'temperature': 'unique',
        'top_k': 'unique',
        'top_p': 'unique',
        'max_new_tokens': 'unique'
    })

    # Save summary as .txt
    with open(os.path.join(out_path, 'summary.txt'), 'w') as f:
        f.write(summary_table.to_string())

    print(f"✅ Finished: {csv_file}")
