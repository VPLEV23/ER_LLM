import transformers
import torch
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import itertools

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def normalized_edit_distance(str1, str2):
    return 1 - SequenceMatcher(None, str1, str2).ratio()

def dice_coefficient(a, b):
    a_bigrams = set(zip(a, a[1:]))
    b_bigrams = set(zip(b, b[1:]))
    if len(a_bigrams) + len(b_bigrams) == 0:
        return 1.0 
    overlap = len(a_bigrams & b_bigrams)
    return 2 * overlap / (len(a_bigrams) + len(b_bigrams))

def jaccard_similarity(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    if len(a | b) == 0:
        return 1.0  
    return len(a & b) / len(a | b)

def cosine_similarity_text(str1, str2):
    vectorizer = CountVectorizer().fit([str1, str2])
    vecs = vectorizer.transform([str1, str2]).toarray()
    return cosine_similarity(vecs)[0][1]

txt_path = "./extracted_text.txt"
ground_truth_path = "./UPD_Ground_truth.txt"
extracted_text = read_text_from_file(txt_path)
ground_truth = read_text_from_file(ground_truth_path)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16, "do_sample": True},
    device_map="auto"
)

task_description = '''You are an ER diagram expert. You are tasked with analyzing a text that describes database entities and their relationships. Your objectives are to:
1. Identify all entities (tables) mentioned in the text.
2. Extract attributes (columns) for each entity and determine their data types if mentioned.
3. Identify primary keys and foreign keys to understand the relationships between entities.
4. Identify relationships between entities (one-to-one, one-to-many, many-to-many) using the notation:
   - Entity01 }|..|| Entity02
   - Entity03 }o..o| Entity04
   - Entity05 ||--o{ Entity06
   - Entity07 |o--|| Entity08
5. If some attributes are not in the text, don't add them to the diagram.

6. Generate a PlantUML script to represent these entities and relationships in an ER diagram. The output should be solely the PlantUML code.'''
full_input_text = task_description + "\n" + extracted_text

temperature_values = [0.7, 1.0, 1.3]
top_k_values = [0, 50]
top_p_values = [0.9, 1.0]
repetition_penalty_values = [1.0, 1.1, 1.2]
max_new_tokens_values = [512, 1024, 2048, 3072, 4096] 

parameter_sets = []
param_grid = list(itertools.product(
    temperature_values,
    top_k_values,
    top_p_values,
    repetition_penalty_values,
    max_new_tokens_values
))

for idx, (temperature, top_k, top_p, repetition_penalty, max_new_tokens) in enumerate(param_grid):
    params = {
        "temperature": temperature,
        "top_k": top_k if top_k > 0 else None, 
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
    }
    parameter_sets.append(params)

def generate_with_params(params):
    try:
        response = pipeline(
            full_input_text,
            max_new_tokens=params.get("max_new_tokens", 512),
            temperature=params.get("temperature", 1.0),
            top_k=params.get("top_k", None),
            top_p=params.get("top_p", None),
            repetition_penalty=params.get("repetition_penalty", None),
            return_full_text=False 
        )[0]["generated_text"]

        if full_input_text in response:
            generated_output = response.replace(full_input_text, '').strip()
        else:
            generated_output = response.strip()

        return generated_output
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory error occurred.")
        torch.cuda.empty_cache()
        return "Error: CUDA out of memory"
    except Exception as e:
        print(f"Error during generation: {e}")
        return f"Error: {e}"
results = []

for idx, params in enumerate(parameter_sets):
    print(f"Running generation with parameter set {idx + 1}/{len(parameter_sets)}: {params}")
    generated_output = generate_with_params(params)

    if generated_output.startswith("Error:"):
        results.append({
            "Parameter Set": idx + 1,
            "Parameters": params,
            "Generated Output": generated_output,
            "Normalized Edit Distance": None,
            "Jaccard Similarity": None,
            "Cosine Similarity": None,
            "Dice Coefficient": None,
            "BERT Score (F1)": None
        })
        continue

    print(f"Generated Output for parameter set {idx + 1}:\n{generated_output}\n")

    norm_edit_dist = normalized_edit_distance(generated_output, ground_truth)
    jaccard_sim = jaccard_similarity(generated_output, ground_truth)
    cosine_sim = cosine_similarity_text(generated_output, ground_truth)
    dice_coef = dice_coefficient(generated_output, ground_truth)
    P, R, F1 = bert_score([generated_output], [ground_truth], lang="en", verbose=False)
    bert_f1 = F1.mean().item()

    results.append({
        "Parameter Set": idx + 1,
        "Parameters": params,
        "Generated Output": generated_output,
        "Normalized Edit Distance": norm_edit_dist,
        "Jaccard Similarity": jaccard_sim,
        "Cosine Similarity": cosine_sim,
        "Dice Coefficient": dice_coef,
        "BERT Score (F1)": bert_f1
    })

    df_results = pd.DataFrame(results)
    df_results.to_csv("evaluation_results_with_output.csv", index=False)

df_results = pd.DataFrame(results)
print(df_results)

df_results.to_csv("evaluation_results_with_output.csv", index=False)
