import os
import torch
import pandas as pd
import logging
from docx import Document
from transformers import AutoTokenizer, LlamaForCausalLM
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_fn
from bert_score import score
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper functions
def read_text_from_docx(docx_path):
    doc = Document(docx_path)
    full_text = [paragraph.text for paragraph in doc.paragraphs]
    return "\n".join(full_text)

def read_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

# def compute_embedding_similarity(reference, generated):
#     ref_embedding = sentence_model.encode(reference, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
#     gen_embedding = sentence_model.encode(generated, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
#     return cosine_similarity_fn(ref_embedding, gen_embedding)[0][0]

def cosine_similarity_text(str1, str2):
    vectorizer = CountVectorizer().fit([str1, str2])
    vecs = vectorizer.transform([str1, str2]).toarray()
    return cosine_similarity_fn(vecs)[0][1]

# Folder paths
docx_folder = "./instructions"
ground_truth_folder = "../finetune_dataset"

# Model setup
model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id + 1
model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto')

task_description = '''You are an ER diagram expert. You are tasked with analyzing a text that describes database entities and their relationships. Your objectives are to:
    1. Identify all entities (tables) mentioned in the text and define them as EMF EClasses.
    2. For each entity, extract attributes (columns) including:
       - Name: Identify attribute names.
       - Data Type: Specify the data type if mentioned in the text (e.g., EString, EInt, EBoolean).
       - Properties: If mentioned, include properties like "required", "default value", etc.
    3. Identify primary keys and foreign keys to understand relationships between entities. 
       - Designate attributes as primary or foreign keys where applicable.
    4. Identify relationships between entities and define them using EMF-compatible syntax:
       - Specify relationship types (one-to-one, one-to-many, or many-to-many).
       - Include role names if provided.
       - Define multiplicities (e.g., 1..1, 0..*, 1..*) and set EReferences to capture relationships.
    5. Exclude any attributes or details not explicitly mentioned in the text.
    6. Generate the output as EMF-compatible code in XMI or Ecorefor mat, ensuring it’s suitable for importing into an EMF model. The output should solely be in EMF-compatible syntax.
    Output should be only the EMF-compatible code for the entities, attributes, and relationships identified in the text. Once you start writing code, do not write any additional text interupting your code.

Example: 1
<ecore:EPackage
    xmlns:ecore="http://www.eclipse.org/emf/2002/Ecore"
    name="CustomerOrderModel"
    nsPrefix="custorder"
    nsURI="http://www.example.com/custorder">
  <eClassifiers xsi:type="ecore:EClass" name="Customer">
    <eStructuralFeatures xsi:type="ecore:EAttribute" name="CustomerID" eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EInt"/>
    <eStructuralFeatures xsi:type="ecore:EAttribute" name="Name" eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EString"/>
    <eStructuralFeatures xsi:type="ecore:EAttribute" name="Email" eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EString"/>
  </eClassifiers>
  <eClassifiers xsi:type="ecore:EClass" name="Order">
    <eStructuralFeatures xsi:type="ecore:EAttribute" name="OrderID" eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EInt"/>
    <eStructuralFeatures xsi:type="ecore:EAttribute" name="CustomerID" eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EInt"/>
    <eStructuralFeatures xsi:type="ecore:EAttribute" name="Total" eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EFloat"/>
  </eClassifiers>
</ecore:EPackage>

Example 2:
<ecore:EPackage
    xmlns:ecore="http://www.eclipse.org/emf/2002/Ecore"
    name="BTMS"
    nsPrefix="btms"
    nsURI="http://www.example.com/btms">
  <eClassifiers xsi:type="ecore:EClass" name="BusVehicle">
    <eStructuralFeatures xsi:type="ecore:EAttribute" name="LicencePlate" eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EString"/>
    <eStructuralFeatures xsi:type="ecore:EAttribute" name="InRepairShop" eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EBoolean" defaultValueLiteral="false"/>
  </eClassifiers>
  <eClassifiers xsi:type="ecore:EClass" name="Driver">
    <eStructuralFeatures xsi:type="ecore:EAttribute" name="ID" eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EInt"/>
    <eStructuralFeatures xsi:type="ecore:EAttribute" name="Name" eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EString"/>
    <eStructuralFeatures xsi:type="ecore:EAttribute" name="OnSickLeave" eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EBoolean" defaultValueLiteral="false"/>
  </eClassifiers>
</ecore:EPackage>
'''

# Parameter sets
parameter_sets = [
    {"temperature": 0.8, "top_k": 30, "top_p": 0.5, "max_new_tokens": 2048, "repetition_penalty": 1.2},
]

# Results storage
results = []

for file_name in os.listdir(docx_folder):
    if file_name.endswith('.docx'):
        logger.info(f"Processing file: {file_name}")
        docx_path = os.path.join(docx_folder, file_name)
        ground_truth_file = file_name.replace('.docx', '.txt')
        ground_truth_path = os.path.join(ground_truth_folder, ground_truth_file)

        if not os.path.exists(ground_truth_path):
            logger.warning(f"Ground truth file not found for {file_name}. Skipping.")
            continue

        # Read texts
        extracted_text = read_text_from_docx(docx_path)
        ground_truth = read_text_from_txt(ground_truth_path)

        # Check for empty input or ground truth
        if not extracted_text.strip():
            logger.warning(f"Empty extracted text for file: {file_name}. Skipping.")
            continue

        if not ground_truth.strip():
            logger.warning(f"Empty ground truth for file: {file_name}. Skipping.")
            continue

        # Prepare input for the model
        full_input_text = task_description + '\n' + extracted_text
        tokenized_input = tokenizer(full_input_text, return_tensors='pt', padding=True, truncation=True)
        input_ids = tokenized_input.input_ids.to(model.device)
        attention_mask = tokenized_input.attention_mask.to(model.device)

        for param_set in parameter_sets:
            logger.info(f"Running model with parameters: {param_set}")
            with torch.no_grad():
                output_ids = model.generate(input_ids, attention_mask=attention_mask, **param_set)

            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_output = output_text.replace(full_input_text, '').strip() if full_input_text in output_text else output_text.strip()

            # Handle empty model output
            if not generated_output.strip():
                logger.warning(f"Empty output detected for file: {file_name}, parameters: {param_set}")
                results.append({
                    "File": file_name,
                    "Parameters": str(param_set),
                    "Text Cosine Similarity": 0.0,
                    "BERT F1": 0.0,
                    "Generated Output": generated_output
                })
                continue

            # Compute metrics
            text_score = cosine_similarity_text(ground_truth, generated_output)
            P, R, F1 = score([generated_output], [ground_truth], lang="en", verbose=False)

            # Store results
            results.append({
                "File": file_name,
                "Parameters": str(param_set),
                "Text Cosine Similarity": text_score,
                "BERT F1": F1.mean().item(),
                "Generated Output": generated_output
            })


# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Compute statistics for each parameter set
logger.info("Calculating statistics...")
stats = results_df.groupby('Parameters')[['Text Cosine Similarity', 'BERT F1']].agg(['min', 'max', 'mean', 'std']).reset_index()

# Save outputs and statistics
logger.info("Saving results to CSV files.")
results_df.to_csv("few_shot_experiment_results.csv", index=False)
stats.to_csv("few_shot_experiment_statistics.csv", index=False)

logger.info("Experiment completed. Results saved to 'experiment_results.csv' and statistics saved to 'experiment_statistics.csv'.")
