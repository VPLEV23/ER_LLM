import transformers
import torch
from deap import creator, base, tools, algorithms
import numpy as np
import pandas as pd
import random
import copy
import time
import os
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

class HyperparameterOptimizer:
    def __init__(self, options={}):
        # Genetic algorithm parameters
        self.numGen = int(options["numgen"])
        self.mutProb = float(options["mut_prob"])
        self.crossProb = float(options["cross_prob"])
        self.muSel = int(options["mu_sel"])
        self.lambdaSel = int(options["lambda_sel"])
        self.innerMutProb = float(options["inner_mut_prob"])
        self.populationSize = int(options["population_size"])
        self.weights = options["weights"]
        
        # Model and data parameters
        self.model_id = options["model_id"]
        self.device = options.get("device", "cuda")
        self.task_description = options["task_description"]
        self.extracted_text_path = options["extracted_text_path"]
        self.ground_truth_path = options["ground_truth_path"]

        # Load data
        self.extracted_text = self.read_docx(self.extracted_text_path)
        self.ground_truth = self.read_text_file(self.ground_truth_path)
        self.full_input_text = f"{self.task_description}\n{self.extracted_text}"

        # Initialize model
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16, "do_sample": True},
            device_map="auto" if self.device == "cuda" else None,
        )

        # Initialize similarity models
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit([self.ground_truth, self.extracted_text])

        # Parameter search space
        self.param_ranges = {
            "temperature": np.arange(0.5, 2.1, 0.1).round(1).tolist(),  # 0.5 to 2.0, step 0.1
            "top_k": list(range(0, 101, 10)),  # 0 to 100, step 10
            "top_p": np.arange(0.5, 1.05, 0.1).round(1).tolist(),  # 0.5 to 1.0, step 0.1
            "repetition_penalty": np.arange(1.0, 2.1, 0.1).round(1).tolist(),  # 1.0 to 2.0, step 0.1
            "max_new_tokens": [512, 1024, 2048, 3072, 4096, 8192],
        }


    def read_docx(self, file_path):
        """Read text from .docx file"""
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])

    def read_text_file(self, file_path):
        """Read text from plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def create_individual(self):
        """Create random individual"""
        return {
            "temperature": random.choice(self.param_ranges["temperature"]),
            "top_k": random.choice(self.param_ranges["top_k"]),
            "top_p": random.choice(self.param_ranges["top_p"]),
            "repetition_penalty": random.choice(self.param_ranges["repetition_penalty"]),
            "max_new_tokens": random.choice(self.param_ranges["max_new_tokens"]),
        }

    def mutate(self, individual):
        """Mutation operator"""
        ind2 = copy.deepcopy(individual)
        for key in individual.keys():
            if random.random() < self.innerMutProb:
                ind2[key] = random.choice(self.param_ranges[key])
        return (ind2,)

    def crossover(self, ind1, ind2):
        """Crossover operator"""
        ind1_copy, ind2_copy = copy.deepcopy(ind1), copy.deepcopy(ind2)
        for key in ind1.keys():
            if random.random() < 0.5:
                ind1_copy[key], ind2_copy[key] = ind2_copy[key], ind1_copy[key]
        return ind1_copy, ind2_copy

    def generate_text(self, params):
        """Generate text with given parameters"""
        try:
            response = self.pipeline(
                self.full_input_text,
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_k=params["top_k"],
                top_p=params["top_p"],
                repetition_penalty=params["repetition_penalty"],
                return_full_text=False,
            )[0]["generated_text"]
            return response.strip()
        except Exception as e:
            print(f"Generation error: {e}")
            return ""

    def evaluate(self, individual):
        """Evaluate parameters on training data"""
        config = {
            "temperature": individual["temperature"],
            "top_k": int(individual["top_k"]),
            "top_p": individual["top_p"],
            "repetition_penalty": individual["repetition_penalty"],
            "max_new_tokens": int(individual["max_new_tokens"]),
        }

        similarities = []
        for _ in range(3):  # 3 trials for stability
            # Generate output
            start_time = time.time()
            generated = self.generate_text(config)
            inference_time = time.time() - start_time

            # Calculate similarities
            tfidf_sim = cosine_similarity(
                self.vectorizer.transform([self.ground_truth]),
                self.vectorizer.transform([generated])
            )[0][0]

            bert_sim = torch.nn.functional.cosine_similarity(
                self.bert_model.encode(self.ground_truth, convert_to_tensor=True),
                self.bert_model.encode(generated, convert_to_tensor=True),
                dim=0
            ).item()

            similarities.append((tfidf_sim, bert_sim, inference_time))

        # Average results
        return tuple(np.mean(similarities, axis=0))

    def optimize(self):
        """Run NSGA-II optimization"""
        creator.create("FitnessMulti", base.Fitness, weights=self.weights)
        creator.create("Individual", dict, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, self.create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", self.crossover)
        toolbox.register("mutate", self.mutate)
        toolbox.register("select", tools.selNSGA2)

        population = toolbox.population(n=self.populationSize)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        population, logbook = algorithms.eaMuPlusLambda(
            population, toolbox,
            mu=self.muSel,
            lambda_=self.lambdaSel,
            cxpb=self.crossProb,
            mutpb=self.mutProb,
            ngen=self.numGen,
            stats=stats,
            verbose=True
        )

        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        
        # Save the best parameters
        best_params = []
        for i, params in enumerate(pareto_front):
            best_params.append({
                "params_id": i+1,
                "temperature": params["temperature"],
                "top_k": params["top_k"],
                "top_p": params["top_p"],
                "repetition_penalty": params["repetition_penalty"],
                "max_new_tokens": params["max_new_tokens"],
                "cosine_similarity": params.fitness.values[0],
                "bert_similarity": params.fitness.values[1],
                "inference_time": params.fitness.values[2]
            })
        
        pd.DataFrame(best_params).to_csv("optimized_parameters.csv", index=False)
        print("Optimization complete. Best parameters saved to optimized_parameters.csv")
        return best_params

if __name__ == "__main__":
    # Configuration
    config = {
        "numgen": 10,
        "mut_prob": 0.2,
        "cross_prob": 0.3,
        "mu_sel": 20,
        "lambda_sel": 60,
        "inner_mut_prob": 0.15,
        "population_size": 30,
        "weights": (1.0, 1.0, -1.0),  # Maximize similarities, minimize time
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "task_description": '''You are an ER diagram expert. You are tasked with analyzing a text that describes database entities and their relationships. Your objectives are to:
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
    6. Generate the output as EMF-compatible code in XMI or Ecore format, ensuring it’s suitable for importing into an EMF model. The output should solely be in EMF-compatible syntax.
    Output should be only the EMF-compatible code for the entities, attributes, and relationships identified in the text. Once you start writing code, do not write any additional text interrupting your code.

    Example:
    ```XML <?xml version="1.0" encoding="UTF-8"?>
<ecore:EPackage xmi:version="2.0" xmlns:xmi="http://www.omg.org/XMI" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:ecore="http://www.eclipse.org/emf/2002/Ecore" name="UPDGround" nsURI="http://university.edu/UPDGround" nsPrefix="upd">
  <eClassifiers xsi:type="ecore:EClass" name="CriteriDiInclusione">
    <eStructuralFeatures xsi:type="ecore:EAttribute" name="record_id" eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EString"
        derived="true"/>
    <eStructuralFeatures xsi:type="ecore:EAttribute" name="diabete_mellito" eType="ecore:EDataType http://www.eclipse.org/emf/2002/Ecore#//EString"/>
  </eClassifiers>```''',  # Your full task description
        "extracted_text_path": "./instructions/03_LabTracker.docx",  # Path to your document
        "ground_truth_path": "../groundtruth/03_LabTracker.txt"  # Path to ground truth
    }

    # Run optimization
    optimizer = HyperparameterOptimizer(config)
    best_params = optimizer.optimize()