import transformers
import torch
from bert_score import score as bert_score
from deap import creator, base, tools, algorithms
import numpy as np
import pandas as pd
import random
import copy
import time

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

class NSGA2Optimizer:
    def __init__(self, options={}, others={}):
        # GA parameters
        self.numGen = int(options["numgen"])
        self.mutProb = float(options["mut_prob"])
        self.crossProb = float(options["cross_prob"])
        self.muSel = int(options["mu_sel"])
        self.lambdaSel = int(options["lambda_sel"])
        self.innerMutProb = float(options["inner_mut_prob"])
        self.populationSize = int(options["population_size"])
        self.weights = options["weights"]
        self.prompt = options.get("prompt", "")
        self.model_id = options["model_id"]
        self.device = options.get("device", "cuda") 

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16, "do_sample": True},
            device_map="auto" if self.device == "cuda" else None,
        )

        self.ground_truth = read_text_from_file(options["ground_truth_path"])
        extracted_text = read_text_from_file(options["extracted_text_path"])
        self.full_input_text = options["task_description"] + "\n" + extracted_text

    def createElem(self):
        param_ranges_dict = {
            "temperature": random.uniform(0.5, 1.3),
            "top_k": random.randint(0, 100),
            "top_p": random.uniform(0.5, 1.0),
            "repetition_penalty": random.uniform(1.0, 1.2),
            "max_new_tokens": random.choice([512, 1024, 2048, 3072, 4096]),
        }
        return param_ranges_dict

    def randomInit(self, icls):
        ind = self.createElem()
        return icls(ind)

    def mutUniform(self, individual):
        ind2 = copy.deepcopy(individual)
        for key in individual.keys():
            if random.random() < self.innerMutProb:
                # Mutate each parameter appropriately
                if key == "temperature":
                    ind2[key] = random.uniform(0.7, 1.3)
                elif key == "top_k":
                    ind2[key] = random.randint(0, 100)
                elif key == "top_p":
                    ind2[key] = random.uniform(0.9, 1.0)
                elif key == "repetition_penalty":
                    ind2[key] = random.uniform(1.0, 1.2)
                elif key == "max_new_tokens":
                    ind2[key] = random.choice([512, 1024, 2048, 3072, 4096])
        return (ind2,)

    def crossOverDict(self, ind1, ind2):
        ind1_copy = copy.deepcopy(ind1)
        ind2_copy = copy.deepcopy(ind2)
        for key in ind1.keys():
            if random.random() < 0.5:
                ind1_copy[key], ind2_copy[key] = ind2_copy[key], ind1_copy[key]
        return ind1_copy, ind2_copy

    def generate_with_params(self, params):
        try:
            response = self.pipeline(
                self.full_input_text,
                max_new_tokens=params.get("max_new_tokens", 512),
                temperature=params.get("temperature", 1.0),
                top_k=params.get("top_k", None),
                top_p=params.get("top_p", None),
                repetition_penalty=params.get("repetition_penalty", None),
                return_full_text=False,
            )[0]["generated_text"]

            generated_output = response.strip()
            return generated_output
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory error occurred.")
            torch.cuda.empty_cache()
            return "Error: CUDA out of memory"
        except Exception as e:
            print(f"Error during generation: {e}")
            return f"Error: {e}"

    def evalFitness(self, individual):
        print("Evaluating Fitness")
        configuration = {
            "temperature": individual["temperature"],
            "top_k": int(individual["top_k"]) if individual["top_k"] > 0 else None,
            "top_p": individual["top_p"],
            "repetition_penalty": individual["repetition_penalty"],
            "max_new_tokens": int(individual["max_new_tokens"]),
        }

        start_time = time.time()
        generated_output = self.generate_with_params(configuration)
        inference_time = time.time() - start_time

        if generated_output.startswith("Error:"):
            return (-1e6, 1e6)

        P, R, F1 = bert_score([generated_output], [self.ground_truth], lang="en", verbose=False)
        bert_f1 = F1.mean().item()

        return (bert_f1, inference_time)

    def optimize(self):
        print("Optimizing...")
        creator.create("FitnessMulti", base.Fitness, weights=self.weights)
        creator.create("Individual", dict, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        toolbox.register("individual", self.randomInit, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evalFitness)
        toolbox.register("mate", self.crossOverDict)
        toolbox.register("mutate", self.mutUniform)
        toolbox.register("select", tools.selNSGA2)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        population = toolbox.population(n=self.populationSize)

        algorithms.eaMuPlusLambda(
            population,
            toolbox,
            mu=self.muSel,
            lambda_=self.lambdaSel,
            cxpb=self.crossProb,
            mutpb=self.mutProb,
            ngen=self.numGen,
            stats=stats,
            verbose=True,
        )

        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)
        print('Pareto front:', pareto_front[0])

        return population, pareto_front[0]

if __name__ == "__main__":
    options = {
    "numgen": 10,
    "mut_prob": 0.2,
    "cross_prob": 0.9,
    "mu_sel": 10,
    "lambda_sel": 20,
    "inner_mut_prob": 0.2,
    "population_size": 30,
    "weights": (1.0, -1.0),  # Maximize BERT F1, minimize inference time
    "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "ground_truth_path": "./UPD_Ground_truth.txt",
    "extracted_text_path": "./extracted_text.txt",
    "task_description": '''You are an ER diagram expert. You are tasked with analyzing a text that describes database entities and their relationships. Your objectives are to:
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
}

    optimizer = NSGA2Optimizer(options)
    population, pareto_front = optimizer.optimize()

    df_results = pd.DataFrame([
        {
            "Parameters": ind,
            "Fitness": ind.fitness.values,
        } for ind in pareto_front
    ])
    df_results.to_csv("optimization_results.csv", index=False)