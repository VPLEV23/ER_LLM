
# Project Structure

```
/data
  ├── /raw
       ├── extracted_text.txt         # Input text for model to process and generate outputs
       ├── LIFEMap_eCRF_v2.0_25.06.2024.pdf  # External document used in project
       └── UPD_Ground_truth.txt       # Ground truth text for evaluating model outputs

/outputs
  ├── /er_diagrams
       └── exp-64.png                 # Generated ER diagram based on text inputs
  ├── evaluation_results_gridsearch.csv   # Evaluation results from grid search optimization
  └── optimization_results_nsga.csv   # NSGA-II optimization results and evaluations

/src
  ├── /models
       ├── simple_model.py            # A basic model for generating ER diagram
  ├── /optimizers
       ├── gridsearch.py              # Grid search optimizer for hyperparameter tuning
       └── NSGA2Optimizer.py          # NSGA-II optimizer for multi-objective optimization
```

### Directory Details

- **/data/raw**: Contains raw input data files used by models. This includes extracted text, inital pdf documents, and ground truth data.
  
- **/outputs/er_diagrams**: Contains images representing Entity-Relationship (ER) diagrams generated by models.

- **/outputs**: Other output files, including results from different optimization runs, such as NSGA-II and grid search results.

- **/src/models**: Contains model scripts used for text generation, diagram generation.

- **/src/optimizers**: Includes optimization algorithms, such as grid search and NSGA-II, used for tuning model parameters and improving performance.
