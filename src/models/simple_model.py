import transformers
import torch
import PyPDF2
import os
from datetime import datetime

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


txt_path = "./extracted_text.txt"
extracted_text = read_text_from_file(txt_path)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16, "temperature": 1.1, "do_sample": True}, 
    device_map="auto"
)


task_description = '''
You are an ER diagram expert. You are tasked with analyzing a text that describes database entities and their relationships. Your objectives are to:
1. Identify all entities (tables) mentioned in the text.
2. Extract attributes (columns) for each entity and determine their data types if mentioned.
3. Identify primary keys and foreign keys to understand the relationships between entities.
4. Identify relationships between entities (one-to-one, one-to-many, many-to-many). Using Entity01 }|..|| Entity02 Entity03 }o..o| Entity04 Entity05 ||--o{ Entity06 Entity07 |o--|| Entity08 this notation.
5. If some attributes not in text dont add them to the diagram.

5. Generate a PlantUML script to represent these entities and relationships in an ER diagram. The output should be solely the PlantUML code.
## Complete Example

<plantuml>
@startuml

' hide the spot
' hide circle

' avoid problems with angled crows feet
skinparam linetype ortho

entity "User" as e01 {
  *user_id : number <<generated>>

}

entity "Card" as e02 {
  *card_id : number <<generated>>

}

entity "CardHistory" as e05 {
  *card_history_id : number <<generated>>

entity "CardsAccounts" as e04 {
  *id : number <<generated>>

}


entity "Account" as e03 {
  *account_id : number <<generated>>


entity "Stream" as e06 {
  *id : number <<generated>>
}




@enduml
</plantuml>
The output should be only the PlantUML code. 
'''
messages = [
    {"role": "system", "content": task_description},
    {"role": "user", "content": extracted_text},
]

outputs = pipeline(
    messages,
    max_new_tokens=4096,
)
print(outputs[0]["generated_text"])