import os
from transformers import GenerationConfig
from datasets import load_dataset
import pandas as pd
from peft import PeftModel
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

data_files = {"train": os.getenv('TRAIN_DATA_PATH'),
              "test": os.getenv('TEST_DATA_PATH'),
              "validation": os.getenv('VALIDATION_DATA_PATH')
              }
dataset = load_dataset("json", data_files=data_files, field="data")

baseModelName = os.getenv('BASE_MODEL_NAME')
tokenizer = AutoTokenizer.from_pretrained(baseModelName)
base_model = AutoModelForSeq2SeqLM.from_pretrained(baseModelName)

peft_model = PeftModel.from_pretrained(base_model, os.getenv('FINE_TUNNING_MODEL_NAME'),
                                       is_trainable=False)

questions = dataset['test']['Question']
actual_answers = dataset['test']['Answer']
peft_model_answers = []

for _, question in enumerate(questions):
    prompt = f"""

Assuming you are working as General Knowladge instructor. Can you please answer the below question?

{question}
Answer:"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    peft_model_outputs = peft_model.generate(input_ids=input_ids,
                                             generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
    peft_model_answers.append(peft_model_text_output)

answers = list(zip(questions, actual_answers, peft_model_answers))
df = pd.DataFrame(answers, columns=['question', 'actual answer', 'peft model answer'])

rouge = evaluate.load('rouge')
peft_model_results = rouge.compute(
    predictions=peft_model_answers,
    references=actual_answers,
    use_aggregator=True,
    use_stemmer=True,
)
print(peft_model_results)
