import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import time

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


def prompt_generator(batchData):
    start = 'Assuming you are working as General Knowledge instructor. Can you please answer the below question?\n\n'
    end = '\n Answer: '
    training_prompt = [start + question + end for question in batchData['Question']]
    batchData['input_ids'] = tokenizer(training_prompt, padding="max_length", return_tensors="pt").input_ids
    batchData['labels'] = tokenizer(batchData['Answer'], padding="max_length", return_tensors="pt").input_ids
    return batchData


instructed_datasets = dataset.map(prompt_generator, batched=True)
instructed_datasets = instructed_datasets.remove_columns(['id', 'Question', 'Answer'])
print(instructed_datasets)

lora_config = LoraConfig(
    r=32,  # Rank
    lora_alpha=32,  # LoRA scaling factor
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM  # FLAN-T5
)
peft_model = get_peft_model(base_model,
                            lora_config)

output_dir = f"{os.getenv('MODEL_OUTPUT_DIR')}flan-output-{str(int(time.time()))}"

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=1e-3,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=1,
    max_steps=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=instructed_datasets['train'],
    eval_dataset=instructed_datasets['validation']
)

trainer.train()

saved_dir = f"{os.getenv('MODEL_SAVE_DIR')}flan-trained-{str(int(time.time()))}"
tokenizer.save_pretrained(saved_dir)
peft_model.save_pretrained(saved_dir)
