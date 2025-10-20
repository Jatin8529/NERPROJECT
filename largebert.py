import pandas as pd
import os
from sklearn.model_selection import train_test_splitm
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments

# Load the dataset
file_path = 'C:\\Users\\91852\\OneDrive\\Desktop\\book2.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Define a function to convert your dataset into the expected format
def create_dataset(df):
    # This assumes your dataset has 'Sentence' and 'Labels' columns
    sentences = df['Sentence'].tolist()
    labels = df['Labels'].tolist()
    
    # Convert labels to the appropriate format for the model
    # This should be done according to your specific label set and encoding scheme
    # Example: Convert labels to integers or use a label map

    return Dataset.from_dict({'sentence': sentences, 'labels': labels})

# Create dataset
dataset = create_dataset(data)

# Split the dataset
train_dataset, eval_dataset = train_test_split(dataset, test_size=0.1, random_state=42)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER", num_labels=num_labels)  # Update num_labels

# Tokenize the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['sentence'], padding="max_length", truncation=True, is_split_into_words=True)
    # Align labels with tokens
    labels = [label_map[label] for label in examples['labels']]  # Convert labels to IDs
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)

# Load metrics
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    evaluation_strategy="epoch",     # evaluate after each epoch
    learning_rate=2e-5,               # learning rate
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    num_train_epochs=3,              # number of epochs
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         # the instantiated model
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset,           # evaluation dataset
    compute_metrics=compute_metrics,     # compute metrics function
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./fine_tuned_bert')
tokenizer.save_pretrained('./fine_tuned_bert')
