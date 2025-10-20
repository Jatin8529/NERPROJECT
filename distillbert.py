import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_metric
from transformers import AutoTokenizer,TFDistilBertForTokenClassification,Trainer,TrainingArguments

import tensorflow as tf

# Load the dataset
file_path = 'C:\\Users\\91852\\OneDrive\\Desktop\\training dataset.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Function to preprocess the data
def preprocess_data(df):
    print(df)
    print(df.columns)

    sentences = df['Sentence'].tolist()
    print(sentences)

    if 'Tokens' in df.columns:
        tokens = df['Tokens'].apply(lambda x: x.split()).tolist()
        print(tokens)
    else:
        print("The 'Tokens' column is not present in the DataFrame.")
    
    if 'Labels' in df.columns:
        labels = df['Labels'].apply(lambda x: x.split()).tolist()
        print(labels)
    else:
        print("The 'Labels' column is not present in the DataFrame.")
        

    # Flatten the lists for token and label alignment
    processed_data = {'tokens': [],'labels': []}
    for sentences, sentence_tokens, sentence_labels in zip(sentences, tokens, labels):
        processed_data['tokens'].append(sentence_tokens)
        processed_data['labels'].append(sentence_labels)
    return Dataset.from_dict(processed_data)

# Prepare the dataset
dataset = preprocess_data(data)
print(dataset)

df = pd.read_csv("C:\\Users\\91852\\OneDrive\\Desktop\\training dataset.csv")

dataset_df = dataset.to_pandas(df)  # Convert to DataFrame if using datasets library
print(dataset_df)
train_dataset, eval_dataset = train_test_split(dataset_df, test_size=0.1, random_state=42)



# Load tokenizer and model  
tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")

model = TFDistilBertForTokenClassification.from_pretrained("dslim/distilbert-NER", 97)  # Set num_labels

print(train_dataset.columns)

label_to_id = {
    "O": 0,
    "B-ANIMAL": 1,
    "I-ANIMAL": 2,
    # Add other labels as necessary
    # "B-PERSON": 3,
    # "I-PERSON": 4,
    # etc.
}

unique_labels = set([label for example in dataset for label in example['labels']])
print("Unique Labels in Dataset:", unique_labels)

# Step 2: Ensure 'label_to_id' has all the labels
for label in unique_labels:
    if label not in label_to_id:
        print(f"Missing label: {label}")
        # Add the missing label with a unique ID
        label_to_id[label] = len(label_to_id)

def clean_labels(example):
    example['labels'] = [label_to_id[label] for label in example['labels']]
    return example

# Clean the labels in the dataset
dataset = dataset.map(clean_labels)



# Tokenize the dataset
def tokenize_and_align_labels(examples):
    print(examples)
    tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True)
    labels = examples['labels']
    print(labels)
    print(tokenized_inputs)


    
    # Align labels with tokenized inputs
    labels = []
    for i, label in enumerate(examples['labels']):
        try:
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to words in the original sentence
        except IndexError:
            print(f"IndexError: batch_index={i} is out of range for word_ids.")
            continue

        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special token (like [CLS], [SEP])
            elif word_idx != previous_word_idx:  # New word
                if word_idx < len(label):
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)  # If word_idx exceeds label length
            else:
                label_ids.append(-100)  # Subword token (not first in the word)
            previous_word_idx = word_idx
        labels.append(label_ids)

    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
print(tokenized_dataset)

if isinstance(train_dataset, pd.DataFrame):
    train_dataset = Dataset.from_pandas(train_dataset)

train_dataset = train_dataset.map(tokenize_and_align_labels,batched=True)
print(train_dataset)
exit()
eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)
print(eval_dataset)

# Load metrics
metric = load_metric("seqeval")
print(metric)

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
model.save_pretrained('./fine_tuned_distilbert')
tokenizer.save_pretrained('./fine_tuned_distilbert')
