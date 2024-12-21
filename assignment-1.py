import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import gradio as gr

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Function to generate text
def generate_text(prompt, max_length=50, temperature=0.7, top_k=50):
    # Encode the input prompt (convert to token IDs)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Generate text using GPT-2
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=temperature,
        top_k=top_k
    )
    
    # Decode the generated tokens back into text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Load the wikitext dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Print an example from the dataset
print("Example from dataset:", dataset[0])

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], return_tensors="pt", max_length=512, padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenizer.pad_token = tokenizer.eos_token

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,  # Increased number of epochs
    save_steps=10_000,
    save_total_limit=2,
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Set to True for masked language modeling
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# Start fine-tuning
trainer.train()

# Define the Gradio interface
def generate_text_gradio(prompt):
    return generate_text(prompt, max_length=100)

# Create and launch the Gradio interface
interface = gr.Interface(fn=generate_text_gradio, inputs="text", outputs="text", title="GPT-2 Text Generator", description="Enter a prompt to generate text.")
interface.launch()