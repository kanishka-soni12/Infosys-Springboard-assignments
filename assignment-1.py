import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import gradio as gr

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

# Function to generate text
def generate_text(prompt, max_length=50, temperature=0.7, top_k=50):
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True)
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=temperature,
        top_k=top_k
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

print("Example from dataset:", dataset[0])

def tokenize_function(examples):
    return tokenizer(examples["text"], return_tensors="pt", max_length=512, padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenizer.pad_token = tokenizer.eos_token

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,  # Increased number of epochs
    save_steps=10_000,
    save_total_limit=2,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Set to True for masked language modeling
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

trainer.train()

def generate_text_gradio(prompt):
    return generate_text(prompt, max_length=100)

# Create and launch the Gradio interface
interface = gr.Interface(fn=generate_text_gradio, inputs="text", outputs="text", title="GPT-2 Text Generator", description="Enter a prompt to generate text.")
interface.launch()
