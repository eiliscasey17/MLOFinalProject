# Import necessary libraries
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import random
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data  # List of dictionaries with 'input_ids' and 'labels'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {'input_ids': item['input_ids'], 'labels': item['labels']}

# Load tokenizer 
teacher_model_name = 'meta-llama/Llama-2-7b-chat-hf'  # Teacher model (aligned model)
student_model_name = 'meta-llama/Llama-2-7b-hf'       # Student model (base model)

# Ensure you have access to Llama 2 models and have accepted the license on Hugging Face

# Load tokenizer from teacher model
tokenizer = LlamaTokenizer.from_pretrained(teacher_model_name)

# If pad_token is not set, set it to '<pad>'
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

tokenizer_vocab_size = len(tokenizer)

# Load teacher model
teacher_model = LlamaForCausalLM.from_pretrained(
    teacher_model_name,
    device_map='auto',
    torch_dtype=torch.float16
)
teacher_model.resize_token_embeddings(tokenizer_vocab_size)
teacher_model.eval()

# Load student model
student_model = LlamaForCausalLM.from_pretrained(
    student_model_name,
    torch_dtype=torch.float16
)
student_model.resize_token_embeddings(tokenizer_vocab_size)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
student_model.to(device)

# Freeze student model parameters
for param in student_model.parameters():
    param.requires_grad = False

# Apply LoRA to student model for parameter-efficient fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']  # LLaMA model's attention projections
)
student_model = get_peft_model(student_model, lora_config)

# Only train LoRA parameters
for name, param in student_model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True

# Define optimizer
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, student_model.parameters()), lr=1e-4)

# Load and preprocess dataset
dataset = load_dataset('tatsu-lab/alpaca')  # Use Alpaca dataset

# Use the first 10000 samples
max_samples = 2000
data = dataset['train'].select(range(max_samples))

# Preprocess data
def preprocess_data(tokenizer, data, max_length=512):
    processed_data = []
    for sample in data:
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output_text = sample.get('output', '')
        # Construct dialogue text
        if input_text.strip() != '':
            text = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output_text}"
        else:
            text = f"Instruction: {instruction}\nResponse: {output_text}"
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        labels = input_ids.clone()
        # Set padding tokens to -100 to ignore in loss calculation
        labels[encoding['attention_mask'].squeeze(0) == 0] = -100
        processed_data.append({'input_ids': input_ids, 'labels': labels})
    return processed_data

# Preprocess the data
data_processed = preprocess_data(tokenizer, data)

# Shuffle the data
random.shuffle(data_processed)

# Split into training and validation sets
split_ratio = 0.9
split_index = int(len(data_processed) * split_ratio)
train_data = data_processed[:split_index]
validation_data = data_processed[split_index:]

# Prepare data loaders
batch_size = 8
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)
train_dataset = CustomDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

validation_dataset = CustomDataset(validation_data)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

# Define evaluation function with metrics
def evaluate(model, data_loader, device='cuda'):
    model.eval()
    total_loss = 0
    total_perplexity = 0
    total_accuracy = 0
    total_batches = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Calculate perplexity
            perplexity = torch.exp(loss)
            total_perplexity += perplexity.item()
            
            # Calculate accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            # Consider only non-padding tokens
            mask = labels != -100
            correct = (predictions == labels) & mask
            accuracy = correct.sum().item() / mask.sum().item()
            total_accuracy += accuracy
            total_batches += 1
    
    average_loss = total_loss / total_batches
    average_perplexity = total_perplexity / total_batches
    average_accuracy = (total_accuracy / total_batches) * 100  # Convert to percentage
    print(f'Validation - Loss: {average_loss:.4f}, Perplexity: {average_perplexity:.4f}, Accuracy: {average_accuracy:.2f}%')
    return average_loss, average_perplexity, average_accuracy

# Training function with dynamic data selection
def train_with_mode(student_model, train_data, validation_loader, use_data_selection, kl_threshold=1.0, kl_decrease_rate=1.02, min_kl_threshold=0.1, num_epochs=10):
    train_losses = []
    validation_losses = []
    validation_perplexities = []
    validation_accuracies = []
    kl_thresholds = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 3

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs} ({ "Data Selection" if use_data_selection else "No Data Selection" })')
        student_model.train()
        total_train_loss = 0
        total_steps = 0

        if use_data_selection:
            # Dynamic data selection based on KL divergence
            selected_train_data = []
            student_model.eval()  # Switch to evaluation mode for data selection
            with torch.no_grad():
                for sample in train_data:
                    input_ids = sample['input_ids'].unsqueeze(0).to(device)
                    labels = sample['labels'].unsqueeze(0).to(device)

                    # Get logits from teacher model
                    teacher_outputs = teacher_model(input_ids=input_ids)
                    teacher_logits = teacher_outputs.logits

                    # Get logits from student model
                    student_outputs = student_model(input_ids=input_ids)
                    student_logits = student_outputs.logits

                    # Compute KL divergence
                    kl_div = torch.nn.functional.kl_div(
                        input=torch.nn.functional.log_softmax(student_logits.squeeze(0), dim=-1),
                        target=torch.nn.functional.softmax(teacher_logits.squeeze(0), dim=-1),
                        reduction='batchmean'
                    )
                    if kl_div.item() > kl_threshold:
                        selected_train_data.append(sample)

            # If no data is selected, use the entire training data
            if not selected_train_data:
                print(f"No data selected with KL threshold {kl_threshold:.4f}. Using all training data.")
                selected_train_data = train_data.copy()
            else:
                print(f"Selected {len(selected_train_data)} out of {len(train_data)} samples with KL threshold {kl_threshold:.4f}.")
            
            # Update KL threshold
            kl_threshold = max(kl_threshold * kl_decrease_rate, min_kl_threshold)
            kl_thresholds.append(kl_threshold)
        else:
            # Use all training data without selection
            selected_train_data = train_data.copy()

        # Prepare DataLoader for selected data
        selected_train_dataset = CustomDataset(selected_train_data)
        selected_train_loader = DataLoader(selected_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

        # Training loop
        for step, batch in enumerate(selected_train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = student_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            total_steps += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0:
                print(f'  Step {step + 1}/{len(selected_train_loader)} - Loss: {loss.item():.4f}')

        average_train_loss = total_train_loss / total_steps
        print(f'  Average Training Loss: {average_train_loss:.4f}')
        train_losses.append(average_train_loss)

        # Evaluate on validation set
        val_loss, val_perplexity, val_accuracy = evaluate(student_model, validation_loader, device=device)
        validation_losses.append(val_loss)
        validation_perplexities.append(val_perplexity)
        validation_accuracies.append(val_accuracy)

        # Early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    return train_losses, validation_losses, validation_perplexities, validation_accuracies, kl_thresholds

# 1. Train with data selection
print("\n### Training with Data Selection ###")
train_losses_with_selection, val_losses_with_selection, perplexities_with_selection, accuracies_with_selection, kl_thresholds_with_selection = train_with_mode(
    student_model=student_model,
    train_data=train_data,
    validation_loader=validation_loader,
    use_data_selection=True
)

# 2. Train without data selection
print("\n### Training without Data Selection ###")
train_losses_no_selection, val_losses_no_selection, perplexities_no_selection, accuracies_no_selection, _ = train_with_mode(
    student_model=student_model,
    train_data=train_data,
    validation_loader=validation_loader,
    use_data_selection=False
)

# Plot training and validation loss comparison
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses_with_selection) + 1), train_losses_with_selection, label='Train Loss (Data Selection)')
plt.plot(range(1, len(val_losses_with_selection) + 1), val_losses_with_selection, label='Validation Loss (Data Selection)')
plt.plot(range(1, len(train_losses_no_selection) + 1), train_losses_no_selection, label='Train Loss (No Data Selection)', linestyle='--')
plt.plot(range(1, len(val_losses_no_selection) + 1), val_losses_no_selection, label='Validation Loss (No Data Selection)', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Comparison')
plt.legend()
plt.savefig('loss_comparison.png')
plt.show()


# Plot perplexity comparison
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(perplexities_with_selection) + 1), perplexities_with_selection, label='Perplexity (Data Selection)')
plt.plot(range(1, len(perplexities_no_selection) + 1), perplexities_no_selection, label='Perplexity (No Data Selection)', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Perplexity Comparison')
plt.legend()
plt.savefig('perplexity_comparison.png')
plt.show()

# Plot accuracy comparison
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(accuracies_with_selection) + 1), accuracies_with_selection, label='Accuracy (Data Selection)')
plt.plot(range(1, len(accuracies_no_selection) + 1), accuracies_no_selection, label='Accuracy (No Data Selection)', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison')
plt.legend()
plt.savefig('accuracy_comparison.png')
plt.show()
