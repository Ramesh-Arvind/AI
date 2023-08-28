import pandas as pd
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModel
import numpy as np
from torch import nn
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.cuda.amp import autocast, GradScaler
import os
import random
from sklearn.metrics import hamming_loss, jaccard_score, f1_score
import argparse

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--warmup_epoch', default=5, type=int)
parser.add_argument('--max_epoch', default=18, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--eval_batch_size', default=4, type=int)
parser.add_argument('--accumulate_step', default=8, type=int)
args = parser.parse_args()

argus = argparse.Namespace(
        n_candidates=0,   # No retrieval candidates
        n_classes=2,      # Number of classes (citation required or not)
        disable_retrieval=True  # No retrieval, directly classify the claim
    )


# Load a multilingual BERT tokenizer and model
bert_model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BERTModelForClassification(nn.Module):

    def __init__(self, model_name, argus):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, argus.n_classes)
        self.n_classes = argus.n_classes

    def forward(self, input_ids, attention_mask):
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        query_cls_embeddings = hidden_states[:, 0, :]
        logits = self.linear(query_cls_embeddings)
        return logits


model = BERTModelForClassification(bert_model_name, argus).to(device)

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# Define label mapping
label_mapping = {
    "true": 0,
    "mostly true": 0,
    "false": 1,
    "neither": 1,
    "mostly false": 1,
    "partly true/misleading": 1,
    "complicated/hard to categorise": 1,
    "other": 1,
    "half true": 1
}

num_labels = len(label_mapping)

# Load and preprocess Parquet files
train_data = pd.read_parquet("train_xfact.parquet")
validation_data = pd.read_parquet("dev_xfact.parquet")
test_data = pd.read_parquet("test_xfact.parquet")

train_labels = torch.tensor([label_mapping[label] for label in train_data["label"]])
train_labels_array = train_labels.cpu().numpy()
max_sequence_length = 512  # Define the maximum sequence length based on BERT's limit

# Calculate class counts for class imbalance
class_counts = np.bincount(train_labels.numpy())
total_samples = len(train_labels)

class_weights = torch.tensor([total_samples / count for count in class_counts], dtype=torch.float32)
# Adjusted loss function
loss_fn = nn.BCEWithLogitsLoss()

# Training configuration
model.to(device)

# Preprocess and encode data
def preprocess_data(data):
    tokenized_data = []
    attention_masks = []
    labels = []
    for _, row in data.iterrows():
        claim = row["claim"]
        label = row["label"]
        
        claim_tokens = tokenizer.encode(claim, add_special_tokens=True, max_length=max_sequence_length, truncation=True)
        attention_mask = [1] * len(claim_tokens)
        
        # Pad the sequence if it's shorter than the maximum length
        padding_length = max_sequence_length - len(claim_tokens)
        claim_tokens += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        
        tokenized_data.append(claim_tokens)
        attention_masks.append(attention_mask)
        
        label_id = label_mapping.get(label, -1)  # Use -1 for unknown labels
        labels.append(label_id)
        
    # Convert lists to tensors
    tokenized_data = torch.tensor(tokenized_data)
    attention_masks = torch.tensor(attention_masks)
    one_hot_labels = torch.eye(2)[torch.tensor(labels, dtype=torch.long)]
    
    return tokenized_data, attention_masks, one_hot_labels


# Convert lists to tensors for validation and test data preprocessing
train_tokenized_data, train_attention_masks, train_labels = preprocess_data(train_data)
validation_tokenized_data, validation_attention_masks, validation_labels = preprocess_data(validation_data)
test_tokenized_data, test_attention_masks, test_labels = preprocess_data(test_data)

# Pad or truncate tokenized sequences to the same length
train_tokenized_data = train_tokenized_data[:, :max_sequence_length]
validation_tokenized_data = validation_tokenized_data[:, :max_sequence_length]
test_tokenized_data = test_tokenized_data[:, :max_sequence_length]

# Convert tokenized data to PyTorch tensors
train_inputs = torch.tensor(train_tokenized_data)
train_attention_masks = torch.tensor(train_attention_masks)
train_labels = torch.tensor(train_labels)

validation_inputs = torch.tensor(validation_tokenized_data)
validation_attention_masks = torch.tensor(validation_attention_masks)
validation_labels = torch.tensor(validation_labels)

test_inputs = torch.tensor(test_tokenized_data)
test_attention_masks = torch.tensor(test_attention_masks)
test_labels = torch.tensor(test_labels)

# Create separate data loaders for train, validation, and test
train_dataset = TensorDataset(train_inputs, train_attention_masks, train_labels)
validation_dataset = TensorDataset(validation_inputs, validation_attention_masks, validation_labels)
test_dataset = TensorDataset(test_inputs, test_attention_masks, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True)


batch_num = len(train_dataset) // (args.batch_size * args.accumulate_step)
+ (len(train_dataset) % (args.batch_size * args.accumulate_step) != 0)

# Training configuration

param_groups = [{'params': model.parameters(), 'lr': 2e-5}]
optimizer = AdamW(params=param_groups)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=batch_num * args.warmup_epoch,
                                            num_training_steps=batch_num * args.max_epoch)
# Gradient accumulation setup
gradient_accumulation_steps = 4  # Accumulate gradients over 4 batches

# Mixed precision training setup
scaler = GradScaler()

# Early stopping parameters
patience = 3  # Number of epochs without improvement to wait before stopping
best_validation_loss = float("inf")
epochs_without_improvement = 0

# Training loop
train_losses = []
validation_losses = []

# Training loop
for epoch in range(args.max_epoch):
    model.train()
    total_train_predictions = 0
    correct_train_predictions = 0
    accumulated_loss = 0
    small_batch_counter = 0  # Counter to track smaller batches within accumulation steps
    
    for batch_idx, (batch_inputs, batch_attention_masks, batch_labels) in enumerate(train_dataloader):
        batch_inputs = batch_inputs.to(device)
        batch_attention_masks = batch_attention_masks.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(batch_inputs, attention_mask=batch_attention_masks)
            loss = loss_fn(logits, batch_labels)

        scaler.scale(loss).backward()

        accumulated_loss += loss.item()

        small_batch_counter += 1
        if small_batch_counter == gradient_accumulation_steps:
            scaler.step(optimizer)  # Wrap the step in an autocast context
            scaler.update()
            optimizer.zero_grad()
            small_batch_counter = 0  # Reset the counter

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            average_loss = accumulated_loss / gradient_accumulation_steps
            print(f"Epoch {epoch+1}/{args.max_epoch} - Batch {batch_idx+1}/{len(train_dataloader)} - "
                  f"Training Loss: {average_loss:.4f}")
            accumulated_loss = 0

        # Calculate training accuracy
        predicted_labels = (torch.sigmoid(logits) > 0.5).long()
        total_train_predictions += len(batch_labels)
        correct_train_predictions += (predicted_labels == batch_labels).sum().item()

    # Calculate and print training accuracy
    train_accuracy = correct_train_predictions / total_train_predictions
    print(f"Epoch {epoch+1}/{args.max_epoch} - Training Accuracy: {train_accuracy:.4f}")

    # Evaluation on validation data
    model.eval()
    with torch.no_grad():
        total_predictions = 0
        correct_predictions = 0
        total_validation_loss = 0
        for batch_inputs, batch_attention_masks, batch_labels in validation_dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_attention_masks = batch_attention_masks.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_inputs, attention_mask=batch_attention_masks)
            loss = loss_fn(logits, batch_labels)
            total_validation_loss += loss.item()

            predicted_labels = (torch.sigmoid(logits) > 0.5).long()
            total_predictions += len(batch_labels)
            correct_predictions += (predicted_labels == batch_labels).sum().item()

    validation_accuracy = correct_predictions / total_predictions
    average_validation_loss = total_validation_loss / len(validation_dataloader)
    
    # Append losses to lists
    train_losses.append(average_loss)  # Add this line
    validation_losses.append(average_validation_loss) 
    
    print(f"Epoch {epoch+1}/{args.max_epoch} - Validation Accuracy: {validation_accuracy:.4f} - "
          f"Validation Loss: {average_validation_loss:.4f}")
          
    # Check if validation loss has improved
    '''if average_validation_loss < best_validation_loss:
        print(f"Validation loss improved: {best_validation_loss:.4f} -> {average_validation_loss:.4f}")
        best_validation_loss = average_validation_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        print(f"No improvement in validation loss: {best_validation_loss:.4f}")
        epochs_without_improvement += 1

    # Stop training if no improvement for 'patience' epochs
    if epochs_without_improvement >= patience:
        print(f"Early stopping: No improvement for {patience} epochs.")
        break'''
        
    print()
    
# Plotting the loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, args.max_epoch + 1), train_losses, label="Training Loss")
plt.plot(range(1, args.max_epoch + 1), validation_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.show()

# Update learning rate scheduler
scheduler.step(validation_accuracy  * scaler.get_scale())
    
torch.save(model.state_dict(), "best_model_factual.pth")

# Inference
input_claim = "There is a good way of achieving this parameter."
encoded_claim = tokenizer.encode(input_claim, add_special_tokens=True, max_length=max_sequence_length, truncation=True)
attention_mask_claim = [1] * len(encoded_claim)
input_tensor = torch.tensor([encoded_claim]).to(device)
attention_mask_tensor = torch.tensor([attention_mask_claim]).to(device)

with torch.no_grad():
    logits = model(input_tensor, attention_mask=attention_mask_tensor)
    predicted_probs = torch.sigmoid(logits)
    citation_required_prob = predicted_probs[:, 0]  # Probability for the "Citation required" class

if citation_required_prob > 0.5:
    print("Citation required for the claim:", input_claim)
else:
    print("No citation required for the claim:", input_claim)
