# ==============================================================================
# OCULAR DISEASE DETECTION: SIMPLE TRAINING SCRIPT
# ==============================================================================

import os
import json
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.optim import AdamW
from tqdm import tqdm
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


def setup_kaggle_api():
    """Configures the Kaggle API key."""
    print("--- 1. Setting up Kaggle API ---")
    kaggle_json_content = {
        "username": "username_here",
        "key": "api_key_here"
    }
    kaggle_dir = os.path.expanduser("~/.kaggle")
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(kaggle_json_path, "w") as f:
        json.dump(kaggle_json_content, f)
    os.chmod(kaggle_json_path, 0o600)
    print("✅ Kaggle API configured.\n")


def download_and_prepare_dataset():
    """Downloads and prepares the dataset."""
    print("--- 2. Downloading Dataset ---")
    local_dataset_path = "./ocular-disease-dataset"
    if not os.path.exists(local_dataset_path):
        import kagglehub
        print("Downloading from Kaggle...")
        download_path = kagglehub.dataset_download("andrewmvd/ocular-disease-recognition-odir5k")
        shutil.copytree(download_path, local_dataset_path)
        print(f"✅ Dataset saved to {local_dataset_path}\n")
    else:
        print(f"✅ Dataset already exists\n")

    data_folder_path = os.path.join(local_dataset_path, "ODIR-5K", "ODIR-5K")
    image_dir = os.path.join(data_folder_path, "Training Images")
    xlsx_path = os.path.join(data_folder_path, "data.xlsx")

    print("--- 3. Processing Data ---")
    df = pd.read_excel(xlsx_path)
    rows_list = []
    
    for _, row in df.iterrows():
        img_id = str(row['ID']).strip()
        # Left eye
        left_path = os.path.join(image_dir, f"{img_id}_left.jpg")
        if os.path.exists(left_path):
            left_row = {'filepath': left_path, 'D': row['D'], 'A': row['A'], 'G': row['G']}
            rows_list.append(left_row)
        # Right eye
        right_path = os.path.join(image_dir, f"{img_id}_right.jpg")
        if os.path.exists(right_path):
            right_row = {'filepath': right_path, 'D': row['D'], 'A': row['A'], 'G': row['G']}
            rows_list.append(right_row)
    
    df_long = pd.DataFrame(rows_list)
    
    # Filter for 3 diseases
    df_filtered = df_long[(df_long['D'] == 1) | (df_long['A'] == 1) | (df_long['G'] == 1)].copy()
    
    # Assign labels: D=0, A=1, G=2
    disease_columns = ['D', 'A', 'G']
    df_filtered['label'] = df_filtered[disease_columns].idxmax(axis=1)
    label_to_id = {'D': 0, 'A': 1, 'G': 2}
    df_filtered['label_id'] = df_filtered['label'].map(label_to_id)
    
    print(f"✅ Found {len(df_filtered)} images\n")
    return df_filtered


class OcularDataset(Dataset):
    """Simple PyTorch Dataset for images."""
    def __init__(self, filepaths, labels, image_processor):
        self.filepaths = filepaths
        self.labels = labels
        self.image_processor = image_processor
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        # Load and process image
        image = Image.open(self.filepaths[idx]).convert("RGB")
        inputs = self.image_processor(image, return_tensors="pt")
        
        # Remove batch dimension
        pixel_values = inputs['pixel_values'].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {'pixel_values': pixel_values, 'labels': label}


def train_model(df_filtered):
    """Trains the model using simple PyTorch training loop."""
    print("--- 4. Training Model ---")
    
    # Split data
    train_df, temp_df = train_test_split(df_filtered, test_size=0.3, random_state=42, 
                                          stratify=df_filtered['label_id'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, 
                                        stratify=temp_df['label_id'])
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Setup model and processor
    model_checkpoint = "google/vit-base-patch16-224"
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    
    class_names = ['DR', 'AMD', 'Glaucoma']
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        num_labels=3,
        id2label={0: 'DR', 1: 'AMD', 2: 'Glaucoma'},
        label2id={'DR': 0, 'AMD': 1, 'Glaucoma': 2},
        ignore_mismatched_sizes=True
    )
    
    # Create datasets
    train_dataset = OcularDataset(train_df['filepath'].tolist(), 
                                   train_df['label_id'].tolist(), 
                                   image_processor)
    val_dataset = OcularDataset(val_df['filepath'].tolist(), 
                                 val_df['label_id'].tolist(), 
                                 image_processor)
    test_dataset = OcularDataset(test_df['filepath'].tolist(), 
                                  test_df['label_id'].tolist(), 
                                  image_processor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 5
    
    best_val_acc = 0
    model_save_path = "./trained_ocular_vit_model"
    
    # Training loop
    print("🚀 Starting training...\n")
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]  "):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(pixel_values=pixel_values)
                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(model_save_path)
            image_processor.save_pretrained(model_save_path)
            print(f"✅ Best model saved (Val Acc: {val_acc:.4f})\n")
    
    print("✅ Training complete!\n")
    return model_save_path, test_loader, class_names


def evaluate_model(model_path, test_loader, class_names):
    """Evaluates the model and creates visualizations."""
    print("--- 5. Evaluating Model ---")
    
    # Load model
    model = AutoModelForImageClassification.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Get predictions
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values)
            preds = outputs.logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Confusion Matrix
    print("\n📊 Creating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    print("✅ Saved as confusion_matrix.png")
    plt.close()
    
    # Classification Report
    print("\n📋 Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    
    with open("classification_report.txt", "w") as f:
        f.write("OCULAR DISEASE CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    print("✅ Saved as classification_report.txt")
    
    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


def main():
    """Main pipeline."""
    print("\n" + "="*70)
    print("OCULAR DISEASE DETECTION - TRAINING PIPELINE")
    print("="*70 + "\n")
    
    setup_kaggle_api()
    df_filtered = download_and_prepare_dataset()
    model_path, test_loader, class_names = train_model(df_filtered)
    evaluate_model(model_path, test_loader, class_names)
    
    print("\n" + "="*70)
    print("🎉 ALL DONE! 🎉")
    print("="*70)
    print(f"\n📦 Model: {model_path}")
    print("📊 Confusion matrix: confusion_matrix.png")
    print("📋 Report: classification_report.txt\n")


if __name__ == "__main__":
    main()