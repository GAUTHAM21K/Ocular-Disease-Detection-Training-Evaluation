# ==============================================================================
# OCULAR DISEASE DETECTION: TRAINING AND EVALUATION SCRIPT
# ==============================================================================
# This script performs the following steps:
# 1. Downloads the ODIR-5K dataset from Kaggle.
# 2. Filters and preprocesses the data for three diseases:
#    - Diabetic Retinopathy (DR)
#    - Age-related Macular Degeneration (AMD)
#    - Glaucoma (G)
# 3. Splits the data into training, validation, and test sets.
# 4. Fine-tunes a Vision Transformer (ViT) model on the training data.
# 5. Saves the best model locally to './trained_ocular_vit_model'.
# 6. Loads the saved model and evaluates its performance on the test set.
# 7. Saves the confusion matrix and classification report as image/text files.
# ==============================================================================

# --- 0. Imports ---
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
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)
import warnings

# Suppress warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


def setup_kaggle_api():
    """Configures the Kaggle API key for the environment."""
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
    print("✅ Kaggle API configured.")


def download_and_prepare_dataset():
    """Downloads the dataset if not present and prepares the initial DataFrame."""
    print("\n--- 2. Downloading & Preparing Dataset ---")
    local_dataset_path = "./ocular-disease-dataset"
    if not os.path.exists(local_dataset_path):
        import kagglehub
        print("Downloading dataset from Kaggle Hub...")
        download_path = kagglehub.dataset_download("andrewmvd/ocular-disease-recognition-odir5k")
        shutil.copytree(download_path, local_dataset_path)
        print(f"✅ Dataset saved to {local_dataset_path}")
    else:
        print(f"✅ Dataset already exists at {local_dataset_path}")

    data_folder_path = os.path.join(local_dataset_path, "ODIR-5K", "ODIR-5K")
    image_dir = os.path.join(data_folder_path, "Training Images")
    xlsx_path = os.path.join(data_folder_path, "data.xlsx")

    # --- Create Long Format DataFrame (one row per image) ---
    print("Processing annotations...")
    df = pd.read_excel(xlsx_path)
    rows_list = []
    for _, row in df.iterrows():
        img_id = str(row['ID']).strip()
        # Left eye
        left_path = os.path.join(image_dir, f"{img_id}_left.jpg")
        if os.path.exists(left_path):
            left_row = row.to_dict()
            left_row['filepath'] = left_path
            left_row['eye'] = 'left'
            rows_list.append(left_row)
        # Right eye
        right_path = os.path.join(image_dir, f"{img_id}_right.jpg")
        if os.path.exists(right_path):
            right_row = row.to_dict()
            right_row['filepath'] = right_path
            right_row['eye'] = 'right'
            rows_list.append(right_row)
            
    df_long = pd.DataFrame(rows_list)

    # --- Filter for the 3 target diseases ---
    df_filtered = df_long[(df_long['D'] == 1) | (df_long['A'] == 1) | (df_long['G'] == 1)].copy()
    
    # --- Assign Primary Label ---
    disease_columns = ['D', 'A', 'G']
    df_filtered['label'] = df_filtered[disease_columns].idxmax(axis=1)
    label_to_id = {'D': 0, 'A': 1, 'G': 2}
    df_filtered['label_id'] = df_filtered['label'].map(label_to_id)
    
    print(f"✅ Found {len(df_filtered)} images for DR, AMD, and Glaucoma.")
    return df_filtered


def create_hf_dataset(df_filtered):
    """Splits the DataFrame and creates a Hugging Face DatasetDict."""
    print("\n--- 3. Splitting Data and Creating Hugging Face Dataset ---")
    
    # Split: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(
        df_filtered,
        test_size=0.3,
        random_state=42,
        stratify=df_filtered['label_id']
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['label_id']
    )
    print(f"Dataset splits: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")

    # Create Hugging Face Datasets - Keep as file paths, don't cast to HFImage yet
    class_names = ['DR', 'AMD', 'Glaucoma']
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df[['filepath', 'label_id']], preserve_index=False)
                        .rename_column('label_id', 'label'),
        'validation': Dataset.from_pandas(val_df[['filepath', 'label_id']], preserve_index=False)
                             .rename_column('label_id', 'label'),
        'test': Dataset.from_pandas(test_df[['filepath', 'label_id']], preserve_index=False)
                       .rename_column('label_id', 'label')
    })
    dataset = dataset.cast_column('label', ClassLabel(names=class_names))
    print("✅ Hugging Face DatasetDict created.")
    return dataset


def train_model(dataset):
    """Trains the ViT model and saves it locally."""
    print("\n--- 4. Training Model ---")
    model_checkpoint = "google/vit-base-patch16-224"
    model_save_path = "./trained_ocular_vit_model"

    # --- Preprocessing ---
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

    # Transform function - now 'filepath' column contains paths
    def preprocess_function(examples):
        # Load images from file paths
        images = [Image.open(fp).convert("RGB") for fp in examples["filepath"]]
        # Process images
        inputs = image_processor(images, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs

    # Apply preprocessing
    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing train dataset"
    )
    
    val_dataset = dataset["validation"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["validation"].column_names,
        desc="Preprocessing validation dataset"
    )

    # --- Model Loading ---
    labels = dataset["train"].features["label"].names
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # --- Metrics ---
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=model_save_path,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        learning_rate=2e-5,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        push_to_hub=False,
        seed=42,
        remove_unused_columns=False,
    )

    # --- Trainer Initialization and Training ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training complete.")
    
    print(f"💾 Saving best model to {model_save_path}...")
    trainer.save_model(model_save_path)
    image_processor.save_pretrained(model_save_path)
    print("✅ Model saved.")
    return model_save_path


def evaluate_model(model_path, test_dataset):
    """Loads a saved model and evaluates it on the test set."""
    print("\n--- 5. Evaluating Saved Model ---")
    
    # --- Load Model and Processor ---
    model = AutoModelForImageClassification.from_pretrained(model_path)
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    
    # --- Prepare Dataset for Evaluation ---
    def preprocess_function(examples):
        images = [Image.open(fp).convert("RGB") for fp in examples["filepath"]]
        inputs = image_processor(images, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs
    
    processed_test = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Preprocessing test dataset"
    )

    # --- Initialize Trainer for Prediction ---
    eval_args = TrainingArguments(
        output_dir="./eval_temp",
        per_device_eval_batch_size=16,
        remove_unused_columns=False
    )
    trainer = Trainer(model=model, args=eval_args)

    # --- Get Predictions ---
    print("🔮 Generating predictions on the test set...")
    predictions_output = trainer.predict(processed_test)
    pred_labels = np.argmax(predictions_output.predictions, axis=1)
    true_labels = predictions_output.label_ids
    class_names = ["DR", "AMD", "Glaucoma"]

    # --- Confusion Matrix ---
    print("📊 Creating confusion matrix...")
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to {cm_path}")
    plt.close()

    # --- Classification Report ---
    print("\n📋 Generating classification report...")
    report = classification_report(true_labels, pred_labels, target_names=class_names, digits=4)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    print("="*60)

    # Save the report
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("OCULAR DISEASE CLASSIFICATION - TEST SET RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write("\n" + "="*60 + "\n")
    print(f"✅ Classification report saved to {report_path}")
    
    # Calculate and display overall accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\n🎯 Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


def main():
    """Main function to run the entire pipeline."""
    print("\n" + "="*70)
    print("OCULAR DISEASE DETECTION - TRAINING PIPELINE")
    print("="*70 + "\n")
    
    setup_kaggle_api()
    filtered_df = download_and_prepare_dataset()
    hf_dataset = create_hf_dataset(filtered_df)
    saved_model_path = train_model(hf_dataset)
    evaluate_model(saved_model_path, hf_dataset['test'])
    
    print("\n" + "="*70)
    print("🎉 ALL DONE! 🎉")
    print("="*70)
    print(f"\n📦 Trained model saved at: {saved_model_path}")
    print("📊 Confusion matrix saved as: confusion_matrix.png")
    print("📋 Classification report saved as: classification_report.txt")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()