# 🧠 Ocular Disease Detection — Training & Evaluation

A simple pipeline to fine-tune a Vision Transformer (ViT) on the ODIR-5K ocular disease dataset for multi-class classification of three eye diseases:  
- Diabetic Retinopathy (DR)  
- Age-related Macular Degeneration (AMD)  
- Glaucoma (G)  

The primary script is `run_training.py`.

---

## 📚 Table of Contents

- [Features](#features)  
- [Results / Artifacts](#results--artifacts)  
- [Prerequisites](#prerequisites)  
- [Recommended Hardware](#recommended-hardware)  
- [Installation](#installation)  
- [Configure Kaggle Credentials (Safe)](#configure-kaggle-credentials-safe)  
- [Prepare Dataset](#prepare-dataset)  
- [Run Training & Evaluation](#run-training--evaluation)  
- [Outputs](#outputs)  
- [Troubleshooting & Tips](#troubleshooting--tips)  
- [Next Steps / Improvements](#next-steps--improvements)  
- [License & Attribution](#license--attribution)

---

## ✅ Features

- Downloads and preprocesses the ODIR-5K dataset (if not present)  
- Filters images for DR, AMD, and Glaucoma  
- Splits data into Train / Validation / Test (70/15/15) with stratification  
- Fine-tunes `google/vit-base-patch16-224` using Hugging Face Transformers  
- Saves best model locally  
- Evaluates on test set and saves confusion matrix and classification report

---

## 📦 Results / Artifacts

After running `run_training.py`, the following are saved:

- `./trained_ocular_vit_model/` — saved model + processor  
- `confusion_matrix.png` — heatmap of test confusion matrix  
- `classification_report.txt` — classification metrics

---

## 🛠️ Prerequisites

- Python 3.8+ (3.10/3.11 recommended)  
- Git (optional)  
- PowerShell (Windows examples used)  
- Kaggle account with access to ODIR-5K dataset

---

## 💻 Recommended Hardware

- GPU with CUDA (NVIDIA) — at least 8 GB VRAM  
- CPU fallback available (slow)

---

## ⚙️ Installation

Create a virtual environment and install dependencies.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

Here’s your text converted into a **clean, professional README.md format** — fully copy-paste-ready for GitHub or any documentation platform:

---

````markdown
# 🧠 Ocular Disease Recognition – Training & Evaluation Guide

This guide explains how to safely configure Kaggle credentials, prepare the dataset, and run the model training and evaluation pipeline.

---

## 🔐 Configure Kaggle Credentials (Safe)

Avoid hardcoding credentials. Use one of the following methods:

### **Option 1: Environment Variables (temporary)**
```powershell
$env:KAGGLE_USERNAME="usernmae_here"
$env:KAGGLE_KEY="api_key_here"
````

### **Option 2: Kaggle CLI (recommended)**

1. Place your `kaggle.json` file in:

   ```
   %USERPROFILE%\.kaggle\kaggle.json
   ```
2. Set file permissions to **user-only**.
3. Install the Kaggle CLI:

   ```bash
   pip install kaggle
   ```

---

## 📁 Prepare Dataset

The script downloads the dataset:

```
andrewmvd/ocular-disease-recognition-odir5k
```

using Kaggle CLI or the Python API.

### **Expected Directory Layout**

```
./ocular-disease-dataset/ODIR-5K/ODIR-5K/Training Images/
./ocular-disease-dataset/ODIR-5K/ODIR-5K/data.xlsx
```

Update the following in your `run_training.py` if needed:

* `local_dataset_path`
* `data_folder_path`

---

## 🚀 Run Training & Evaluation

From the repository root, run:

```powershell
python run_training.py
```

The script will:

1. Set up Kaggle credentials
2. Download and preprocess the dataset
3. Train the model for **5 epochs** (configurable)
4. Save model outputs and evaluation reports

---

## 📂 Outputs

| File / Folder                 | Description                    |
| ----------------------------- | ------------------------------ |
| `./trained_ocular_vit_model/` | Saved trained model            |
| `confusion_matrix.png`        | Confusion matrix visualization |
| `classification_report.txt`   | Detailed evaluation metrics    |

---

## ⚙️ Modify Hyperparameters

To adjust hyperparameters, edit the `TrainingArguments` block in `run_training.py`.

---

## 🧩 Troubleshooting & Tips

### **GPU Out of Memory**

* Reduce `per_device_train_batch_size`
* Use `gradient_accumulation_steps` or set `fp16=True`

### **Missing Packages**

* Activate your virtual environment
* Re-run:

  ```bash
  pip install -r requirements.txt
  ```

### **Kaggle Download Fails**

* Verify your Kaggle credentials
* Accept dataset rules on Kaggle
* Prefer the official `kaggle` package/CLI

### **Dataset Path Errors**

* Confirm that `Training Images` and `data.xlsx` exist in correct paths
* Match the script’s `data_folder_path`

### **Slow CPU Training**

* Use a GPU with proper CUDA setup
* Try [🤗 Hugging Face Accelerate](https://huggingface.co/docs/accelerate) for multi-GPU support

---

## ✅ Summary

| Step | Description                                        |
| ---- | -------------------------------------------------- |
| 1    | Configure Kaggle credentials safely                |
| 2    | Prepare and verify dataset paths                   |
| 3    | Run `python run_training.py`                       |
| 4    | View results in `trained_ocular_vit_model/` folder |

---

**Author:** *Your Name*
**Dataset:** [ODIR-5K (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
**Frameworks Used:** PyTorch, Hugging Face Transformers, Scikit-Learn

```


