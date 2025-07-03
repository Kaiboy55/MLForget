# Machine Unlearning Library

This library provides modular implementations of machine unlearning techniques for time-series data. It is designed for use in research, evaluation, and deployment scenarios where data privacy and the right to be forgotten are important.

## 📦 Features

- Support for **Exact Retraining**, **SISA**, and **Knowledge Distillation** as unlearning strategies
- LSTM-based model architecture for building energy consumption prediction
- Modular preprocessing, training, evaluation, and unlearning components
- Support for measuring carbon emissions via CodeCarbon
- Designed as a reusable, extendable Python package

---

## 📁 Project Structure

```
├── data_utils.py         # Data loading, preprocessing, and forget set selection
├── model_utils.py        # LSTM model definition
├── training.py           # Training loops and model evaluation
├── evaluation.py         # Metrics: RMSE, MAE, R²
├── unlearning.py         # Implementations of unlearning methods
├── workflow.py           # High-level run_* functions for each unlearning method
├── __init__.py           # Public API
```

---

## 🚀 Getting Started

### Installation

Clone the repository and make sure dependencies are installed:

```bash
pip install torch pandas scikit-learn
```

---

### Example Usage

```python
from Machine_Unlearning_Tool import (
    load_dataset, preprocess_data, run_sisa_unlearning
)

# Load and preprocess data
df = load_dataset("data/train.csv")
config = { ... }  # preprocessing configuration
df = preprocess_data(df, config)

# Convert to tensors and run SISA
X = ...
y = ...
results = run_sisa_unlearning(X, y, df, input_cols, target_col, id_col, forget_ids, device)
```

---

## 🧠 Public API Overview

### `load_dataset(filepath: str) -> pd.DataFrame`
Loads a dataset from CSV.

### `preprocess_data(df: pd.DataFrame, config: dict) -> pd.DataFrame`
Preprocesses the dataset using normalization and categorical encoding.

### `train_model(model, ...)`
Trains a PyTorch model on provided data.

### `run_sisa_unlearning(...)`
Performs SISA unlearning: slices and shards the dataset, retrains on retained data.

### `run_exact_retraining(...)`
Performs full retraining on retained data.

### `run_knowledge_distillation(...)`
Trains a student model using teacher model outputs while excluding forget set.

### `evaluate_on_loader(model, loader, loss_fn, device)`
Computes RMSE, MAE, and R² metrics on a test or validation loader.

---

## 📊 Output

Each `run_*` method returns:
- Trained model or model dictionary
- Pre- and post-unlearning performance metrics
- (Optional) CodeCarbon energy emission data

---

## 📌 Notes

- Forget set is defined using user-specified `id_column` and list of IDs
- Can be extended with new models or additional unlearning methods
- Suitable for research on GDPR-compliant model behavior

---

## 📃 License

No License specified.
