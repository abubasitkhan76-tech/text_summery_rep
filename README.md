# BART Text Summarizer & Deep Learning Template

This repository serves as a base template for deep learning projects in Python. It includes a text summarization system using the **BART** model, trained on the **CNN/DailyMail** dataset.

---

## 📋 Prerequisites
* **Python**: `3.10.6`
* **Package Manager**: `uv`

---

## ⚙️ Setup & Installation

### 1. Create and Activate Virtual Environment
```bash
# Create environment
uv venv

# Activate on Windows
.venv\Scripts\activate

# Activate on Linux/macOS
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
uv sync
```

---

## 🏆 Model Performance
The fine-tuned BART model achieves the following **ROUGE** scores:

* **ROUGE-1**: `0.4070`
* **ROUGE-2**: `0.1890`
* **ROUGE-L**: `0.3798`

---

## 📂 Project Structure

```text
├── app/
│   ├── .keep
│   └── streamlit.py        # Streamlit web application interface
├── data/
│   └── dawnload.py         # Script to download datasets
├── notebook/
│   └── eda.ipynb           # Exploratory Data Analysis notebook
├── src/
│   ├── dataaaa/            # Data loading and cleaning logic
│   ├── model/              # BART model definition and training loop
│   └── pipelines/          # Training and prediction workflows
├── .gitignore              # Files to ignore in Git
├── __init__.py            # Makes src/ a Python package
├── README.md               # Project documentation
├── main.py                 # Main entry point to trigger training
├── requirement.txt         # List of project dependencies
└── utils.py                # Helper functions (save/load models)
```

> 💡 **Note on Model Weights:** The raw model weights are saved inside `trained_bart_model/`. Download them from Google Drive before running the application.
