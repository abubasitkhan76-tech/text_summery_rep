text_summerization

The purpose of this repository is to serve as a base template for deep learning projects in python.

### Pre-Requirements

1. Python 3.10.6
2. uv

#### Setup

1. Create and activate virtual Environment

```
    Create: uv venv
    Activate on Windows: .venv\Scripts\activate
    Activate on Linux/Macos: source .venv/bin/activate
```

2. Install dependencies
   ` uv sync`

steby step flow
text_summery_rep/
├── .venv/                # Your virtual environment
├── artifacts/            # Stores trained model weights and logs
├── data/                 # Raw and processed CSVs
├── notebooks/            # Experimental Jupyternotebooks (.ipynb)
├── src/                  # Core logic (The heart of the project)
│   ├── components/       # DataIngestion, DataTransformation, ModelTrainer
│   ├── pipelines/        # Training and Prediction pipelines
│   ├── constants/        # Fixed paths and Hyperparameters
│   ├── entity/           # Configuration classes
│   └── utils/            # Helper functions (save/load models)
├── app/                  # FastAPI implementation
├── main.py               # Entry point to trigger training
├── requirements.txt      # List of dependencies
└── setup.py              # To install 'src' as a local package


# BART Text Summarizer
A deep learning project to summarize news articles using the BART-base model, trained on the CNN/DailyMail dataset.

### 🏆 Model Performance (ROUGE Scores)
* **ROUGE-1:** 0.4070
* **ROUGE-2:** 0.1890
* **ROUGE-L:** 0.3798

### 📂 Project Structure
* `src/dataaaa/`: Data loading and cleaning logic.
* `src/model/`: BART model definition, training loop, and ROUGE evaluation.
* `trained_bart_model/`: Saved model weights (Download from Google Drive).
