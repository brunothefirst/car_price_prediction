# Car Price Prediction Project

## Overview

This project focuses on predicting second-hand car prices using machine learning techniques. The goal is to build accurate predictive models that can estimate the market value of used cars based on various features such as make, model, year, mileage, condition, and other relevant characteristics.

The project implements multiple machine learning algorithms including traditional methods like Random Forest and XGBoost, as well as gradient boosting techniques using LightGBM. Model interpretability is enhanced using SHAP (SHapley Additive exPlanations) values to understand feature importance and model predictions.

## Project Structure

```
car_price_prediction/
├── data/                   # Data directory
│   ├── raw/               # Raw, unprocessed data files
│   └── processed/         # Cleaned and processed data files
├── notebooks/             # Jupyter notebooks for exploration and analysis
├── src/                   # Source code directory
│   ├── config.py          # Configuration management and environment variables
│   ├── data/              # Data processing and feature engineering scripts
│   ├── models/            # Model training, prediction, and evaluation scripts
│   └── utils/             # Utility functions and helper modules
├── models/                # Serialized model files (.pkl, .joblib)
├── tests/                 # Unit tests and integration tests
├── .env                   # Environment variables (not tracked in git)
├── .gitignore            # Git ignore file
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd car_price_prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment setup:**
   - Copy the `.env` file and update the paths if needed
   - Ensure data and models directories are properly configured
   - Place your raw data files in the `data/raw/` directory

5. **Verify installation:**
   ```bash
   python -c "import src.config; print('Setup successful!')"
   ```

## Usage

### Data Processing
1. Place your raw car data files in `data/raw/`
2. Use the data processing scripts in `src/data/` to clean and prepare the data
3. Processed data will be saved to `data/processed/`

### Model Training
1. Use the scripts in `src/models/` to train different machine learning models
2. Trained models will be automatically saved to the `models/` directory
3. Model evaluation metrics and reports will be generated

### Exploratory Analysis
1. Open Jupyter notebooks in the `notebooks/` directory
2. Use these notebooks for data exploration, visualization, and experimentation
3. Results and insights can be documented within the notebooks

### Model Inference
1. Load trained models from the `models/` directory
2. Use the prediction scripts to make price predictions on new car data
3. SHAP values can be generated for model interpretability

### Example Usage
```python
from src.config import PROCESSED_DATA_PATH, MODELS_PATH
from src.models.predict import CarPricePredictor

# Load and use a trained model
predictor = CarPricePredictor(model_path=MODELS_PATH / "best_model.pkl")
price_prediction = predictor.predict(car_features)
```
