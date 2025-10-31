from setuptools import setup, find_packages

setup(
    name="car_price_prediction",
    version="0.1.0",
    description="Car Price Prediction ML Project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "python-dotenv",
        "pandas", 
        "polars",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "jupyter",
        "shap",
    ],
)
