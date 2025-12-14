# 🍕🚚 Food Delivery Time Prediction Model

## Overview
This project predicts estimated food delivery time in minutes from order and contextual features via an end‑to‑end pipeline including ingestion, validation, transformation, and modeling, plus a ready‑to‑use Streamlit app for inference. The repository contains reproducible notebooks, a modular Python package, persisted artifacts for preprocessor and model, and experiment tracking using MLflow for offline comparison and auditing.

## Key features
- End‑to‑end ML pipeline: data ingestion, schema validation, missing‑value checks, drift detection, feature engineering, model selection, and artifact persistence for reproducibility.
- Robust preprocessing: One‑Hot for nominal categoricals, explicit Ordinal encodings for traffic and time of day, and numeric scaling with appropriate imputation strategies.
- Model selection with hyperparameter search across DecisionTree, RandomForest, AdaBoost, GradientBoosting, XGBoost, and CatBoost, selecting the best by R² and logging to MLflow.
- Interactive Streamlit inference app that loads preprocessor and model from final_model for quick, user‑friendly predictions.

## Project structure
The repository separates configuration, pipeline stages, utilities, notebooks, data, and final artifacts for clarity and maintainability. The following layout can be used as a reference tree in the README and mirrors common patterns for ML projects.
```
Food-Delivery-Time-Prediction-main/
├─ app.py                              # Streamlit app for inference [run]
├─ main.py                             # Orchestrates the full training pipeline
├─ requirements.txt                    # Python dependencies
├─ setup.py                            # Packaging metadata
├─ Data/
│  ├─ RawData/Food_Delivery_Times.csv  # Raw dataset
│  └─ Processed/delivery_data_cleaned.csv
├─ Data_Schema/__init__.py             # SCHEMA and Required_Columns dicts
├─ final_model/
│  ├─ model.pkl                        # Best trained regressor
│  └─ preprocessor.pkl                 # Fitted preprocessing pipeline
├─ FoodDeliveryTimePrediction/
│  ├─ Constants/__init__.py            # All constants and paths
│  ├─ Entity/                          # Config and artifact dataclasses
│  ├─ Exception/exception.py           # Custom exception
│  ├─ Logging/logger.py                # Centralized logging
│  ├─ Src/                             # Pipeline steps
│  │  ├─ Data_Ingestion/data_ingestion.py
│  │  ├─ Data_Validation/data_validation.py
│  │  ├─ Data_Transformation/data_transformation.py
│  │  └─ Model_Trainer/model_trainer.py
│  └─ Utils/main_utils.py              # I/O, search, metrics, wrapper model
├─ NoteBook/                           # Data cleaning, EDA, and training notebooks
└─ mlruns/                             # MLflow runs and model tracking artifacts
```


## Data schema
Required feature columns and types are defined in Data_Schema/__init__.py as Required_Columns with consistent names and expected dtypes. Features include Distance_km (float), Weather (string), Traffic_Level (Low/Medium/High), Time_of_Day (Morning/Afternoon/Evening/Night), Vehicle_Type (Bike/Scooter/Car), and Preparation_Time_min (float/int), with target Delivery_Time_min (float/int).

## How the pipeline works
- Data Ingestion reads the raw CSV, creates a feature‑store copy, and splits train/test using a configured ratio under timestamped Artifacts folders.
- Data Validation enforces schema conformance, applies per‑column missing‑value thresholds, and runs Kolmogorov–Smirnov drift detection, producing a YAML drift report.
- Data Transformation builds a ColumnTransformer that scales numerics, one‑hot encodes Weather and Vehicle_Type, ordinal‑encodes Traffic_Level and Time_of_Day with explicit order, and saves arrays and the fitted preprocessor.
- Model Trainer tunes and evaluates multiple regressors via RandomizedSearchCV, selects the best by R², logs metrics to MLflow, and persists both a TimePredictionModel wrapper and final_model artifacts for the app.

## Installation
It is recommended to use a virtual environment for isolation, then install dependencies from requirements.txt to reproduce versions reliably. Ensure Python and pip are present, then run the following from the repository root.
```bash
pip install -r requirements.txt
```

## Training
The complete training pipeline is orchestrated by main.py, which creates timestamped Artifacts for each stage and records run logs to logs/ and MLflow runs to mlruns/. Execute training from the repository root with the command below.
```bash
python main.py
```

## Outputs
- Artifacts/<timestamp>/data_ingestion/...: feature‑store CSV and train/test splits for traceability.
- Artifacts/<timestamp>/data_validation/...: validated train/test data and drift report YAML for quality checks.
- Artifacts/<timestamp>/data_transformation/...: transformed arrays and preprocessing object for modeling.
- Artifacts/<timestamp>/model_trainer/...: trained wrapper model and metrics for evaluation.
- final_model/model.pkl and final_model/preprocessor.pkl: final assets used by the Streamlit app for inference.

## Inference app (Streamlit)
The Streamlit app in app.py loads final_model/preprocessor.pkl and final_model/model.pkl to predict delivery time given interactive inputs. Inputs cover distance, weather, traffic level, time of day, vehicle type, and preparation time, with results shown after clicking Predict Delivery Time.
```bash
streamlit run app.py
```

## Configuration and constants
Directory names, file names, split ratios, accuracy thresholds, and preprocessing object names are centralized in FoodDeliveryTimePrediction/Constants/__init__.py for consistency. Artifact and step‑level configs are constructed via dataclasses in FoodDeliveryTimePrediction/Entity/config_entity.py and passed to each pipeline stage.

## Experiment tracking
Model Trainer logs metrics like r2_score, mean_squared_error, and root_mean_squared_error to MLflow, organizing runs and versioned models under mlruns/ for offline inspection. This enables fair comparisons across hyperparameter searches and preserves artifacts for future evaluation or deployment.

## Notebooks
Three notebooks illustrate data cleaning, EDA, and trainer experimentation: 1‑DataCleaning.ipynb, 2‑EDA.ipynb, and 3‑ModelTrainer.ipynb. They demonstrate imputations, encodings, visualizations, and model selection insights, e.g., CatBoost handling non‑linear interactions effectively in example workflows.

## Technical details
- Preprocessing: SimpleImputer(mean) + StandardScaler for numeric, SimpleImputer(most_frequent) + OneHotEncoder(drop='first') for Weather and Vehicle_Type, and OrdinalEncoder with explicit orders for Traffic_Level and Time_of_Day.
- Models searched: DecisionTreeRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, XGBRegressor, CatBoostRegressor with RandomizedSearchCV for tuning.
- Metrics: R², MSE, RMSE computed for train/test and captured via a RegressionMetricArtifact pattern in the trainer stage.

## Data paths and portability notes
Data Ingestion should read from the repository path Data/RawData/Food_Delivery_Times.csv rather than any hardcoded OS‑specific paths to maintain portability. Likewise, ensure any schema path variables in constants point to Data_Schema/__init__.py within the repo for cross‑platform consistency.

## How to extend
- Add features or categorical levels by updating Data_Schema.Required_Columns and the Data Transformation encoders and mappings accordingly.
- Add new models or tuning grids by editing Model_Trainer/model_trainer.py, then re‑run main.py to evaluate and log results.

## Requirements
All required packages are pinned in requirements.txt, including Streamlit, scikit‑learn, XGBoost, CatBoost, and MLflow, to standardize environments. Install from this file to replicate local development settings and avoid version drift across machines.

## Troubleshooting
- If the app fails to load the model, ensure final_model/model.pkl and final_model/preprocessor.pkl exist; re‑train with python main.py if missing.
- If training fails on data read or schema validation, verify Data/RawData/Food_Delivery_Times.csv exists and matches Data_Schema.Required_Columns.

## Author
Author: Sarvesh Chhabra (per setup.py metadata) and contributors as listed in the repository history or future contribution guidelines. Add a LICENSE and contribution guide to clarify usage rights and collaboration practices as the project evolves.

## Quick start
- Install dependencies: pip install -r requirements.txt from the repo root.
- Train the pipeline: python main.py to generate Artifacts and MLflow runs.
- Launch the app: streamlit run app.py to interactively predict delivery time.

## Collaboration
This README is aligned for readability and onboarding; feel free to refine sections, add badges, or include architecture diagrams as the project grows. A concise table of contents and consistent headings can further improve navigation for larger READMEs as more details are added.


