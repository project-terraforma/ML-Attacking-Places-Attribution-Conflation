# Rule-Based & ML Attacking Places Attribution Conflation
CRWN 102 Project B | Shreya Handa, Sriya Ramachandruni

# Project Overview
This project solves places attributes conflation, a common issue within the geospatial industry. The project tests two methods, a rule-based algorithm and machine learning model, comparing which would be best for long-usability.

## Repository Structure

```markdown
project/
├── Archived/                         # Old pipeline used earlier in the project (logic has since been changed)
│     ├── cleanData.py                # Original OMF/Yelp cleaning + matching script
│     ├── cleaned_omf_data.csv       
│     ├── cleaned_yelp_data.csv
│     ├── fuzzy_matched_candidates.csv
│     ├── matched_candidates.csv
│     ├── rule_based_conflation.py    # First version of conflation algorithm
│     └── rule_based_conflation_v2.py # Improved rule-based algorithm
│
├── Machine Learning-Based/           # All ML models, training code, inference, evaluation
│     ├── models/                     # Saved .joblib trained models for each attribute
│     │     ├── address_model.joblib
│     │     ├── categories_model.joblib
│     │     ├── name_model.joblib
│     │     ├── phone_model.joblib
│     │     └── website_model.joblib
│     │
│     ├── ML_BEST_ATTRIBUTES.csv         # Final ML predictions (per place_id)
│     ├── ML_GOLDEN_DATASET_TEMPLATE.csv # Labeled truth set for training/evaluation
│
│     ├── ML_INFER_FEATURES_address.csv   # Features used during ML inference
│     ├── ML_INFER_FEATURES_categories.csv
│     ├── ML_INFER_FEATURES_name.csv
│     ├── ML_INFER_FEATURES_phone.csv
│     ├── ML_INFER_FEATURES_website.csv
│
│     ├── ML_TRAIN_FEATURES_address.csv   # Features extracted from golden dataset
│     ├── ML_TRAIN_FEATURES_categories.csv
│     ├── ML_TRAIN_FEATURES_name.csv
│     ├── ML_TRAIN_FEATURES_phone.csv
│     ├── ML_TRAIN_FEATURES_website.csv
│
│     ├── ml_best_attributes.py        # Main ML inference pipeline
│     ├── ml_eval.py                   # ML evaluation script (attribute-based accuracy)
│     └── ml_golden.py                 # Builds ML golden dataset template
│
├── Rule-Based/                       # Production-ready rule-based system
│     ├── RULE_BEST_ATTRIBUTES.csv    # Output predictions from rule-based algorithm
│     ├── RULE_GOLDEN_DATASET_TEMPLATE.csv # Labeled truth set for rule-based evaluation
│     ├── rule_best_attributes.py     # Main rule-based pipeline
│     ├── rule_eval.py                # Rule-based evaluation script (attribute-based accuracy)
│     └── rule_golden.py              # Builds rule-based golden dataset template
│
├── __pycache__/                      # Python cache (auto-generated)
│
├── .gitignore
├── LICENSE
├── NORMALIZED_SOURCES.csv            # Master normalized source file (shared by ML + Rule)
│                                     # Both systems import from this file
│
├── OMF_Yelp_compare.py               # Script comparing OMF vs Yelp coverage
├── OMF_normalize_data.py             # Normalization pipeline used before conflation
│                                     # Outputs NORMALIZED_SOURCES.csv
│
├── README.md                         # Documentation (this file)
├── VALID_MATCHES.csv                 # Verified matches used for truth assignment
├── project_b_samples_2k.csv
├── project_b_samples_2k.parquet       # Original OMF sample dataset
└── requirements.txt                   # Project dependencies
```

## Installation & Usage
### Installation
#### 1. Clone the repository
#### 2. Create a virtual environment
```markdown
python3 -m venv .venv
source .venv/bin/activate     # Linux/macOS
# or
.\.venv\Scripts\activate      # Windows
```
#### 3. Install dependencies
```markdown
pip install -r requirements.txt
```
### Usage
#### 1. Normalize and Prepare the OMF Data
If you need to re-generate NORMALIZED_SOURCES.csv:
```markdown
python3 OMF_normalize_data.py
```
Outputs: ```NORMALIZED_SOURCES.csv``` used by both the rule-based and ML pipelines. 
### Rule-Based Pipeline
The rule-based system chooses the best attribute value using confidence scoring and tie-breaking logic. 
#### 1. Build the golden dataset template
```markdown
python3 Rule-Based/rule_golden.py
```
Outputs:
```
Rule-Based/RULE_GOLDEN_DATASET_TEMPLATE.csv
```
#### 2. Run the rule-based conflation
```
python3 Rule-Based/rule_best_attributes.py
```
Outputs:
```
Rule-Based/RULE_BEST_ATTRIBUTES.csv
```
#### 3. Evaluate rule-based accuracy
Outputs predicted best attribute accuracies (name, phone, address, categories, website).
```
python3 Rule-Based/rule_eval.py
```
### Machine Learning-Based Pipeline
The ML pipeline trains a classifier per attribute and predicts the best provider per place. 

#### 1. Build ML golden dataset
```markdown
python3 Machine\ Learning-Based/ml_golden.py
```
Outputs:
```
Machine Learning-Based/ML_GOLDEN_DATASET_TEMPLATE.csv
```
#### 2. Train all ML models + run inference
Trains models (```models/*.joblib```), creates train feature files(```ML_TRAIN_FEATURES_*.csv```), creates inference feautre files (```ML_INFER_FEATURES_*.csv```), produces final attribute predictions (```ML_BEST_ATTRIBUTES.csv```). 
```
python3 Machine\ Learning-Based/ml_best_attributes.py
```
Outputs:
```
models/*.joblib
ML_TRAIN_FEATURES_*.csv
ML_INFER_FEATURES_*.csv
ML_BEST_ATTRIBUTES.csv
```
#### 3. Evaluate ML accuracy
Outputs predicted best attribute accuracies (name, phone, address, categories, website).
```
python3 Machine\ Learning-Based/ml_eval.py
```
## Results Summary
```
=== RULE-BASED ACCURACY (NORMALIZED) ===

name      : 88.24%
phone     : 94.12%
address   : 100.00%
categories: 64.71%
website   : 68.75%

OVERALL ACCURACY: 83.16%

=== ML VALUE-BASED ACCURACY ===

name      : 76.00%
phone     : 88.00%
address   : 100.00%
website   : 88.00%
category  : 76.00%

OVERALL ACCURACY: 85.60%
```
## Project Objective & Key Results (OKRs)

**Objective:** Develop rule-based and machine learning pipelines for resolving place-attribute conflation.

- **KR1**  
  Build and document a normalized dataset covering **20+ location patterns** that are reused by **both** the rule-based and ML pipelines, and validate that it has **<10% parsing errors**.

- **KR2**  
  Implement **5 rules per attribute** (name, phone, address, website, categories) and achieve **≥ 80% attribute-level accuracy** on the rule-based conflation dataset.

- **KR3**  
  Train **5 ML models** (one per attribute) and achieve **≥ 80% normalized, attribute-based accuracy** on the ML-based conflation dataset.

- **KR4**  
  By Week 10, deliver evaluation comparing rule-based and ML pipelines, identifying **2 strengths and 2 limitations** of each, and concluding which approach is preferable for **long-term maintainability** and **explainability**.
