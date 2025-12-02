---
title: "Tabular Data 'Vibe Modeling' or Its Equivalent Using Gemini 3.0 Pro."
emoji: "🍀"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["googlecloud", "vertexai", "gemini"]
published: false
---

## Introduction

Recently, development methods utilizing generative AI are being adopted in various places. Among these, so-called Vibe Coding, where you provide instructions in natural language only, sometimes even building with hands-off driving, is being tested.

In my work, I often have opportunities to train and infer machine learning models using tabular data. Although the data differs for each project, I frequently feel that I am writing similar code every time for preprocessing or pipeline construction, or I think, "Can I offload this customization part to generative AI to make it more efficient? Can I do Vibe Modeling?"

Generally, in generative AI assistance for software development, the quality of the output tends to increase the more specific the instructions given are. In this article, I will generate machine learning pipeline code by varying the instruction level of the prompt (which I will arbitrarily call Vibe Modeling in this article) and examine how the generated code changes and its quality.

Furthermore, this article will not take an approach of iteratively refining the generated code through dialogue; instead, I will experiment with how far I can get with a single initial instruction or through the LLM's autonomous trial and error.

![](/images/articles/vibe-ml-gemini/vibe_modeling.jpg)
*An image of vibe modeling (really?)*

## Setup

The dataset used in this article is [Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci), and the beginning of the data looks like this. I have confirmed the data beforehand before attempting the subsequent code generation.

![](/images/articles/vibe-ml-gemini/data.png)
*Input data*

For the coding assistance tool, I'm using [Gemini CLI](https://github.com/google-gemini/gemini-cli), and I have configured the LLM as [Gemini 3.0 Pro](https://deepmind.google/models/gemini/pro/). Upon checking the execution logs, I also observed instances where the processing was automatically switched to Gemini 2.5 Flash for some operations. I will vary the instruction level of the prompt across the following three stages to confirm how the generated model training and inference code changes, and to check its quality.

- 1. Describe only the objective, problem setting, and important precautions.

    :::details 01_simple_requirements.txt
    ```text
    # Requirements for Model Training and Inference Notebook
    - Data to Use
        - ./data.csv
    - Problem Setting
        - Predict purchase amounts for the next 30 days using data from the 30 days before the prediction date
        - Aggregate by CustomerID to create features and target variables
    - Perform data exploration in advance and conduct necessary preprocessing
    - Validation Design
        - Decide the train/validation data split method based on data characteristics
        - Thoroughly check to absolutely avoid leakage
    - Other
        - Write code in `01_simple.py`
        - Run `uv run 01_simple.py` for verification
    ```
    :::

- 2. In addition to 1, promote trial and error to encourage improvement.

    :::details 02_try_and_error_requirements.txt
    ```text
    # Requirements for Model Training and Inference Notebook
    - Data to Use
        - ./data.csv
    - Problem Setting
        - Predict purchase amounts for the next 30 days using data from the 30 days before the prediction date
        - Aggregate by CustomerID to create features and target variables
    - Perform data exploration in advance and conduct necessary preprocessing
    - Validation Design
        - Decide the train/validation data split method based on data characteristics
        - Thoroughly check to absolutely avoid leakage
    - Other
        - After design and implementation, carefully consider whether the content is appropriate, verify operation, and improve/fix as needed
        - Write code in `02_try_and_error.py`
        - Run `uv run 02_try_and_error.py` for verification

    ```
    :::

- 3. In addition to 1, provide detailed descriptions of the pipeline and features (which correspond to what I intend to achieve)

    :::details 03_detail_requirements.txt
    ```text
    # Requirements for Model Training and Inference Notebook
    - Data to Use
        - ./data.csv
    - Problem Setting
        - Based on the prediction date, "predict purchase amounts for the next 30 days using data from the 30 days before the prediction date"
            - This setting may vary depending on data characteristics, business processes, and operational methods, so please consider a better approach based on actual constraints
        - For example, when the prediction date is 2011-03-01
            - Aggregate records from the 90 days before 2011-02-28 to create features
            - Use the sum of purchase amounts for the 30 days from 2011-03-01 onwards as the target variable
        - By shifting the prediction date, multiple sets of features and target variables can be created, which can be combined to prepare training and validation data
    - Based on the results of prior data exploration, perform the following preprocessing
        - Specify `encoding='shift-jis'` when loading data with `pd.read_csv` to avoid encoding errors
        - Type specification for `CustomerID` (`str`)
        - Convert the `InvoiceDate` column to datetime type
        - Remove records with missing `CustomerID`
        - Remove records with negative `Quantity` values
    - First, enable accuracy evaluation with simple features
        - For both features and target variables, aggregation should be unique by CustomerID and prediction_date
        - Features
            - Number of records
            - nunique of InvoiceNo
            - nunique of StockCode
            - nunique of InvoiceDate
            - mode of Country
            - sum, max, min, median of purchase amounts so far
        - Target Variable
            - Sum of (UnitPrice * Quantity)
        - Cross-validation

            | Fold | Training Data | Validation Data |
            | --- | --- | --- |
            | 1 | "2011-03-01", "2011-04-01", "2011-05-01" | "2011-06-01" |
            | 2 | "2011-04-01", "2011-05-01", "2011-06-01" | "2011-07-01" |
            | 3 | "2011-05-01", "2011-06-01", "2011-07-01" | "2011-08-01" |
            | 4 | "2011-06-01", "2011-07-01", "2011-08-01" | "2011-09-01" |
            | 5 | "2011-07-01", "2011-08-01", "2011-09-01" | "2011-10-01" |
            | test | "2011-08-01", "2011-09-01", "2011-10-01" | "2011-11-01" |

        - Use RMSE, commonly used in regression, as the objective function and evaluation metric
        - Use LightGBM as the model
            - Use the LightGBM Python API, not the scikit-learn wrapper
            - Use model parameters commonly found in public Kaggle notebooks
        - Pipeline
            - Convert categorical features with LabelEncoder before inputting to the model
            - Save artifacts obtained from training (lgb model, label encoder, feature columns) so they can be loaded during inference
            - Structure it so that inference can be executed independently
    - Other
        - After design and implementation, carefully consider whether the content is appropriate, verify operation, and improve on your own as needed
        - Write code in `03_detail.py`
        - Run `uv run 03_detail.py` for verification
    ```
    :::

With the Gemini CLI, I instructed the code implementation while specifying a separately created requirements file. The basic prompt is as follows, and I change the section marked `@01_simple_requirements.txt` according to each condition.

```text
First, please review @01_simple_requirements.txt.
Based on this document, please implement the model training and inference code.
```

## 1. Describe only the objective, problem setting, and important precautions.

I will check what kind of code comes out with a single, one-shot prompt. The requirements only include the objective and the minimum necessary precautions.
The generated code is as follows.

:::details 1. Code with only objective and problem setting
```python
# /// script
# requires-python = ">=3.9"
dependencies = [
    "pandas",
    "scikit-learn",
    "numpy",
]
# ///

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import warnings

# Suppress warnings (such as FutureWarning)
warnings.filterwarnings('ignore')

def load_and_preprocess(filepath):
    """Load data and perform basic preprocessing"""
    print("Loading data...")
    # Load with date parsing
    try:
        df = pd.read_csv(filepath, parse_dates=['InvoiceDate'], encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, parse_dates=['InvoiceDate'], encoding='cp1252')
    
    print(f"Original data shape: {df.shape}")

    # Remove data without CustomerID as it cannot be used for per-customer prediction
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int)

    # Calculate amount
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    # Create date-only column (discard time)
    df['Date'] = df['InvoiceDate'].dt.normalize()
    
    print(f"Data shape after dropping null CustomerID: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df

def create_features_and_target(df, cutoff_date, target_days=30, feature_days=30):
    """
    Create features and target variable based on the specified cutoff_date

    Features: Data from the period [cutoff_date - feature_days, cutoff_date)
    Target: Data from the period [cutoff_date, cutoff_date + target_days)
    """

    # Define periods
    feature_start = cutoff_date - pd.Timedelta(days=feature_days)
    feature_end = cutoff_date # Up to the day before cutoff_date (not including cutoff_date, or up to 00:00 on cutoff_date)
    
    target_start = cutoff_date
    target_end = cutoff_date + pd.Timedelta(days=target_days)

    # Data filtering
    # Data for features
    feat_df = df[(df['Date'] >= feature_start) & (df['Date'] < feature_end)]

    # Data for target variable
    target_df = df[(df['Date'] >= target_start) & (df['Date'] < target_end)]

    # Base customer list (whether to base on customers who existed in the feature period or all customers)
    # Here, both "customers who had actions in the feature period" and "customers who had actions in the target period" should be considered,
    # but since we're "predicting purchase amounts based on the prediction date...",
    # the actual targets are "customers active at the prediction time" or "all customers".
    # For simplicity, we target customers who have data in either the feature period or the target period.
    # However, during training, the common format is to link "future sales" to "customers for whom features can be obtained".
    # Even if there are no purchases in the feature period, they might be prediction targets if they were registered in the past,
    # but since this dataset has no customer master, we only know customers with transactions.
    # Therefore, we base on customers that exist in feat_df (if no purchases in the last 30 days = 0 yen, we would need to expand the base).

    # Since the requirement is "using data from the last 30 days",
    # we use customers with data in the feature period as the main population,
    # while those with data only in the target period (new/returning) are excluded from the training data this time as they are difficult to predict without historical data,
    # or could be treated as having 0 features.
    # For simplification, we target "customers with purchase history in the feature period".

    base_customers = feat_df['CustomerID'].unique()

    # Feature aggregation
    features = feat_df.groupby('CustomerID').agg({
        'TotalAmount': ['sum', 'mean', 'count'],
        'Quantity': ['sum'],
        'InvoiceDate': ['max'] # For calculating last purchase date
    })
    features.columns = ['_'.join(col).strip() for col in features.columns.values]

    # Recency (cutoff_date - last purchase date)
    features['Recency'] = (cutoff_date - features['InvoiceDate_max'].dt.normalize()).dt.days
    features = features.drop('InvoiceDate_max', axis=1)

    # Rename columns
    features = features.add_prefix('Past30_')

    # Target variable aggregation
    targets = target_df.groupby('CustomerID')['TotalAmount'].sum().rename('Target_Next30_Amount')

    # Join (Left Join: Link correct answers to customers with features. If no correct answer, treat as 0 yen)
    data = pd.DataFrame(index=base_customers)
    data.index.name = 'CustomerID'

    data = data.join(features, how='left')
    data = data.join(targets, how='left')

    # Fill missing values
    # Features shouldn't have missing values since base_customers comes from feat_df, but just in case
    data = data.fillna(0)
    
    return data

def main():
    # 1. Load data
    df = load_and_preprocess('data.csv')

    # Check maximum date in data
    max_date = df['Date'].max()
    min_date = df['Date'].min()
    print(f"Data covers from {min_date} to {max_date}")

    # 2. Validation design
    # Split by time series to avoid leaking future information.
    # Test cutoff date: 30 days before the end of the dataset (this allows using the last 30 days as test data ground truth)
    test_cutoff = max_date - pd.Timedelta(days=30)
    
    print(f"\n--- Preparing Test Set (Cutoff: {test_cutoff}) ---")
    test_data = create_features_and_target(df, test_cutoff)
    print(f"Test data shape: {test_data.shape}")

    # Training cutoff date: Set so it doesn't overlap with the test period.
    # Test period feature period: [test_cutoff - 30, test_cutoff)
    # Test period target variable period: [test_cutoff, test_cutoff + 30)
    #
    # It's safe for training data to use only information before the test data's "feature period",
    # but for training the prediction model itself, we just need "past features" and "past ground truth (future at that point)".
    # To prevent leakage, we need to ensure that the training data's "target variable period" doesn't overlap with the test data's "target variable period" (or doesn't contain information beyond the test data's feature calculation time).
    # The safest approach is for the training data's target variable period end date to be before or on the test data's cutoff date.
    # In other words: Train_Target_End <= Test_Cutoff
    # Train_Cutoff + 30 <= Test_Cutoff
    # Train_Cutoff <= Test_Cutoff - 30

    # Create several training data with 30-day sliding window to increase data
    train_cutoffs = []
    current_cutoff = test_cutoff - pd.Timedelta(days=60) # Start from 60 days before with some margin

    while current_cutoff > min_date + pd.Timedelta(days=30):
        train_cutoffs.append(current_cutoff)
        current_cutoff -= pd.Timedelta(days=30) # Go back 30 days at a time
    
    print(f"\n--- Preparing Train Set (Cutoffs: {[t.date() for t in train_cutoffs]}) ---")
    
    train_dfs = []
    for cutoff in train_cutoffs:
        train_dfs.append(create_features_and_target(df, cutoff))
    
    if not train_dfs:
        print("Error: Not enough data to create training set.")
        return

    train_data = pd.concat(train_dfs)
    print(f"Train data shape: {train_data.shape}")

    # 3. Model training
    X_train = train_data.drop('Target_Next30_Amount', axis=1)
    y_train = train_data['Target_Next30_Amount']

    X_test = test_data.drop('Target_Next30_Amount', axis=1)
    y_test = test_data['Target_Next30_Amount']

    print("\n--- Training Model ---")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 4. Evaluation
    print("\n--- Evaluation ---")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Feature importances
    print("\n--- Feature Importances ---")
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances)

    # Specific prediction examples
    print("\n--- Prediction Examples (Top 5) ---")
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    print(results.head())

if __name__ == "__main__":
    main()
```
:::

### Evaluation of Results

Upon examining the code contents, several positive aspects were found. First, in the initial generation, it confirmed that an encoding error occurred when executing `pd.read_csv` and modified it to specify the appropriate encoding. Additionally, the training and test data were split by offsetting their time periods, showing awareness of leakage prevention. Furthermore, it incorporated the perspective of RFM analysis (Recency, Frequency, Monetary), which is common in marketing analysis, and created features considering Recency and Monetary.
On the other hand, there were several areas for improvement. First, despite having sufficient data, cross-validation was not included to improve generalization performance. Also, Random Forest was used as the model, whereas gradient boosting models such as LightGBM, which are [recommended by the Kaggle community as a first choice](https://upura.hatenablog.com/entry/2019/10/29/184617) for tabular data, would have been preferable.
In data preprocessing, prior data exploration revealed that Quantity contained negative values, and since there was no apparent reason or explanation for them, it would be prudent to remove records with negative values. However, the generated code did not consider negative values and aggregated them as-is. There were also no features related to Frequency. The lack of inference-only processing was another area that fell short.

## 2. In addition to 1, promote trial and error to encourage improvement.

Next is a condition where I instruct the LLM to iterate and improve through trial and error in addition to 1. I expect the LLM to analyze the data itself and write better code. Gemini first generated and executed code to analyze the data and verify it.

:::details 2. Code when conveying design overview and encouraging improvement through trial and error (Data Exploration)
```python
# /// script
# requires-python = ">=3.10"
dependencies = [
    "pandas",
    "numpy",
]
# ///

import pandas as pd
import numpy as np

def analyze_data():
    print("Loading data...")
    # Load with date parsing specified
    try:
        df = pd.read_csv('./data.csv', parse_dates=['InvoiceDate'])
    except Exception as e:
        print(f"Failed to read csv with date parsing: {e}")
        # The format might be special, so read normally
        df = pd.read_csv('./data.csv')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    print("\nData Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nData Range:")
    print(f"Start: {df['InvoiceDate'].min()}")
    print(f"End: {df['InvoiceDate'].max()}")

    # Check data without CustomerID since per-customer prediction is not possible
    missing_customer = df['CustomerID'].isnull().sum()
    print(f"\nMissing CustomerID count: {missing_customer} ({missing_customer/len(df):.2%})")

    # Descriptive statistics
    print("\nDescriptive Stats:")
    print(df.describe())

if __name__ == "__main__":
    analyze_data()
```
:::

Based on these analysis results, the generated training code is as follows.

:::details 2. Code when conveying design overview and encouraging improvement through trial and error (Training and Inference)
```python
# /// script
# requires-python = ">=3.10"
dependencies = [
    "pandas",
    "numpy",
    "scikit-learn",
    "lightgbm",
]
# ///

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(file_path='./data.csv'):
    """
    Load data from CSV file. Handles different encodings.
    """
    print(f"Loading data from {file_path}...")
    try:
        # Try default UTF-8 first
        df = pd.read_csv(file_path, parse_dates=['InvoiceDate'])
    except UnicodeDecodeError:
        # Fallback to ISO-8859-1 (common for UK data)
        print("UTF-8 decode failed, trying ISO-8859-1...")
        df = pd.read_csv(file_path, encoding='ISO-8859-1', parse_dates=['InvoiceDate'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    return df

def preprocess(df):
    """
    Basic preprocessing:
    - Remove missing CustomerID
    - Calculate TotalAmount
    - Ensure correct data types
    """
    print("Preprocessing data...")
    
    # Drop missing CustomerID as we need to aggregate by customer
    initial_rows = len(df)
    df = df.dropna(subset=['CustomerID'])
    print(f"Dropped {initial_rows - len(df)} rows with missing CustomerID.")
    
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    # Calculate TotalAmount
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    # Ensure InvoiceDate is datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    return df

def create_features(df_history, ref_date):
    """
    Calculate features from historical data for each customer.
    """
    # Aggregations
    aggs = df_history.groupby('CustomerID').agg({
        'TotalAmount': ['sum', 'mean', 'count', 'std', 'min', 'max'],
        'Quantity': ['sum', 'mean'],
        'InvoiceDate': ['max'] # Used for Recency
    })
    
    # Flatten MultiIndex columns
    aggs.columns = ['_'.join(col).strip() for col in aggs.columns.values]
    
    # Recency: Days since last purchase
    # Note: (ref_date - last_purchase)
    aggs['Recency'] = (ref_date - aggs['InvoiceDate_max']).dt.days
    aggs = aggs.drop(columns=['InvoiceDate_max'])
    
    # Fill NaN values that might result from std calculation with single record
    aggs = aggs.fillna(0)
    
    return aggs

def create_dataset(df, lookback_days=30, target_days=30, step_days=7):
    """
    Create a dataset using a sliding window approach.
    
    Parameters:
    - lookback_days: Number of days to look back for features (X)
    - target_days: Number of days to look forward for target (y)
    - step_days: Step size for moving the window
    """
    start_date = df['InvoiceDate'].min()
    end_date = df['InvoiceDate'].max()
    
    print(f"Data Date Range: {start_date} to {end_date}")
    
    # Generate reference dates
    # Logic:
    # Feature Window: [ref_date - lookback, ref_date - 1]
    # Target Window:  [ref_date, ref_date + target - 1]
    
    # First ref_date must allow for full lookback
    first_ref = start_date + pd.Timedelta(days=lookback_days)
    # Last ref_date must allow for full target window
    last_ref = end_date - pd.Timedelta(days=target_days)
    
    if first_ref > last_ref:
        print("Error: Data duration is too short for the specified lookback and target periods.")
        return None, None, None

    ref_dates = pd.date_range(start=first_ref, end=last_ref, freq=f'{step_days}D')
    
    print(f"Generating dataset for {len(ref_dates)} reference dates (Sliding Window)...")
    
    X_list = []
    y_list = []
    meta_list = [] 
    
    # Sort dataframe once for faster slicing
    df = df.sort_values('InvoiceDate')
    
    for ref_date in ref_dates:
        # Define time windows
        feat_start = ref_date - pd.Timedelta(days=lookback_days)
        feat_end = ref_date - pd.Timedelta(days=1)
        
        target_start = ref_date
        target_end = ref_date + pd.Timedelta(days=target_days) # Exclusive in slice logic below if we use <
        
        # Slicing
        # Note: slice is inclusive for start, inclusive for end if using label-based indexing with datetime index,
        # but here we use boolean masking.
        # Features: [start, end] inclusive
        mask_feat = (df['InvoiceDate'] >= feat_start) & (df['InvoiceDate'] <= feat_end)
        df_feat = df.loc[mask_feat]
        
        # Target: [start, end) - let's be precise: target_days from ref_date
        mask_target = (df['InvoiceDate'] >= target_start) & (df['InvoiceDate'] < target_start + pd.Timedelta(days=target_days))
        df_target = df.loc[mask_target]
        
        if df_feat.empty:
            continue
            
        # 1. Calculate Features
        # We focus on customers who were active in the lookback period.
        features = create_features(df_feat, ref_date)
        
        # 2. Calculate Target
        # Sum of TotalAmount per customer in target window
        targets = df_target.groupby('CustomerID')['TotalAmount'].sum().rename('TargetAmount')
        
        # 3. Merge
        # Left join: We predict for customers observed in the feature window.
        # Customers who don't buy in the target window get 0 target amount.
        dataset = features.join(targets, how='left').fillna({'TargetAmount': 0})
        
        # Store metadata
        dataset['RefDate'] = ref_date
        
        X_list.append(dataset.drop(columns=['TargetAmount', 'RefDate']))
        y_list.append(dataset['TargetAmount'])
        meta_list.append(dataset[['RefDate']].reset_index()) # Includes CustomerID

    if not X_list:
        return None, None, None

    X = pd.concat(X_list)
    y = pd.concat(y_list)
    meta = pd.concat(meta_list)
    
    return X, y, meta

def train_and_evaluate(X, y, meta):
    """
    Split data based on time, train model, and evaluate.
    """
    # Chronological Split
    # Use the last 20% of reference dates for testing to avoid leakage
    unique_dates = sorted(meta['RefDate'].unique())
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]
    
    print(f"\nSplitting data at RefDate: {split_date}")
    
    # Create masks based on RefDate aligned with X/y rows
    train_mask = meta['RefDate'] < split_date
    test_mask = meta['RefDate'] >= split_date
    
    X_train = X[train_mask.values]
    y_train = y[train_mask.values]
    X_test = X[test_mask.values]
    y_test = y[test_mask.values]
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    
    # Model Training (LightGBM)
    print("\nTraining LightGBM model...")
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Prediction
    preds = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    print("------------------------------")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print("------------------------------")
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Features:")
    print(importance.head(5))
    
    return model, preds, y_test

def main():
    # 1. Load
    df = load_data()
    if df is None:
        return

    # 2. Preprocess
    df = preprocess(df)
    
    # 3. Create Dataset
    print("\n--- Creating Dataset ---")
    X, y, meta = create_dataset(df, lookback_days=30, target_days=30, step_days=7)
    
    if X is None:
        print("Failed to create dataset.")
        return
        
    print(f"Total samples created: {X.shape[0]}")
    
    # 4. Train & Evaluate
    print("\n--- Model Training & Evaluation ---")
    train_and_evaluate(X, y, meta)

if __name__ == "__main__":
    main()
```
:::

### Evaluation of Results

Although it didn't iterate as much as I had expected, the following improvements were made in the initial output, resulting in the final code:
- Added comments
- Added LightGBM parameters
- Properly functionalized and refactored the processing within the `main` function

However, the main points regarding positive aspects and areas for improvement were not much different from 1. Even when including instructions to "iterate and improve," it seems difficult to achieve clear improvements unless specific implementation goals and requirements such as features to prioritize are defined.


## 3. In addition to 1, provide detailed descriptions of the pipeline and features

Finally, this is a pattern where humans check the data in advance and write specific instructions based on that content. In the requirements document for 3, I described the pipeline structure and features to be created as much as I could think of.

The generated code is as follows.

:::details 3. Describing detailed pipeline and features
```python
# /// script
# requires-python = ">=3.9"
dependencies = [
    "pandas",
    "numpy",
    "lightgbm",
    "scikit-learn",
]
# ///

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = './data.csv'
ARTIFACTS_DIR = './artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Feature generation settings
# The requirements document says "30 days of data up to the previous day", but the example says "aggregate 90 days of records",
# so we adopt the 90 days from the example considering the amount of information.
FEATURE_WINDOW_DAYS = 90
TARGET_WINDOW_DAYS = 30

def load_and_preprocess(file_path):
    print("Loading data...")
    # Specify encoding='shift-jis' when loading data with pd.read_csv
    # CustomerID cannot be specified as str at load time in some cases (becomes float when NA exists), so convert later
    df = pd.read_csv(file_path, encoding='shift-jis')

    # Convert InvoiceDate column to datetime type
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Remove records with missing CustomerID
    df = df.dropna(subset=['CustomerID'])

    # Convert CustomerID to str (e.g., 12345.0 -> "12345")
    df['CustomerID'] = df['CustomerID'].astype(float).astype(int).astype(str)

    # Remove records with negative Quantity values
    df = df[df['Quantity'] > 0]

    # Add amount column
    df['Amount'] = df['Quantity'] * df['UnitPrice']
    
    return df.reset_index(drop=True)

def create_features(df, prediction_date):
    """
    Create features based on the specified prediction_date
    Target data: [prediction_date - FEATURE_WINDOW_DAYS, prediction_date)
    """
    start_date = prediction_date - pd.Timedelta(days=FEATURE_WINDOW_DAYS)
    end_date = prediction_date

    mask = (df['InvoiceDate'] >= start_date) & (df['InvoiceDate'] < end_date)
    target_df = df[mask]

    if target_df.empty:
        return pd.DataFrame()

    # Function to get mode of Country
    def get_mode(x):
        m = x.mode()
        if not m.empty:
            return m.iloc[0]
        return np.nan

    # Aggregation definitions
    # InvoiceNo: nunique
    # StockCode: nunique
    # InvoiceDate: nunique
    # Country: mode
    # Amount: sum, max, min, median
    # count: Use count of InvoiceNo as the number of records
    agg_funcs = {
        'InvoiceNo': ['count', 'nunique'],
        'StockCode': ['nunique'],
        'InvoiceDate': ['nunique'],
        'Country': [get_mode],
        'Amount': ['sum', 'max', 'min', 'median']
    }

    grouped = target_df.groupby('CustomerID').agg(agg_funcs)

    # Flatten column names
    new_cols = []
    for col, stat in grouped.columns:
        if stat == '<lambda>' or stat == 'get_mode': # get_mode
            stat = 'mode'
        if col == 'InvoiceNo' and stat == 'count':
            new_cols.append('record_count')
        else:
            new_cols.append(f'{col}_{stat}')

    grouped.columns = new_cols
    grouped = grouped.reset_index()

    # Add prediction_date
    grouped['prediction_date'] = prediction_date

    return grouped

def create_target(df, prediction_date):
    """
    Create target variable based on the specified prediction_date
    Target data: [prediction_date, prediction_date + TARGET_WINDOW_DAYS)
    """
    start_date = prediction_date
    end_date = prediction_date + pd.Timedelta(days=TARGET_WINDOW_DAYS)
    
    mask = (df['InvoiceDate'] >= start_date) & (df['InvoiceDate'] < end_date)
    target_df = df[mask]
    
    if target_df.empty:
        return pd.DataFrame(columns=['CustomerID', 'target_amount'])

    grouped = target_df.groupby('CustomerID')['Amount'].sum().reset_index()
    grouped.rename(columns={'Amount': 'target_amount'}, inplace=True)
    
    return grouped

def make_dataset(df, prediction_dates, is_train=True):
    features_list = []
    
    for date_str in prediction_dates:
        pred_date = pd.to_datetime(date_str)
        # print(f"Processing prediction date: {pred_date}")

        # Create features
        feats = create_features(df, pred_date)
        if feats.empty:
            continue

        if is_train:
            # Create target variable
            targets = create_target(df, pred_date)
            # Left join: Join target to users with features
            merged = pd.merge(feats, targets, on='CustomerID', how='left')
            # Set users with no purchases in the period to 0 yen
            merged['target_amount'] = merged['target_amount'].fillna(0)
            features_list.append(merged)
        else:
            features_list.append(feats)
            
    if not features_list:
        return pd.DataFrame()
        
    return pd.concat(features_list, ignore_index=True)

def main():
    # 1. Load data
    df = load_and_preprocess(INPUT_FILE)
    print(f"Data loaded. Shape: {df.shape}")

    # 2. Define folds
    folds = [
        {
            'train': ["2011-03-01", "2011-04-01", "2011-05-01"],
            'valid': ["2011-06-01"]
        },
        {
            'train': ["2011-04-01", "2011-05-01", "2011-06-01"],
            'valid': ["2011-07-01"]
        },
        {
            'train': ["2011-05-01", "2011-06-01", "2011-07-01"],
            'valid': ["2011-08-01"]
        },
        {
            'train': ["2011-06-01", "2011-07-01", "2011-08-01"],
            'valid': ["2011-09-01"]
        },
        {
            'train': ["2011-07-01", "2011-08-01", "2011-09-01"],
            'valid': ["2011-10-01"]
        }
    ]
    
    # Test set definition
    test_set_def = {
        'train': ["2011-08-01", "2011-09-01", "2011-10-01"],
        'test': ["2011-11-01"]
    }

    # Prepare label encoding
    print("Preparing label encoder...")
    le = LabelEncoder()
    # Assuming no missing values, but use astype(str) just in case
    le.fit(df['Country'].astype(str).unique())

    # Save
    with open(os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    # Model parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    # Cross-validation loop
    models = []
    rmse_scores = []
    feature_columns = None

    print("Starting cross-validation...")
    for i, fold in enumerate(folds):
        print(f"\n=== Fold {i+1} ===")

        # Create training data
        train_df = make_dataset(df, fold['train'], is_train=True)
        valid_df = make_dataset(df, fold['valid'], is_train=True)

        if train_df.empty or valid_df.empty:
            print("Skipping fold due to empty data.")
            continue

        # Identify feature columns
        drop_cols = ['CustomerID', 'prediction_date', 'target_amount']
        features = [c for c in train_df.columns if c not in drop_cols]
        feature_columns = features # Keep for saving

        X_train = train_df[features].copy()
        y_train = train_df['target_amount']
        X_valid = valid_df[features].copy()
        y_valid = valid_df['target_amount']

        # Convert categorical variables
        if 'Country_mode' in X_train.columns:
            X_train['Country_mode'] = le.transform(X_train['Country_mode'].astype(str))
            X_valid['Country_mode'] = le.transform(X_valid['Country_mode'].astype(str))
            categorical_feats = ['Country_mode']
        else:
            categorical_feats = []

        # LGBM Dataset
        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feats)
        lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_feats)

        # Training
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_eval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )
        
        models.append(model)

        # Evaluation
        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        rmse_scores.append(rmse)
        print(f"Fold {i+1} RMSE: {rmse:.4f}")

    if rmse_scores:
        print(f"\nAverage RMSE: {np.mean(rmse_scores):.4f}")

    # Save artifacts
    if models:
        print("\nSaving artifacts...")
        best_model = models[-1] # Use the last model for simplicity
        best_model.save_model(os.path.join(ARTIFACTS_DIR, 'lgb_model.txt'))

        with open(os.path.join(ARTIFACTS_DIR, 'feature_columns.pkl'), 'wb') as f:
            pickle.dump(feature_columns, f)
        print(f"Artifacts saved to {ARTIFACTS_DIR}")

    # Inference test
    print("\n=== Inference on Test Set (Simulation) ===")
    # Perform prediction on test set dates
    test_date_str = test_set_def['test'][0]
    print(f"Predicting for date: {test_date_str}")

    test_features_df = make_dataset(df, [test_date_str], is_train=False)

    if not test_features_df.empty and os.path.exists(os.path.join(ARTIFACTS_DIR, 'feature_columns.pkl')):
        # Load
        with open(os.path.join(ARTIFACTS_DIR, 'feature_columns.pkl'), 'rb') as f:
            loaded_features = pickle.load(f)
        with open(os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl'), 'rb') as f:
            loaded_le = pickle.load(f)
        loaded_model = lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, 'lgb_model.txt'))

        X_test = test_features_df[loaded_features].copy()

        if 'Country_mode' in X_test.columns:
            X_test['Country_mode'] = loaded_le.transform(X_test['Country_mode'].astype(str))

        preds = loaded_model.predict(X_test)
        test_features_df['predicted_amount'] = preds

        # Display
        print("Top 5 predictions:")
        print(test_features_df[['CustomerID', 'prediction_date', 'predicted_amount']].head())

        # Evaluation (compare with actual values)
        print("Evaluating test set performance...")
        actual_target = create_target(df, pd.to_datetime(test_date_str))
        merged_res = pd.merge(test_features_df, actual_target, on='CustomerID', how='left')
        merged_res['target_amount'] = merged_res['target_amount'].fillna(0)

        test_rmse = np.sqrt(mean_squared_error(merged_res['target_amount'], merged_res['predicted_amount']))
        print(f"Test Set RMSE: {test_rmse:.4f}")

if __name__ == "__main__":
    main()
```
:::

### Evaluation of Results

Although the content described in the requirements document is mostly reflected in the implementation, there are still some areas for improvement upon closer examination. For example, the LabelEncoder `fit` is performed before splitting the training and validation data, which poses a risk of leakage. Additionally, the configuration doesn't output training progress logs, and personally I would appreciate being able to check progress during training (though I think this is a matter of personal preference).
However, with this code, it seems possible to move on to the next steps such as trial and error for accuracy improvement with just minor adjustments. In the cases of 1 and 2, there were many additional adjustments needed, resulting in the need to continue conversations or add implementations myself. Personally (partly because I don't particularly enjoy frequent back-and-forth conversations), method 3 seemed to best suit my preferences. I think the balance in this regard is also influenced by individual preferences.


## Conclusion

In this experiment, Vibe Modeling (my arbitrary term) for machine learning model training and inference code using generative AI did not produce significantly off-target code even with minimal instructions, but it did leave some areas where I thought "I wish it did this differently." When describing what I wanted to achieve in detail, it was mostly reflected in the implementation, but the human side needed to patiently perform data exploration before being able to describe what they wanted, and naturally, I felt a trade-off between the effort of preparation and the quality of the generated output.
While the effort of implementation itself can be saved and code generation can be leveraged even for data exploration, making it much easier than implementing from scratch, my honest current impression is that I'm still uncertain about the balance of work division between AI and humans.
Model performance will continue to improve in the future, and I hope to continue exploring comfortable workflows and division strategies.
