---
title: "Tabular 'Vibe Modeling' or Its Equivalent with Gemini 3.0 Pro. (Conv.)"
emoji: "🍀"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["googlecloud", "vertexai", "gemini"]
published: false
---

This is the article for Day 19 of the [JP_Google Developer Experts Advent Calendar 2025](https://adventar.org/calendars/11658).

## Introduction

In the [previous article](https://zenn.dev/kikuriyou/articles/2511271718_vibe-ml-gemini_en), I examined the quality of generated code when producing model training and inference code for tabular data in a single shot.

In this article, I'll experiment with starting from a simple requirement to generate code, then continuing the conversation to gradually approach the desired outcome. The rule is to proceed only through instructions to the AI without making any direct code changes myself.

I'll verify how well I can achieve improvements according to my intentions through conversation, and compare the differences in generated results and subjective experience versus providing detailed requirements upfront as in the previous article.

## Experiment Setup

The initial setup is essentially the same as in the previous article.

- Coding assistance: Gemini CLI
- LLM: Gemini 3.0 Pro
- Data: [Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)

The instruction to Gemini CLI for model training and inference is as follows. I'll start with the same instruction as pattern 1 (only purpose and problem definition) from the previous article, then continue the conversation to attempt improvements.

```docker
First, please check @01_simple_requirements.txt.
Based on this document, implement model training and inference code.
```

Throughout this experiment, please note that AI-generated code includes changes from automatic linting.

## 1. When You Know What You Want to Achieve

First, let's examine a pattern where the final desired configuration (LightGBM, cross-validation, specific features, etc.) is determined, but we start with simple instructions and gradually add more.

### Initial Generation

First, let's check the code generated in a single shot.

:::details Initial Generated Code
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def preprocess_data(df):
    df = df.copy() # Operate on a copy to avoid SettingWithCopyWarning
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Calculate Amount
    df['Amount'] = df['Quantity'] * df['UnitPrice']

    # Drop rows with missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int)

    # Filter out returns (negative Quantity) and invalid prices/quantities
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # Sort by CustomerID and InvoiceDate for time-series operations
    df = df.sort_values(by=['CustomerID', 'InvoiceDate'])

    return df

def create_features_and_target(df, prediction_date):
    # Data for feature window (past 30 days relative to prediction_date)
    feature_start_date = prediction_date - pd.Timedelta(days=30)
    feature_end_date = prediction_date - pd.Timedelta(days=1) # Up to day before prediction date

    # Data for target window (next 30 days from prediction_date)
    target_start_date = prediction_date
    target_end_date = prediction_date + pd.Timedelta(days=29) # Inclusive, so 30 days

    # Features: aggregate past 30 days purchases for each customer
    features_df = df[(df['InvoiceDate'] >= feature_start_date) & (df['InvoiceDate'] <= feature_end_date)]
    customer_features = features_df.groupby('CustomerID').agg(
        last_30_days_total_amount=('Amount', 'sum'),
        last_30_days_order_count=('InvoiceNo', 'nunique'),
        last_30_days_item_count=('Quantity', 'sum')
    ).reset_index()

    # Target: aggregate next 30 days purchases for each customer
    target_df = df[(df['InvoiceDate'] >= target_start_date) & (df['InvoiceDate'] <= target_end_date)]
    customer_target = target_df.groupby('CustomerID').agg(
        next_30_days_total_amount=('Amount', 'sum')
    ).reset_index()

    # Merge features and target
    # Use outer merge to keep all customers, then fill NaNs for those who didn't purchase in a window
    # It's important to merge on all unique CustomerIDs present in either feature or target window for this prediction_date
    all_customer_ids = pd.concat([customer_features['CustomerID'], customer_target['CustomerID']]).unique()

    # Create a base dataframe with all relevant CustomerIDs for this prediction_date
    base_customers_df = pd.DataFrame({'CustomerID': all_customer_ids})

    merged_df = pd.merge(base_customers_df, customer_features, on='CustomerID', how='left')
    merged_df = pd.merge(merged_df, customer_target, on='CustomerID', how='left')

    # Fill NaN targets with 0 (customers who didn't purchase in the next 30 days)
    merged_df['next_30_days_total_amount'] = merged_df['next_30_days_total_amount'].fillna(0)

    # Fill NaN features with 0 (customers who didn't purchase in the past 30 days)
    merged_df = merged_df.fillna(0)

    return merged_df

def main():
    print("Loading data...")
    df = pd.read_csv('data.csv', encoding='latin1')
    print("Data loaded.")

    print("Preprocessing data...")
    df = preprocess_data(df)
    print("Data preprocessed.")

    min_date = df['InvoiceDate'].min()
    max_date = df['InvoiceDate'].max()
    print(f"Date range: {min_date} to {max_date}")

    # Define a cutoff date for training and validation to avoid data leakage.
    # The validation set will predict for dates *after* this cutoff.
    # Let's use 80% of the time range for training, 20% for validation.
    total_days = (max_date - min_date).days

    # Ensure train_end_date_for_generating_samples allows enough time for both feature and target windows
    # min_date + 30 days (features) + 30 days (target) = min_date + 60 days is the earliest actual prediction for a sample
    # So, ensure the split point is after this.

    # Let's use the last 3 months of data for validation.
    # So, training prediction dates will be before `max_date - 3 months`
    # Validation prediction dates will be after `max_date - 3 months`

    validation_start_prediction_date = max_date - pd.Timedelta(days=90) # approx 3 months before max date

    # Generate samples for training and validation
    all_data_points = []

    # Earliest possible prediction date that allows for a full 30-day feature window
    # and a full 30-day target window.
    # min_date + 30 days (features) -> then next day is prediction date.
    # Prediction date itself needs 30 days for target. So, min_date + 30 days (features) + 30 days (target)
    current_prediction_date = min_date + pd.Timedelta(days=60) # Start generating samples from here

    # Ensure that current_prediction_date + 29 days (for target window) does not exceed max_date
    while current_prediction_date + pd.Timedelta(days=29) <= max_date:
        # print(f"Creating features for prediction date: {current_prediction_date}") # Too verbose
        features_target = create_features_and_target(df, current_prediction_date)

        # Add the prediction date to the features_target df for splitting
        features_target['prediction_date'] = current_prediction_date
        all_data_points.append(features_target)

        current_prediction_date += pd.Timedelta(days=7) # Move prediction window by 7 days

    full_dataset = pd.concat(all_data_points, ignore_index=True)

    # Split the data into training and validation sets based on prediction_date
    train_df = full_dataset[full_dataset['prediction_date'] < validation_start_prediction_date]
    val_df = full_dataset[full_dataset['prediction_date'] >= validation_start_prediction_date]

    # Drop prediction_date and CustomerID as they are not features for the model
    X_train = train_df.drop(columns=['CustomerID', 'next_30_days_total_amount', 'prediction_date'])
    y_train = train_df['next_30_days_total_amount']

    X_val = val_df.drop(columns=['CustomerID', 'next_30_days_total_amount', 'prediction_date'])
    y_val = val_df['next_30_days_total_amount']

    print(f"Total generated samples: {len(full_dataset)}")
    print(f"Training data points (unique prediction dates): {train_df['prediction_date'].nunique()}")
    print(f"Validation data points (unique prediction dates): {val_df['prediction_date'].nunique()}")

    if len(X_train) == 0 or len(X_val) == 0:
        print("Not enough data to create proper training/validation sets with the chosen split. Consider adjusting validation_start_prediction_date or the overall date range.")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        return

    print("Training model...")
    # Using n_jobs=-1 to utilize all available CPU cores for faster training
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model trained.")

    print("Evaluating model...")
    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"Mean Absolute Error on validation set: {mae:.2f}")
    print(f"R^2 Score on validation set: {r2:.2f}")

    # --- Inference Example ---
    # To demonstrate inference, let's pick a date just after our validation period
    # This simulates making a prediction for a future period.
    inference_prediction_date = max_date + pd.Timedelta(days=1) # The day after the last available data

    # Ensure this inference date allows for a full 30-day feature window
    # If max_date is Dec 9, 2011, and inference_prediction_date is Dec 10, 2011.
    # The feature window would be Nov 10, 2011 - Dec 9, 2011. This uses all available data.

    # For target window, we assume we don't have this data yet, as it's future.
    # create_features_and_target can still be used, but the target part will be empty (all 0s), which is fine for feature extraction.

    print(f"\n--- Performing Inference for a future prediction date: {inference_prediction_date} ---")

    # Re-create features for this specific prediction date using the full original dataframe
    # The `create_features_and_target` function will correctly extract features based on `inference_prediction_date - 30 days` to `inference_prediction_date - 1 day`.
    inference_features_df = create_features_and_target(df, inference_prediction_date)

    # We need to ensure the columns match X_train, so drop CustomerID and the target
    # The target column will be all 0s for this inference_prediction_date, which is correct
    X_inference = inference_features_df.drop(columns=['CustomerID', 'next_30_days_total_amount'])

    # Predict
    if not X_inference.empty:
        inference_predictions = model.predict(X_inference)

        # Map predictions back to CustomerIDs
        inference_results = pd.DataFrame({
            'CustomerID': inference_features_df['CustomerID'],
            'Predicted_Next_30_Days_Amount': inference_predictions
        })

        print("\nTop 5 customers with highest predicted purchase amount for the next 30 days:")
        print(inference_results.sort_values(by='Predicted_Next_30_Days_Amount', ascending=False).head())

        print("\nBottom 5 customers with lowest (or zero) predicted purchase amount for the next 30 days:")
        print(inference_results.sort_values(by='Predicted_Next_30_Days_Amount', ascending=True).head())
    else:
        print("No customers found for inference features. Cannot perform inference.")

if __name__ == "__main__":
    main()
```
:::

**Evaluation:**
- The model uses Random Forest (I wanted LightGBM).
- Evaluation metrics are MAE and R2 (I wanted RMSE).
- Validation uses simple holdout (I wanted cross-validation).
- In feature creation, it filters records where `Quantity` and `UnitPrice` are greater than 0. While this could be considered thoughtful, ideally these decisions about data specifications and business requirements should be confirmed rather than made arbitrarily.

### Improving Generalization Performance (Cross-validation, Feature Addition, etc.)

To improve generalization performance, I'll instruct it to change to cross-validation.

```docker
I want to improve generalization performance. Currently using simple holdout, but please change to cross-validation.
As before, please be very careful about data leakage.
```

Next, I'll add instructions to change the evaluation metric and modify the validation period interval.

```bash
- Add RMSE to the evaluation metrics
- Current_prediction_date is in 7-day intervals, but change it to 30-day intervals
```

This changes the dataset generation interval to 30 days and outputs RMSE. However, such parameter changes would typically be faster to modify myself rather than giving instructions.

Next, I'll change the model to LightGBM.

```docker
I want to change the model to LightGBM to improve accuracy.
Please use the original LightGBM Python API, not the scikit-learn wrapper.
```

To improve interpretability of predictions, I'll have it output feature importance.

```python
I want to make predictions more interpretable, so please output importance
```

After outputting, I noticed only three features were being used.
To improve accuracy, I'll instruct it to add specific features.

```docker
To further improve accuracy, let's try adding features.
Please add the following features:
- Record count
- nunique of InvoiceNo
- nunique of StockCode
- nunique of InvoiceDate
- mode and nunique of country
- sum, max, min, median of historical purchase amounts
```

In response to this instruction, it initially added cumulative aggregations without a fixed period, requiring several iterations of minor corrections.
For detailed logic like aggregation, I felt it was more reliable to either write it myself or provide detailed specifications upfront rather than instructing verbally.

:::details Excerpt from the Correction Process
```python
# Example where aggregation window wasn't fixed (suddenly added 180-day aggregation when previously using 30-day windows)
+     long_window_features = long_window_df.groupby('CustomerID').agg(
+         last_180_days_record_count=('InvoiceNo', 'count'),
... # (rest of the code)
+         last_180_days_amount_median=('Amount', 'median')
+     ).reset_index()
```
:::


### Final Code

The code completed through continued conversation is as follows. At this point, what I wanted to achieve (the detailed requirements from the previous article) is mostly included, but there are several places where it's redundantly verbose or settings need minor adjustments. Editing through conversation alone felt tedious. Even when primarily coding through conversation, I'd want to use direct editing in parallel. (Though this is obvious...)

:::details Final Code (When You Know What You Want)
```python
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

def preprocess_data(df):
    df = df.copy()  # Operate on a copy to avoid SettingWithCopyWarning
    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Calculate Amount
    df["Amount"] = df["Quantity"] * df["UnitPrice"]

    # Drop rows with missing CustomerID
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Filter out returns (negative Quantity) and invalid prices/quantities
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Convert Country to category for LightGBM
    df["Country"] = df["Country"].astype("category")

    # Sort by CustomerID and InvoiceDate for time-series operations
    df = df.sort_values(by=["CustomerID", "InvoiceDate"])

    return df

def create_features_and_target(df, prediction_date):
    # Data for short feature window (past 30 days relative to prediction_date)
    short_window_days = 30
    feature_start_date = prediction_date - pd.Timedelta(days=short_window_days)
    feature_end_date = prediction_date - pd.Timedelta(days=1)

    # Data for target window (next 30 days from prediction_date)
    target_start_date = prediction_date
    target_end_date = prediction_date + pd.Timedelta(days=29)

    # Define a helper for mode that handles empty/multiple values
    def get_mode(x):
        m = x.mode()
        if not m.empty:
            return m.iloc[0]
        return None

    # All Features are now Short Window (Past 30 Days)
    features_df = df[
        (df["InvoiceDate"] >= feature_start_date)
        & (df["InvoiceDate"] <= feature_end_date)
    ]
    customer_features = (
        features_df.groupby("CustomerID")
        .agg(
            # Existing 30-day features
            last_30_days_total_amount=("Amount", "sum"),
            last_30_days_order_count=("InvoiceNo", "nunique"),
            last_30_days_item_count=("Quantity", "sum"),
            # Newly requested features, now also for last 30 days
            last_30_days_record_count=("InvoiceNo", "count"),
            last_30_days_stock_nunique=("StockCode", "nunique"),
            last_30_days_invoicedate_nunique=("InvoiceDate", "nunique"),
            last_30_days_country_mode=("Country", get_mode),
            last_30_days_country_nunique=("Country", "nunique"),
            last_30_days_amount_max=("Amount", "max"),
            last_30_days_amount_min=("Amount", "min"),
            last_30_days_amount_median=("Amount", "median"),
        )
        .reset_index()
    )

    # Target: aggregate next 30 days purchases for each customer
    target_df = df[
        (df["InvoiceDate"] >= target_start_date)
        & (df["InvoiceDate"] <= target_end_date)
    ]
    customer_target = (
        target_df.groupby("CustomerID")
        .agg(next_30_days_total_amount=("Amount", "sum"))
        .reset_index()
    )

    # Merge features and target
    # Base customers should ideally be those active in the feature window or target window
    all_customer_ids = pd.concat(
        [customer_features["CustomerID"], customer_target["CustomerID"]]
    ).unique()

    # Create a base dataframe with all relevant CustomerIDs for this prediction_date
    base_customers_df = pd.DataFrame({"CustomerID": all_customer_ids})

    merged_df = pd.merge(
        base_customers_df, customer_features, on="CustomerID", how="left"
    )
    merged_df = pd.merge(merged_df, customer_target, on="CustomerID", how="left")

    # Fill NaN targets with 0
    merged_df["next_30_days_total_amount"] = merged_df[
        "next_30_days_total_amount"
    ].fillna(0)

    # Fill NaN features
    # Numeric features: fill with 0
    numeric_cols = [
        "last_30_days_total_amount",
        "last_30_days_order_count",
        "last_30_days_item_count",
        "last_30_days_record_count",
        "last_30_days_stock_nunique",
        "last_30_days_invoicedate_nunique",
        "last_30_days_country_nunique",
        "last_30_days_amount_max",
        "last_30_days_amount_min",
        "last_30_days_amount_median",
    ]
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)

    # Categorical feature (Country mode): Ensure dtype is category
    merged_df["last_30_days_country_mode"] = merged_df[
        "last_30_days_country_mode"
    ].astype("category")

    return merged_df

def main():
    print("Loading data...")
    df = pd.read_csv("data.csv", encoding="latin1")
    print("Data loaded.")

    print("Preprocessing data...")
    df = preprocess_data(df)
    print("Data preprocessed.")

    min_date = df["InvoiceDate"].min()
    max_date = df["InvoiceDate"].max()
    print(f"Date range: {min_date} to {max_date}")

    total_days = (max_date - min_date).days

    # Generate samples for training and validation
    all_data_points = []

    current_prediction_date = min_date + pd.Timedelta(
        days=60
    )  # Start generating samples from here

    # Ensure that current_prediction_date + 29 days (for target window) does not exceed max_date
    while current_prediction_date + pd.Timedelta(days=29) <= max_date:
        features_target = create_features_and_target(df, current_prediction_date)

        # Add the prediction date to the features_target df for splitting
        features_target["prediction_date"] = current_prediction_date
        all_data_points.append(features_target)

        current_prediction_date += pd.Timedelta(
            days=30
        )  # Move prediction window by 30 days

    full_dataset = pd.concat(all_data_points, ignore_index=True)

    # Sort by prediction date to ensure correct time-based splitting logic
    full_dataset.sort_values("prediction_date", inplace=True)

    unique_dates = np.sort(full_dataset["prediction_date"].unique())
    print(f"Total generated samples: {len(full_dataset)}")
    print(f"Unique prediction dates: {len(unique_dates)}")

    # Time Series Cross-Validation
    n_splits = 4
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_mae_scores = []
    cv_r2_scores = []
    cv_rmse_scores = []
    best_iterations = []

    print(f"\nStarting Time Series Cross-Validation with {n_splits} splits...")

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": 42,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
    }

    fold = 0
    for train_index, val_index in tscv.split(unique_dates):
        fold += 1
        val_dates = unique_dates[val_index]
        potential_train_dates = unique_dates[train_index]

        # LEAKAGE PREVENTION:
        min_val_date = val_dates.min()
        valid_train_dates = [
            d
            for d in potential_train_dates
            if d + pd.Timedelta(days=30) <= min_val_date
        ]

        if not valid_train_dates:
            print(
                f"Fold {fold}: Skipped - Not enough history for training with strict leakage prevention gap."
            )
            continue

        train_df = full_dataset[full_dataset["prediction_date"].isin(valid_train_dates)]
        val_df = full_dataset[full_dataset["prediction_date"].isin(val_dates)]

        X_train = train_df.drop(
            columns=["CustomerID", "next_30_days_total_amount", "prediction_date"]
        )
        y_train = train_df["next_30_days_total_amount"]

        X_val = val_df.drop(
            columns=["CustomerID", "next_30_days_total_amount", "prediction_date"]
        )
        y_val = val_df["next_30_days_total_amount"]

        # Create LightGBM datasets
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Train model
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0), # period=0 disables verbose logging
            ],
        )

        best_iterations.append(model.best_iteration)

        y_pred = model.predict(X_val, num_iteration=model.best_iteration)

        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)

        cv_mae_scores.append(mae)
        cv_r2_scores.append(r2)
        cv_rmse_scores.append(rmse)

        print(
            f"Fold {fold}: Train Dates: {len(valid_train_dates)}, Val Dates: {len(val_dates)} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}, Best Iter: {model.best_iteration}"
        )

    if cv_mae_scores:
        print(
            f"\nAverage MAE across {len(cv_mae_scores)} folds: {np.mean(cv_mae_scores):.2f}"
        )
        print(
            f"Average RMSE across {len(cv_rmse_scores)} folds: {np.mean(cv_rmse_scores):.2f}"
        )
        print(
            f"Average R2 across {len(cv_mae_scores)} folds: {np.mean(cv_r2_scores):.2f}"
        )
        avg_best_iter = int(np.mean(best_iterations))
        print(f"Average best iteration: {avg_best_iter}")
    else:
        print("\nNo valid folds found for cross-validation.")
        return

    # --- Final Retraining for Inference ---
    print("\nRetraining model on all available data for future inference...")
    X_final_train = full_dataset.drop(
        columns=["CustomerID", "next_30_days_total_amount", "prediction_date"]
    )
    y_final_train = full_dataset["next_30_days_total_amount"]

    dtrain_final = lgb.Dataset(X_final_train, label=y_final_train)

    # Train with the average best iteration found in CV
    model_final = lgb.train(
        params,
        dtrain_final,
        num_boost_round=avg_best_iter,
        valid_sets=[dtrain_final],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    print("Final model trained.")

    # --- Feature Importance ---
    print("\n--- Feature Importances ---")
    feature_importances = model_final.feature_importance()
    feature_names = model_final.feature_name()

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    )
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    print("Top 20 most important features:")
    print(importance_df.head(20))

    # --- Inference Example ---
    inference_prediction_date = max_date + pd.Timedelta(days=1)
    print(
        f"\n--- Performing Inference for a future prediction date: {inference_prediction_date} ---"
    )

    inference_features_df = create_features_and_target(df, inference_prediction_date)

    X_inference = inference_features_df.drop(
        columns=["CustomerID", "next_30_days_total_amount"]
    )

    if not X_inference.empty:
        inference_predictions = model_final.predict(
            X_inference, num_iteration=model_final.best_iteration
        )

        inference_results = pd.DataFrame(
            {
                "CustomerID": inference_features_df["CustomerID"],
                "Predicted_Next_30_Days_Amount": inference_predictions,
            }
        )

        print(
            "\nTop 5 customers with highest predicted purchase amount for the next 30 days:"
        )
        print(
            inference_results.sort_values(
                by="Predicted_Next_30_Days_Amount", ascending=False
            ).head()
        )

        print(
            "\nBottom 5 customers with lowest (or zero) predicted purchase amount for the next 30 days:"
        )
        print(
            inference_results.sort_values(
                by="Predicted_Next_30_Days_Amount", ascending=True
            ).head()
        )
    else:
        print("No customers found for inference features. Cannot perform inference.")

if __name__ == "__main__":
    main()
```
:::

### Reflections

- For minor modifications like parameter changes, directly editing the code is overwhelmingly faster.
- I found it efficient to delegate changes like implementing cross-validation, which are clear what needs to be done but involve a fair amount of implementation work.
- Arbitrary interpretations can creep in where specifications are ambiguous, so careful attention is needed where clear instructions aren't provided.
- Review is necessary every time code is generated, but frequent reviewing is mentally taxing. Both finding changes and the review itself felt burdensome. While UI improvements might help with the former, the latter will likely remain an issue requiring good countermeasures or adaptation.
- Additionally, as I continued modifying through conversation, the breadth of LLM responses seemed to narrow. Periodically saving conversation content or latest specifications to reset context and thinking appears necessary.

Overall, code almost never completely failed to run during testing, which subjectively feels quite good. However, I found the approach of gradually improving code through conversation burdensome. The method from the previous article of generating from comprehensive requirements upfront and reducing review frequency seems better suited to me. When what you want to achieve is clear, preparing detailed instructions upfront is more efficient with fewer conversational back-and-forth.

As coding assistance tools and LLMs continue to improve, I'll keep exploring better usage patterns and approaches.


## 2. When Exploring Through Discussion

Next, let's examine a pattern where I don't have specific improvement ideas and consult with the LLM even on implementation approaches.
In the context of model training and inference, I don't personally use this method much, but this approach might be used when implementing in unfamiliar languages or domains (like application development in other fields for me), so I'll check it for reference.
Here, I'll proceed basically trusting what the LLM says.

### Initial Generation and Preprocessing Consultation

First things first, let's generate initial code.

```
First, please check @05_wallball_requirements.txt.
Based on this document, implement model training and inference code.
```
:::details Generated Code
# /// script
# dependencies = ["pandas", "scikit-learn", "numpy"]
# ///
"""
Purchase Amount Prediction Model

This script implements a machine learning pipeline to predict the purchase amount
of customers for the next 30 days based on their activity in the past 30 days.

Key Features:
- Data: 'data.csv' (Online Retail Dataset)
- Preprocessing: Cleaning missing IDs, date conversion, total amount calculation.
- Feature Engineering:
    - Aggregated by CustomerID.
    - Window: [Prediction Date - 30 days, Prediction Date).
    - Features: Monetary (Sum), Frequency (Count), Recency (Days since last purchase).
- Target:
    - Window: [Prediction Date, Prediction Date + 30 days).
    - Value: Sum of purchase amount.
- Validation:
    - Time-series split to avoid leakage.
    - Train data: Multiple 30-day windows sliding back from the test period.
    - Test data: The most recent 30-day window with available ground truth.
- Model: Random Forest Regressor.

Usage:
    uv run 05_wallball.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime

def load_and_preprocess(filepath):
    print("Loading data...")
    # Use specific encoding as detected in exploration
    df = pd.read_csv(filepath, encoding='ISO-8859-1')

    # Preprocessing
    print("Preprocessing...")
    # Drop rows with missing CustomerID
    df = df.dropna(subset=['CustomerID'])

    # Convert CustomerID to int (it's usually float when loaded with NaNs)
    df['CustomerID'] = df['CustomerID'].astype(int)

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Calculate TotalAmount
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    # Remove extremely weird dates if any (sanity check not strictly needed if pd.to_datetime works)
    return df

def get_features_and_target(df, cutoff_date, known_customers):
    """
    Generates features and target for a specific cutoff date.
    Features: 30 days before cutoff.
    Target: 30 days after cutoff.
    """
    feature_start = cutoff_date - datetime.timedelta(days=30)
    target_end = cutoff_date + datetime.timedelta(days=30)

    # Data for features
    df_features = df[(df['InvoiceDate'] >= feature_start) & (df['InvoiceDate'] < cutoff_date)]

    # Data for target
    df_target = df[(df['InvoiceDate'] >= cutoff_date) & (df['InvoiceDate'] < target_end)]

    # Initialize result dataframe with all known customers
    # This ensures we predict for inactive customers too (who might purchase 0)
    res = pd.DataFrame({'CustomerID': list(known_customers)})

    # --- Feature Engineering ---
    # Aggregate feature data
    # 1. Monetary: Sum of TotalAmount
    # 2. Frequency: Count of unique InvoiceNo
    # 3. Recency: Days since last purchase (relative to cutoff)

    agg_dict = {
        'TotalAmount': 'sum',
        'InvoiceNo': 'nunique',
        'InvoiceDate': 'max'
    }

    feat_agg = df_features.groupby('CustomerID').agg(agg_dict).reset_index()
    feat_agg.rename(columns={
        'TotalAmount': 'history_monetary',
        'InvoiceNo': 'history_frequency',
        'InvoiceDate': 'last_purchase_date'
    }, inplace=True)

    # Merge features
    res = res.merge(feat_agg, on='CustomerID', how='left')

    # Fill NaNs for customers with no activity in feature window
    res['history_monetary'] = res['history_monetary'].fillna(0)
    res['history_frequency'] = res['history_frequency'].fillna(0)

    # Calculate Recency
    # If no purchase in window, we can set it to 30 (max window size) or a distinct value
    # Since we only look at past 30 days, if last_purchase_date is null, it means > 30 days.
    res['history_recency'] = (cutoff_date - res['last_purchase_date']).dt.days
    res['history_recency'] = res['history_recency'].fillna(30) # Fill missing with window size

    # Drop helper column
    res = res.drop(columns=['last_purchase_date'])

    # --- Target Generation ---
    target_agg = df_target.groupby('CustomerID')['TotalAmount'].sum().reset_index()
    target_agg.rename(columns={'TotalAmount': 'target_amount'}, inplace=True)

    # Merge target
    res = res.merge(target_agg, on='CustomerID', how='left')
    res['target_amount'] = res['target_amount'].fillna(0)

    return res

def main():
    filepath = 'data.csv'
    df = load_and_preprocess(filepath)

    print(f"Data range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

    max_date = df['InvoiceDate'].max()

    # Define Test Cutoff (Leave 30 days for target)
    test_cutoff = max_date - datetime.timedelta(days=30)
    # Adjust to start of day to be clean? Or keep exact time?
    # Let's align cutoffs to 00:00:00 for consistency
    test_cutoff = test_cutoff.replace(hour=0, minute=0, second=0, microsecond=0)

    print(f"Test Cutoff: {test_cutoff}")

    # Define Training Cutoffs
    # We'll create a few slices for training to capture seasonality/trends
    # e.g., monthly back from test_cutoff
    train_cutoffs = []
    current_cutoff = test_cutoff - datetime.timedelta(days=30)
    for _ in range(6): # Take 6 months of history for training samples
        if current_cutoff < df['InvoiceDate'].min() + datetime.timedelta(days=30):
            break
        train_cutoffs.append(current_cutoff)
        current_cutoff -= datetime.timedelta(days=30)

    print(f"Training Cutoffs: {train_cutoffs}")

    train_dfs = []

    # Collect all customers known before the split to avoid lookahead bias in "who exists"
    # But simplistic approach: "All customers seen in dataset up to cutoff"

    for cutoff in train_cutoffs:
        print(f"Generating training data for cutoff: {cutoff}")
        # Customers known up to this cutoff
        known_mask = df['InvoiceDate'] < cutoff
        known_customers = df.loc[known_mask, 'CustomerID'].unique()

        train_chunk = get_features_and_target(df, cutoff, known_customers)
        train_dfs.append(train_chunk)

    train_data = pd.concat(train_dfs, ignore_index=True)

    print(f"Generating test data for cutoff: {test_cutoff}")
    known_customers_test = df.loc[df['InvoiceDate'] < test_cutoff, 'CustomerID'].unique()
    test_data = get_features_and_target(df, test_cutoff, known_customers_test)

    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")

    # Features and Target
    features = ['history_monetary', 'history_frequency', 'history_recency']
    target = 'target_amount'

    X_train = train_data[features]
    y_train = train_data[target]

    X_test = test_data[features]
    y_test = test_data[target]

    # Model Training
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Prediction
    preds = rf.predict(X_test)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print("--- Evaluation Results ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Feature Importance
    importances = pd.Series(rf.feature_importances_, index=features)
    print("\nFeature Importances:")
    print(importances)

    # Optional: Save predictions
    # output = pd.DataFrame({'CustomerID': test_data['CustomerID'], 'Actual': y_test, 'Predicted': preds})
    # output.to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    main()
:::

This code is essentially equivalent to the single-shot generation with only overview instructions, so I'll continue by consulting about data preprocessing.

```python
I want to perform preprocessing based on data content and characteristics. What should I check to consider necessary preprocessing?
```

:::details Response
✦ After examining the data, here's what we should consider and implement for preprocessing:

  1. Excluding Missing CustomerID
   * Current status: 135,080 records (about 25% of total) have missing CustomerID.
   * Action: Given our objective of predicting customer purchases, records without customer IDs cannot be linked, so they should be excluded.

  2. Handling Cancellation Data (Negative Quantity / InvoiceNo 'C')
   * Current status: 10,624 records have negative Quantity. Also, 9,288 records have InvoiceNo starting with 'C' (cancellations).
   * Action:
       * For simple prediction models, cancellations (negative sales) tend to be noise, so excluding them is standard.
       * However, they could be used as features like "customers with many returns have lower future purchase amounts," but for this baseline we recommend excluding cancellation data (Quantity < 0).

  3. Excluding Special Transactions (StockCode / UnitPrice)
   * StockCode: POST (shipping), D (Discount), BANK CHARGES, AMAZONFEE
     and other special codes were found. Since these differ from product purchasing behavior, they should be excluded for pure product purchase prediction.
   * UnitPrice: 2,517 records have prices ≤ 0, with many having missing Description
     as well. These are likely not valid purchase records, so should be excluded.

  4. Removing Duplicate Data
   * Current status: 5,268 completely duplicate rows exist.
   * Action: Remove them as they distort aggregation results.

  Recommended Preprocessing Code Changes

  I recommend adding these processes to the load_and_preprocess function in 05_wallball.py.

    1     # ... (after loading) ...
    2
    3     # 1. Remove duplicates
    4     df = df.drop_duplicates()
    5
    6     # 2. Remove missing CustomerID
    7     df = df.dropna(subset=['CustomerID'])
    8     df['CustomerID'] = df['CustomerID'].astype(int)
    9
   10     # 3. Exclude cancellations/returns (Quantity > 0)
   11     # Note: If you want to subtract cancellations from sales, keep them,
   12     # but we'll focus on "purchase behavior" and exclude them.
   13     df = df[df['Quantity'] > 0]
   14
   15     # 4. Exclude special StockCodes (exclude items not starting with numbers, etc.)
   16     # POST (shipping) etc. are safer to exclude
   17     # Simply keep only those starting with numbers, or exclude specific list
   18     # Here we exclude 'POST', 'D', 'M', 'C2', 'DOT', 'BANK CHARGES', etc.
   19     special_codes = ['POST', 'D', 'M', 'C2', 'DOT', 'BANK CHARGES', 'CRUK']
   20     df = df[~df['StockCode'].isin(special_codes)]
   21
   22     # 5. Use only data with UnitPrice > 0
   23     df = df[df['UnitPrice'] > 0]
   24
   25     # ... (below, date conversion, etc.) ...

  Should I modify the code to add this preprocessing?
:::

While some parts make implementation decisions on ambiguous specifications, I proceeded with the implementation as-is.

### Consulting on Accuracy Improvement

Next, I'll consult about improving model accuracy.

```python
What improvements are needed to enhance generalization performance?
```

:::details Response
✦ To improve generalization performance (prediction accuracy on unseen data), we need to help the model better capture the "underlying patterns" in the data while preventing "overfitting to training data."

  For the current baseline model (only RFM for past 30 days), the following improvements are possible:

  1. Feature Engineering Enhancement
  Currently only Recency, Frequency, Monetary (past 30 days), but we should add features that capture customer behavior from multiple angles.
   * Long-term behavior history:
     Not just the past 30 days, but aggregate for 90 days, 180 days, or entire period. This captures customers like "haven't bought in past 30 days but were frequent buyers in previous 6 months."
   * Trend: Calculate ratios like (last 30 days purchase amount) / (average of past 90 days purchase amount)
     to add trend information about whether purchases are increasing or decreasing.
   * Product diversity: Number of unique products (StockCode) purchased.
   * Customer tenure: Days from first purchase to present. Distinguishes long-time vs new customers.
   * Average unit price/basket size: Average purchase amount or average items per transaction.

  2. Target Variable Transformation (Log Transformation)
  Sales data often follows a "power law," where a few "super valuable customers (outliers)" greatly skew overall error (RMSE).
   * Train with target variable (purchase amount) log-transformed as log(x + 1), then convert back with exp(x) - 1
     after prediction. This suppresses outlier effects and improves model stability.

  3. Model Change
   * Random Forest is powerful, but for tabular data, Gradient Boosting Decision Trees (GBDT) (e.g.,
     scikit-learn's HistGradientBoostingRegressor, XGBoost, LightGBM, etc.)
     tend to achieve higher accuracy.

  4. Handling Zero-Inflation
   * Many customers will "not purchase (0 yen)" in the next 30 days.
       * Two-stage model:
         Separating "will they buy or not (classification)" × "how much will they buy (regression)" is possible, but log transformation with GBDT is an easier and effective first approach.

  ---

  Among these, the most effective and immediately implementable are "adding features (period expansion, Tenure)" and "log transformation." Should we implement these and verify the effect?
:::

Based on this response, I first had it implement feature addition and log transformation. This improved MAE.
However, ideally we should first implement a proper validation method (like cross-validation) to consistently evaluate the accuracy impact of various approaches according to our objectives.
The process continued with changes to GBDT as Gemini suggested, then adding hyperparameter tuning and a zero-inflated model (predicting purchase occurrence first).

:::details Generated Code (GBDT + Zero-Inflated Model)
```python
# /// script
# dependencies = ["pandas", "scikit-learn", "numpy"]
# ///
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.inspection import permutation_importance
import datetime

def load_and_preprocess(filepath):
    print("Loading data...")
    # Use specific encoding as detected in exploration
    df = pd.read_csv(filepath, encoding='ISO-8859-1')

    # Preprocessing
    print("Preprocessing...")

    # 1. Drop duplicates
    df.drop_duplicates(inplace=True)

    # 2. Drop rows with missing CustomerID and convert to int
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)

    # 3. Exclude cancelled orders (Quantity > 0)
    df = df[df['Quantity'] > 0]

    # 4. Exclude special StockCodes that are not actual products
    special_codes = ['POST', 'D', 'M', 'C2', 'DOT', 'BANK CHARGES', 'CRUK', 'AMAZONFEE']
    df = df[~df['StockCode'].isin(special_codes)]

    # 5. Exclude rows with zero or negative UnitPrice
    df = df[df['UnitPrice'] > 0]

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Calculate TotalAmount
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    return df

def get_features_and_target(df, cutoff_date, known_customers):
    """
    Generates features and target for a specific cutoff date.
    Target: 30 days after cutoff.
    Features:
        - 30 days window
        - 90 days window
        - 180 days window
        - Global: Recency, Tenure
    """

    # Initialize result dataframe
    res = pd.DataFrame({'CustomerID': list(known_customers)})

    # --- Feature Engineering ---

    # 1. Global Features (Recency, Tenure)
    # Filter data strictly BEFORE cutoff
    df_hist = df[df['InvoiceDate'] < cutoff_date]

    # Last and First purchase dates
    global_agg = df_hist.groupby('CustomerID')['InvoiceDate'].agg(['max', 'min']).reset_index()
    global_agg.rename(columns={'max': 'last_purchase_date', 'min': 'first_purchase_date'}, inplace=True)

    res = res.merge(global_agg, on='CustomerID', how='left')

    # Recency: Days since last purchase
    # Tenure: Days since first purchase
    res['recency'] = (cutoff_date - res['last_purchase_date']).dt.days
    res['tenure'] = (cutoff_date - res['first_purchase_date']).dt.days

    # Fill missing (inactive customers? technically known_customers should be in df_hist but safe to fill)
    res['recency'] = res['recency'].fillna(365)
    res['tenure'] = res['tenure'].fillna(0)

    # Drop date columns
    res = res.drop(columns=['last_purchase_date', 'first_purchase_date'])

    # 2. Windowed Features (30, 90, 180 days)
    windows = [30, 90, 180]

    for days in windows:
        start_date = cutoff_date - datetime.timedelta(days=days)
        df_window = df[(df['InvoiceDate'] >= start_date) & (df['InvoiceDate'] < cutoff_date)]

        agg_dict = {
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique',
            'StockCode': 'nunique'
        }

        win_agg = df_window.groupby('CustomerID').agg(agg_dict).reset_index()

        # Rename with suffix
        suffix = f"{days}d"
        win_agg.rename(columns={
            'TotalAmount': f'monetary{suffix}',
            'InvoiceNo': f'frequency{suffix}',
            'StockCode': f'unique_items{suffix}'
        }, inplace=True)

        res = res.merge(win_agg, on='CustomerID', how='left')

        # Fill NaNs with 0
        cols = [f'monetary{suffix}', f'frequency{suffix}', f'unique_items{suffix}']
        res[cols] = res[cols].fillna(0)

    # --- Target Generation ---
    target_end = cutoff_date + datetime.timedelta(days=30)
    df_target = df[(df['InvoiceDate'] >= cutoff_date) & (df['InvoiceDate'] < target_end)]

    target_agg = df_target.groupby('CustomerID')['TotalAmount'].sum().reset_index()
    target_agg.rename(columns={'TotalAmount': 'target_amount'}, inplace=True)

    # Merge target
    res = res.merge(target_agg, on='CustomerID', how='left')
    res['target_amount'] = res['target_amount'].fillna(0)

    return res

def main():
    filepath = 'data.csv'
    df = load_and_preprocess(filepath)

    print(f"Data range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

    max_date = df['InvoiceDate'].max()

    # Define Test Cutoff (Leave 30 days for target)
    test_cutoff = max_date - datetime.timedelta(days=30)
    test_cutoff = test_cutoff.replace(hour=0, minute=0, second=0, microsecond=0)

    print(f"Test Cutoff: {test_cutoff}")

    # Define Training Cutoffs (Multiple monthly slices)
    train_cutoffs = []
    current_cutoff = test_cutoff - datetime.timedelta(days=30)
    for _ in range(6):
        if current_cutoff < df['InvoiceDate'].min() + datetime.timedelta(days=30):
            break
        train_cutoffs.append(current_cutoff)
        current_cutoff -= datetime.timedelta(days=30)

    print(f"Training Cutoffs: {train_cutoffs}")

    train_dfs = []

    for cutoff in train_cutoffs:
        print(f"Generating training data for cutoff: {cutoff}")
        known_mask = df['InvoiceDate'] < cutoff
        known_customers = df.loc[known_mask, 'CustomerID'].unique()

        train_chunk = get_features_and_target(df, cutoff, known_customers)
        train_dfs.append(train_chunk)

    train_data = pd.concat(train_dfs, ignore_index=True)

    print(f"Generating test data for cutoff: {test_cutoff}")
    known_customers_test = df.loc[df['InvoiceDate'] < test_cutoff, 'CustomerID'].unique()
    test_data = get_features_and_target(df, test_cutoff, known_customers_test)

    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")

    # Features and Target
    features = ['recency', 'tenure']
    for days in [30, 90, 180]:
        features.extend([f'monetary_{days}d', f'frequency_{days}d', f'unique_items_{days}d'])

    target = 'target_amount'

    X_train = train_data[features]
    y_train = train_data[target]

    X_test = test_data[features]
    y_test = test_data[target]

    # --- Two-Stage Model Implementation ---

    # Hyperparameter Search Space
    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_iter': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_leaf': [10, 20, 50],
        'l2_regularization': [0.0, 0.1, 1.0]
    }

    # Stage 1: Classification (Buy or Not)
    print("\n--- Stage 1: Classification Model (Purchase Probability) ---")
    y_train_class = (y_train > 0).astype(int)
    y_test_class = (y_test > 0).astype(int)

    print("Tuning Classifier...")
    clf = HistGradientBoostingClassifier(random_state=42)
    clf_search = RandomizedSearchCV(clf, param_dist, n_iter=10, cv=3, scoring='roc_auc', random_state=42, n_jobs=-1)
    clf_search.fit(X_train, y_train_class)

    best_clf = clf_search.best_estimator_
    print(f"Best Classifier Params: {clf_search.best_params_}")

    # Predict Probabilities
    prob_purchase_test = best_clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test_class, prob_purchase_test)
    print(f"Classifier ROC AUC: {auc:.4f}")

    # Stage 2: Regression (Amount | Buy)
    print("\n--- Stage 2: Regression Model (Conditional Amount) ---")
    # Filter training data to only include positive purchasers
    mask_positive = y_train > 0
    X_train_reg = X_train[mask_positive]
    y_train_reg = y_train[mask_positive]

    # Log transform target for regression
    y_train_reg_log = np.log1p(y_train_reg)

    print("Tuning Regressor...")
    reg = HistGradientBoostingRegressor(random_state=42)
    reg_search = RandomizedSearchCV(reg, param_dist, n_iter=10, cv=3, scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)
    reg_search.fit(X_train_reg, y_train_reg_log)

    best_reg = reg_search.best_estimator_
    print(f"Best Regressor Params: {reg_search.best_params_}")

    # Predict Amounts (for all test cases, assuming they buy)
    pred_log_amount_test = best_reg.predict(X_test)
    pred_amount_test = np.expm1(pred_log_amount_test)

    # --- Final Combination ---
    print("\n--- Final Prediction Combination ---")
    # Final Prediction = Probability * Estimated Amount
    final_preds = prob_purchase_test * pred_amount_test

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    mae = mean_absolute_error(y_test, final_preds)

    print("--- Evaluation Results (Two-Stage Model) ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Feature Importance (using the Classifier as it determines the zero-inflation mostly)
    # We could also look at Regressor importance
    print("\nCalculating Feature Importances (Classifier)...")
    result = permutation_importance(best_clf, X_test, y_test_class, n_repeats=5, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    importances = pd.Series(result.importances_mean[sorted_idx], index=np.array(features)[sorted_idx])
    print(importances.sort_values(ascending=False))

if __name__ == "__main__":
    main()
```
:::

As a result, RMSE improved significantly (965.4 -> 807.6). This is likely due to the classification model filtering out people with 0 yen purchases.

### Further Improvements and Specification Clarification

When I asked for further improvement suggestions, it proposed adding trend features and purchase interval statistics. However, as mentioned earlier, we need to first establish a proper validation method to compare the effectiveness of various approaches. Since it seemed likely to continue with the current simple holdout as-is, I wanted to reset Gemini by saving the conversation summary to a file.

To clarify changes and future direction, I had the AI summarize the dialogue content into a specification document (`05_spec.txt`) once, then asked for proposals based on that.

```
First, please check @05_spec.txt.
What do you think is needed to further improve generalization performance for this content?
Please tell me along with priorities.
```

:::details Response
```
✦ I've reviewed 05_spec.txt. Here are suggestions for improving generalization performance of the purchase prediction model, in priority order.

  ---

  Proposals for Improving Generalization Performance

  High Priority

   1. Strengthen Feature Engineering (Prioritize implementing unimplemented improvement ideas)
       * Content: Prioritize implementing "trend features" and "purchase interval statistics" listed as "future improvements" in the spec.
           * Trend features: Features capturing the rate of change in customers' recent vs long-term purchasing behavior (e.g., average purchase amount in past 30 days / average purchase amount in past 180 days).
           * Purchase interval statistics:
             Add customer purchase intervals (mean, standard deviation, median, etc.) as features to identify patterns like regular purchasers.
       * Reason:
         These features capture customer purchasing behavior patterns in more detail, enabling the model to more accurately predict individual customer future behavior even for unseen data, directly contributing to improved generalization performance.

   2. Introduce Time Series Cross-Validation
       * Content:
         Replace the current fixed test set evaluation with time series cross-validation. Split training and test data in chronological order and evaluate the model over multiple periods, enabling more reliable generalization performance assessment without overfitting to specific periods.
       * Reason:
         For time-series data, robust prediction performance independent of specific periods is important. Time series cross-validation is essential for objectively evaluating how stably the model functions on future data.

  Medium Priority

   3. Improve and Streamline Hyperparameter Search
       * Content: Consider introducing more efficient hyperparameter optimization methods beyond RandomizedSearchCV (e.g., Bayesian optimization libraries like Optuna
         or Hyperopt
         etc.). This is expected to find optimal hyperparameters in less time with limited computational resources, maximizing the model's potential.
       * Reason:
         Optimal hyperparameters significantly affect model prediction accuracy and generalization performance, so improving search quality is important.

   4. Error Analysis and Domain Knowledge-Based Feature Addition
       * Content:
         Identify customers or periods with large model prediction errors and analyze causes in detail. From those results, add new features reflecting business domain knowledge (e.g., seasonal events, major sales, new product launches, etc.) (e.g., category/NLP
         information, customer demographic information, if available).
       * Reason:
         Beyond purely data-driven approaches, incorporating business background into features enables the model to learn more complex real-world purchasing patterns, with expected improvements in both generalization performance and interpretability.

  Low Priority

   5. Consider Alternative Models and Ensemble Learning
       * Content: In addition to the current HistGradientBoosting model, try other powerful gradient boosting models (LightGBM, XGBoost
         etc.) and alternative models like Tweedie regression
         mentioned in the spec. Also consider introducing ensemble learning combining these multiple models.
       * Reason:
         Ensemble learning combining multiple model predictions can complement weaknesses of individual models and improve prediction stability and generalization performance. Specifically, Tweedie regression
         as an alternative to the Two-Stage
         model could potentially predict both zero and amount with a single model.

  ---

  These improvement measures are important not only for improving model accuracy but also for enhancing robustness to unseen data and evolving toward a more practical purchase prediction model.
```
:::

Finally, the idea of cross-validation emerged. I'll proceed to implement cross-validation and feature addition, judged as high priority.

### Final Code

This is the final code with cross-validation and additional features implemented.

:::details Final Code
```python
# /// script
# dependencies = ["pandas", "scikit-learn", "numpy"]
# ///
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.inspection import permutation_importance
import datetime

def load_and_preprocess(filepath):
    print("Loading data...")
    # Use specific encoding as detected in exploration
    df = pd.read_csv(filepath, encoding='ISO-8859-1')

    # Preprocessing
    print("Preprocessing...")

    # 1. Drop duplicates
    df.drop_duplicates(inplace=True)

    # 2. Drop rows with missing CustomerID and convert to int
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)

    # 3. Exclude cancelled orders (Quantity > 0)
    df = df[df['Quantity'] > 0]

    # 4. Exclude special StockCodes
    special_codes = ['POST', 'D', 'M', 'C2', 'DOT', 'BANK CHARGES', 'CRUK', 'AMAZONFEE']
    df = df[~df['StockCode'].isin(special_codes)]

    # 5. Exclude rows with zero or negative UnitPrice
    df = df[df['UnitPrice'] > 0]

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Calculate TotalAmount
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    return df

def get_features_and_target(df, cutoff_date, known_customers):
    """
    Generates features and target for a specific cutoff date.
    """
    res = pd.DataFrame({'CustomerID': list(known_customers)})

    # --- Feature Engineering ---

    # Filter history strictly BEFORE cutoff
    df_hist = df[df['InvoiceDate'] < cutoff_date].copy()

    # 1. Global Features (Recency, Tenure)
    global_agg = df_hist.groupby('CustomerID')['InvoiceDate'].agg(['max', 'min']).reset_index()
    global_agg.rename(columns={'max': 'last_purchase_date', 'min': 'first_purchase_date'}, inplace=True)

    res = res.merge(global_agg, on='CustomerID', how='left')

    res['recency'] = (cutoff_date - res['last_purchase_date']).dt.days
    res['tenure'] = (cutoff_date - res['first_purchase_date']).dt.days

    res['recency'] = res['recency'].fillna(365)
    res['tenure'] = res['tenure'].fillna(0)

    res = res.drop(columns=['last_purchase_date', 'first_purchase_date'])

    # --- NEW: Purchase Interval Statistics ---
    # Calculate intervals between unique InvoiceDates for each customer
    invoice_dates = df_hist[['CustomerID', 'InvoiceDate']].drop_duplicates().sort_values(['CustomerID', 'InvoiceDate'])
    invoice_dates['prev_date'] = invoice_dates.groupby('CustomerID')['InvoiceDate'].shift(1)
    invoice_dates['days_diff'] = (invoice_dates['InvoiceDate'] - invoice_dates['prev_date']).dt.days

    interval_stats = invoice_dates.groupby('CustomerID')['days_diff'].agg(['mean', 'std']).reset_index()
    interval_stats.rename(columns={'mean': 'interval_mean', 'std': 'interval_std'}, inplace=True)

    res = res.merge(interval_stats, on='CustomerID', how='left')

    # Fill NaNs for intervals (e.g., single purchase customers)
    # -1 indicates undefined interval (only 1 purchase)
    res['interval_mean'] = res['interval_mean'].fillna(-1)
    res['interval_std'] = res['interval_std'].fillna(-1)

    # 2. Windowed Features (30, 90, 180 days)
    windows = [30, 90, 180]

    for days in windows:
        start_date = cutoff_date - datetime.timedelta(days=days)
        df_window = df[(df['InvoiceDate'] >= start_date) & (df['InvoiceDate'] < cutoff_date)]

        agg_dict = {
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique',
            'StockCode': 'nunique'
        }

        win_agg = df_window.groupby('CustomerID').agg(agg_dict).reset_index()

        suffix = f"{days}d"
        win_agg.rename(columns={
            'TotalAmount': f'monetary{suffix}',
            'InvoiceNo': f'frequency{suffix}',
            'StockCode': f'unique_items{suffix}'
        }, inplace=True)

        res = res.merge(win_agg, on='CustomerID', how='left')

        cols = [f'monetary{suffix}', f'frequency{suffix}', f'unique_items{suffix}']
        res[cols] = res[cols].fillna(0)

    # --- NEW: Trend Features ---
    # Ratio of short-term (30d) activity vs long-term (180d) average activity
    # Normalize by days to get daily average rate comparison

    # Avoid division by zero by adding small epsilon
    epsilon = 1e-6

    # Monetary Trend: (30d sum / 30) / (180d sum / 180)
    res['trend_monetary'] = (res['monetary_30d'] / 30) / ((res['monetary_180d'] / 180) + epsilon)

    # Frequency Trend
    res['trend_frequency'] = (res['frequency_30d'] / 30) / ((res['frequency_180d'] / 180) + epsilon)

    # --- Target Generation ---
    target_end = cutoff_date + datetime.timedelta(days=30)
    df_target = df[(df['InvoiceDate'] >= cutoff_date) & (df['InvoiceDate'] < target_end)]

    target_agg = df_target.groupby('CustomerID')['TotalAmount'].sum().reset_index()
    target_agg.rename(columns={'TotalAmount': 'target_amount'}, inplace=True)

    res = res.merge(target_agg, on='CustomerID', how='left')
    res['target_amount'] = res['target_amount'].fillna(0)

    return res

def train_and_evaluate(train_data, test_data, features, target, fold_name="Fold"):
    X_train = train_data[features]
    y_train = train_data[target]

    X_test = test_data[features]
    y_test = test_data[target]

    # Hyperparameter Search Space
    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_iter': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_leaf': [10, 20, 50],
        'l2_regularization': [0.0, 0.1, 1.0]
    }

    # Stage 1: Classification (Buy or Not)
    y_train_class = (y_train > 0).astype(int)
    y_test_class = (y_test > 0).astype(int)

    clf = HistGradientBoostingClassifier(random_state=42)
    clf_search = RandomizedSearchCV(clf, param_dist, n_iter=10, cv=3, scoring='roc_auc', random_state=42, n_jobs=-1)
    clf_search.fit(X_train, y_train_class)

    best_clf = clf_search.best_estimator_
    prob_purchase_test = best_clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test_class, prob_purchase_test)

    # Stage 2: Regression (Amount | Buy)
    mask_positive = y_train > 0
    X_train_reg = X_train[mask_positive]
    y_train_reg = y_train[mask_positive]

    y_train_reg_log = np.log1p(y_train_reg)

    reg = HistGradientBoostingRegressor(random_state=42)
    reg_search = RandomizedSearchCV(reg, param_dist, n_iter=10, cv=3, scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)
    reg_search.fit(X_train_reg, y_train_reg_log)

    best_reg = reg_search.best_estimator_

    pred_log_amount_test = best_reg.predict(X_test)
    pred_amount_test = np.expm1(pred_log_amount_test)

    # Final Combination
    final_preds = prob_purchase_test * pred_amount_test

    rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    mae = mean_absolute_error(y_test, final_preds)

    return {
        'fold': fold_name,
        'rmse': rmse,
        'mae': mae,
        'auc': auc,
        'best_clf_params': clf_search.best_params_,
        'best_reg_params': reg_search.best_params_
    }, best_clf

def main():
    filepath = 'data.csv'
    df = load_and_preprocess(filepath)

    print(f"Data range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

    features = ['recency', 'tenure', 'interval_mean', 'interval_std', 'trend_monetary', 'trend_frequency']
    for days in [30, 90, 180]:
        features.extend([f'monetary_{days}d', f'frequency_{days}d', f'unique_items_{days}d'])
    target = 'target_amount'

    # --- Cross-Validation Loop ---
    n_folds = 3 # Number of time-series splits
    latest_test_cutoff = df['InvoiceDate'].max() - datetime.timedelta(days=30)
    latest_test_cutoff = latest_test_cutoff.replace(hour=0, minute=0, second=0, microsecond=0)

    cv_results = []
    last_clf = None
    last_test_data = None

    print(f"\nStarting Time-Series Cross-Validation with {n_folds} folds...")

    for i in range(n_folds):
        # Shift cutoff back by 30 days for each fold
        test_cutoff = latest_test_cutoff - datetime.timedelta(days=i*30)
        fold_name = f"Fold {i+1} (Test: {test_cutoff.date()})"

        # Check if we have enough history (approx 6 months training + buffer)
        min_history_date = test_cutoff - datetime.timedelta(days=30*7)
        if min_history_date < df['InvoiceDate'].min():
            print(f"Skipping {fold_name}: Insufficient historical data.")
            break

        print(f"\nProcessing {fold_name}...")

        # 1. Generate Training Data (Multiple slices prior to test_cutoff)
        train_cutoffs = []
        current_cutoff = test_cutoff - datetime.timedelta(days=30)
        # Go back 6 months for training samples
        for _ in range(6):
            if current_cutoff < df['InvoiceDate'].min() + datetime.timedelta(days=30):
                break
            train_cutoffs.append(current_cutoff)
            current_cutoff -= datetime.timedelta(days=30)

        train_dfs = []
        for cutoff in train_cutoffs:
            known_mask = df['InvoiceDate'] < cutoff
            known_customers = df.loc[known_mask, 'CustomerID'].unique()
            train_dfs.append(get_features_and_target(df, cutoff, known_customers))

        train_data = pd.concat(train_dfs, ignore_index=True)

        # 2. Generate Test Data
        known_customers_test = df.loc[df['InvoiceDate'] < test_cutoff, 'CustomerID'].unique()
        test_data = get_features_and_target(df, test_cutoff, known_customers_test)

        print(f"  Train samples: {train_data.shape[0]}, Test samples: {test_data.shape[0]}")

        # 3. Train and Evaluate
        metrics, clf = train_and_evaluate(train_data, test_data, features, target, fold_name)
        cv_results.append(metrics)

        last_clf = clf
        last_test_data = test_data

        print(f"  >> RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f} | AUC: {metrics['auc']:.4f}")

    # --- Summary ---
    print("\n=== Cross-Validation Summary ===")
    if cv_results:
        results_df = pd.DataFrame(cv_results)
        cols = ['fold', 'rmse', 'mae', 'auc']
        print(results_df[cols])
        print("\nAverage Metrics:")
        print(results_df[['rmse', 'mae', 'auc']].mean())

        # Feature Importance (from the last fold's classifier)
        if last_clf is not None:
            print("\nFeature Importance (Classifier - Last Fold):")
            result = permutation_importance(last_clf, last_test_data[features], (last_test_data[target] > 0).astype(int), n_repeats=5, random_state=42, n_jobs=-1)
            sorted_idx = result.importances_mean.argsort()
            importances = pd.Series(result.importances_mean[sorted_idx], index=np.array(features)[sorted_idx])
            print(importances.sort_values(ascending=False))
    else:
        print("No folds were completed successfully.")

if __name__ == "__main__":
    main()
```
:::

This code includes essentially all the elements I originally wanted. While minor improvements remain (displaying regression feature importance, increasing fold count, etc.), I'll stop here.

With this approach, I basically accepted the LLM's proposals, but even seemingly sound suggestions could delay the most important validation design and implementation. I felt it's risky to take all strategic-level recommendations at face value.


## Conclusion

Through this experiment, I'd like to summarize several overall observations.

Regarding the balance of instructions, while some conversation was somewhat leading, I was generally able to incorporate what I originally wanted to achieve. While minor modifications are faster to do myself, having AI handle changes where the content is clear but volume is substantial (like cross-validation) was efficient.

Regarding review, both its importance and burden stood out. While the LLM sometimes thoughtfully made decisions on ambiguous specifications, these don't necessarily align with developer intent, so thorough review is always required. Additionally, reviewing changes each time gradually became burdensome. Both finding changes and the review itself felt taxing. UI improvements might help the former, but the latter will likely remain an issue requiring good countermeasures or adaptation. Furthermore, as modifications were repeated in conversation format, the breadth of LLM responses seemed to narrow. Periodically saving conversation content or latest specifications to reset context and thinking appears necessary.

Overall, code almost never completely failed to run during testing, which subjectively feels quite good. However, I found the approach of gradually improving code through conversation burdensome. The method from the previous article of generating from comprehensive requirements upfront and reducing review frequency seems better suited to me. When what you want to achieve is clear, preparing detailed instructions upfront is more efficient with fewer conversational exchanges.

As coding assistance tools and LLMs continue to improve, I'll keep exploring better usage patterns and approaches.
