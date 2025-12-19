---
title: "Gemini CLI と Gemini 3.0 Pro でテーブルデータの Vibe Modeling (のような何か) を試す - 会話編"
emoji: "🍀"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["googlecloud", "vertexai", "gemini"]
published: true
published_at: 2025-12-19 08:00
---

[JP_Google Developer Experts Advent Calendar 2025](https://adventar.org/calendars/11658) の 19 日目の記事です。

## はじめに

[前の記事](https://zenn.dev/kikuriyou/articles/2511271718_vibe-ml-gemini)では、テーブルデータのモデル学習・推論コードを一発出しした場合の生成結果の品質について確認しました。
この記事では、簡単な要件からコードを生成し始め、その後も会話を続けることで本来やりたい内容に近づけることを試します。その際、コードそのものの変更は自分では行わず、AIへの指示のみで進めるルールとします。

会話を通してどこまで筆者の意図通りに改善を行えるか、また前記事で要件を詳しく記述した場合との生成結果や体感の違いなどを確認します。

## 実験の設定

生成開始時点の設定は基本的に前記事と同じ内容とします。

- コーディング支援: Gemini CLI
- LLM: Gemini 3.0 Pro
- データ: [Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)

モデル学習、推論の Gemini CLI での指示は以下の通りです。前記事の 1（目的と問題設定のみ）と同じ指示から開始して、その後も会話を続けて改善を試みます。

```docker
まず @01_simple_requirements.txt を確認してください。
このドキュメントをベースにモデルの学習、推論コードを実装してください。
```

以降全体を通して、AIによるコード生成には自動 Lint による変更が含まれることに注意してください。

## 1. やりたいことが決まっている場合

まずは、最終的にやりたい構成（LightGBM、交差検証、特定の特徴量など）が決まっているが、最初は簡単な指示から始めて、徐々に指示を追加していくパターンです。

### 初期生成

まずは一発出しのコードを確認します。

:::details 初期生成コード
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

**評価:**
- モデルには Random Forest が使われている（LightGBM を使ってほしい）。
- 評価指標は MAE, R2 が使われている（RMSE がほしい）。
- 検証は単純な holdout で行われている（交差検証にしたい）。
- 特徴量の作成において `Quantity` や `UnitPrice` が 0 より大きいレコードに絞っています。これは気が利いているとも取れますが、本来はデータの仕様やビジネス要件に関わる部分なので、勝手に判断せず確認してほしい部分でもあります。

### 性能改善のための会話と修正

汎化性能を向上させるため、交差検証への変更を指示します。

```
汎化性能を向上させたいです。現在は単純なholdoutですが、交差検証に変更してください。
これまでに引き続き、リークには十分に気をつけてください。
```

:::details 生成されたコード（交差検証 / 抜粋）
```python

# Time Series Cross-Validation
n_splits = 4
tscv = TimeSeriesSplit(n_splits=n_splits)

cv_mae_scores = []
cv_r2_scores = []

print(f"\nStarting Time Series Cross-Validation with {n_splits} splits...")

fold = 0
for train_index, val_index in tscv.split(unique_dates):
    fold += 1
    val_dates = unique_dates[val_index]
    potential_train_dates = unique_dates[train_index]
    
    # LEAKAGE PREVENTION:
    # Ensure there is a gap between the training data and validation data.
    # Validation targets are for [val_date, val_date + 29].
    # Training targets are for [train_date, train_date + 29].
    # We must strictly ensure that the Training Target period ends before the Validation Prediction Date?
    # Actually, more strictly: When making a prediction at `val_date`, we only have access to data 
    # where the target outcome is already known.
    # Target outcome for `train_date` is known at `train_date + 30`.
    # So we need `train_date + 30 <= val_date`.
    
    min_val_date = val_dates.min()
    valid_train_dates = [d for d in potential_train_dates if d + pd.Timedelta(days=30) <= min_val_date]
    
    if not valid_train_dates:
        print(f"Fold {fold}: Skipped - Not enough history for training with strict leakage prevention gap.")
        continue
        
    train_df = full_dataset[full_dataset['prediction_date'].isin(valid_train_dates)]
    val_df = full_dataset[full_dataset['prediction_date'].isin(val_dates)]
    
    X_train = train_df.drop(columns=['CustomerID', 'next_30_days_total_amount', 'prediction_date'])
    y_train = train_df['next_30_days_total_amount']

    X_val = val_df.drop(columns=['CustomerID', 'next_30_days_total_amount', 'prediction_date'])
    y_val = val_df['next_30_days_total_amount']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    cv_mae_scores.append(mae)
    cv_r2_scores.append(r2)
    
    print(f"Fold {fold}: Train Dates: {len(valid_train_dates)} ({pd.to_datetime(valid_train_dates[0]).date()} to {pd.to_datetime(valid_train_dates[-1]).date()}), Val Dates: {len(val_dates)} ({pd.to_datetime(val_dates[0]).date()} to {pd.to_datetime(val_dates[-1]).date()}) - MAE: {mae:.2f}, R2: {r2:.2f}")

if cv_mae_scores:
    print(f"\nAverage MAE across {len(cv_mae_scores)} folds: {np.mean(cv_mae_scores):.2f}")
    print(f"Average R2 across {len(cv_mae_scores)} folds: {np.mean(cv_r2_scores):.2f}")
else:
    print("\nNo valid folds found for cross-validation.")
    return

```
:::

この変更は特に問題なさそうです。

さらに、評価指標の変更と、検証期間の刻みを変更する指示を追加します。

```bash
- 評価指標にRMSEを追加して
- current_prediction_dateが7日刻みですが30日ごとにして
```

変更が複数箇所にわたるのでコードは省略しますが、これによりデータセット生成の間隔が30日になり、RMSEが出力されるようになりました。ただ、こういったパラメータ変更は本来はわざわざ指示を出さずとも自分でコードを修正したほうが早いです。
次に、モデルを LightGBM に変更します。

```docker
精度を改善したいのでモデルをLightGBMに変更したいです。
scikit-learnラッパーではなく元のLightGBMのPython APIを用いてください。
```

予測結果の解釈性向上のため、特徴量重要度を出力させます。

```python
予測結果を解釈しやすくしたいのでimportanceを出力して
```

ここまでのコードはこのようになります。大きな問題はありませんが、重要度はグラフで出力してほしいなーとも思います。しかしこのあたりから会話に疲れてきたため指示を省いてしまいました。

:::details 生成されたコード（LightGBM, 特徴量重要度の追加まで）
```
# Create LightGBM datasets
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

# Train model
model = lgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain, dval],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)] # period=0 disables verbose logging
)

best_iterations.append(model.best_iteration)

y_pred = model.predict(X_val, num_iteration=model.best_iteration)

# (中略..)

if not X_inference.empty:
    inference_predictions = model_final.predict(X_inference, num_iteration=model_final.best_iteration)
    
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

```
:::


特徴量に重要度を出力したところで、特徴量が3つしか使われていないことに気づきました。
精度改善のため、特徴量の追加を具体的に指示します。

```docker
さらに精度を改善するために特徴量追加を試します。
以下の特徴量を追加してください。
- レコード数
- InvoiceNoのnunique
- StockCodeのnunique
- InvoiceDateのnunique
- countryのmode, nunique
- これまでの購入金額のsum, max, min, median
```

この指示に対して、最初は集計期間を固定しない過去通算の集計を追加してきたり、微妙な修正を何度か繰り返す必要がありました。例えば、ここまで30日windowで集計していたのに急に180日の集計が加わり、この修正のために2,3往復を要しました。
集計ロジックのような細かい部分は、言葉で指示するよりも自分で書くか事前に詳細な仕様を伝える方が確実だと感じました。

:::details 修正の過程（抜粋）
```python
+ long_window_features = long_window_df.groupby('CustomerID').agg(
+     last_180_days_record_count=('InvoiceNo', 'count'),
      # (中略...)
+     last_180_days_amount_median=('Amount', 'median')
+ ).reset_index()
```
:::


### 最終的なコード

会話を続けて完成したコードは以下の通りです。ここまでで筆者のやりたいこと（前記事の詳細な要件）は概ね入ってはいるのですが、微妙に冗長だったり設定を変更したい箇所が少なからずあり、会話だけでの編集はまどろっこしさがあると感じました。会話主体でコーディングする場合でも直接編集は併用したいですね。（それはそう）

:::details 最終的なコード（やりたいことが決まっている場合）
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

### 感想

- パラメータ変更などの細かい修正は、直接コードを編集する方が明らかに早いです。
- 交差検証の実装など、やることは明確だが実装量が地味に多い、というような変更を任せるのは効率的だと感じました。
- 仕様が曖昧な部分で勝手な解釈が入ることがあるため、明確な指示がない箇所は注意が必要です。
- 生成されるたびにレビューが必要ですが、頻繁なレビューは注意力を要するのと、少しの油断でバグや仕様の勘違いを見落とし得ると感じました。（というか度々見落としていた）


## 2. 壁打ちしながら実装を進める場合

次は具体的な改善案を持たずに、実装方針も含めて LLM に相談しながら進めるパターンです。
モデルの学習や推論の文脈では、筆者自身はこの方法はあまり使わないのですが、実装する言語や内容が不慣れな場合（筆者の場合は他分野のアプリ実装など）にこういった進め方をすることがあると考えられるため、参考のため確認しておきます。
ここでは、LLM が言ったことは基本的に信用して、実装を進めることにします。

### 初期生成と前処理の相談

何はともあれまずは初期コードを生成します。

```
まず @05_wallball_requirements.txt を確認してください。
このドキュメントをベースにモデルの学習、推論コードを実装してください。
```
:::details 生成されたコード
```
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
```
:::

このコードはこれまでの概要指示のみの一発出しのコードとほぼ同等の内容なので、続けてデータの前処理の相談に移ります。

```python
データ内容や特性にもとづいて前処理を行いたい、必要な前処理を考えるためにどんなことを確認すれば良いですか
```

:::details 返答
```
✦ データを確認した結果、以下の前処理を検討・実行すべきことがわかりました。

  1. CustomerID 欠損の除外
   * 現状: 135,080件（全体の約25%）のレコードで CustomerID が欠損しています。
   * 対応: 顧客ごとの購買予測を行うという目的設定上、顧客IDが不明なデータは紐付けようがないため除外します。

  2. キャンセルデータの処理 (Negative Quantity / InvoiceNo 'C')
   * 現状: Quantity が負のレコードが 10,624件あります。また、InvoiceNoが 'C' で始まる（キャンセル）レコードが
     9,288件あります。
   * 対応:
       * 単純な予測モデルでは、キャンセル（負の売上）はノイズになりやすいため除外するのが一般的です。
       * ただし、「返品が多い顧客は将来の購入額が低い」といった特徴量として使うことも可能ですが、今回はベース
         ラインとしてキャンセルデータ（Quantity < 0）は除外することを推奨します。

  3. 特殊な取引の除外 (StockCode / UnitPrice)
   * StockCode: POST (送料), D (Discount), BANK CHARGES, AMAZONFEE
     などの特殊コードが確認できました。これらは商品の購入行動とは異なるため、純粋な商品購買予測を行うなら除外
     すべきです。
   * UnitPrice: 0以下の価格が2,517件あり、その多くは Description
     も欠損しています。これらも有効な購買記録ではない可能性が高いため除外します。

  4. 重複データの削除
   * 現状: 5,268件の完全重複行があります。
   * 対応: 集計結果を歪めるため削除します。

  推奨される前処理コードの変更案

  これらの処理を 05_wallball.py の load_and_preprocess 関数に追加実装することをお勧めします。

    1     # ... (読み込み後) ...
    2 
    3     # 1. 重複削除
    4     df = df.drop_duplicates()
    5 
    6     # 2. CustomerID欠損削除
    7     df = df.dropna(subset=['CustomerID'])
    8     df['CustomerID'] = df['CustomerID'].astype(int)
    9 
   10     # 3. キャンセル・返品の除外 (Quantity > 0)
   11     # 注: キャンセル分を売上からマイナスしたい場合は残しますが、
   12     # 今回は「購入行動」にフォーカスするため除外します。
   13     df = df[df['Quantity'] > 0]
   14 
   15     # 4. 特殊なStockCodeの除外 (数字で始まらないものは除外など)
   16     # POST(送料)などは除外するのが無難です
   17     # 簡易的には、StockCodeが数字で始まるものだけ残す、あるいは特定のリストを除外する
   18     # ここでは 'POST', 'D', 'M', 'C2', 'DOT', 'BANK CHARGES' などを除外
   19     special_codes = ['POST', 'D', 'M', 'C2', 'DOT', 'BANK CHARGES', 'CRUK']
   20     df = df[~df['StockCode'].isin(special_codes)]
   21 
   22     # 5. UnitPrice > 0 のデータのみ利用
   23     df = df[df['UnitPrice'] > 0]
   24 
   25     # ... (以下、日付変換など) ...

  この前処理を追加してコードを修正しますか？
```
:::

仕様が曖昧な部分まで実装方針を決めてしまっている部分もありますが、ここでは LLM の言うことを信じてそのまま実装に移りました。

### 精度向上の相談

次に、モデルの精度向上について相談します。

```python
汎化性能を向上させるためにどんな改善が必要ですか
```

:::details 返答
```
✦ 汎化性能（未知のデータに対する予測精度）を向上させるためには、モデルがデータの「背後にあるパターン」をより
  良く捉えられるようにし、かつ「学習データへの過剰適合」を防ぐ必要があります。

  現在のベースラインモデル（過去30日間のRFMのみ）に対して、以下の改善策が考えられます。

  1. 特徴量の拡充 (Feature Engineering)
  現在 Recency, Frequency, Monetary (過去30日) のみですが、顧客の行動をより多角的に捉える特徴量を追加します。
   * 長期的な行動履歴:
     過去30日だけでなく、90日や180日、あるいは全期間の履歴を集計します。「最近30日は買っていないが、その前の
     半年は頻繁に買っていた」といった顧客を捉えるためです。
   * トレンド: (直近30日の購入額) / (過去90日の購入額平均)
     のような比率を計算し、購入が増えているか減っているかの傾向（Trend）を加えます。
   * 商品の多様性: 購入した商品（StockCode）の種類数（ユニーク数）。
   * 顧客期間 (Tenure): 初回購入日から現在までの日数。古参客か新規客かを区別します。
   * 平均単価・バスケットサイズ: 1回あたりの平均購入額や、平均購入点数。

  2. 目的変数の変換 (Log Transformation)
  売上データは「べき乗則」に従うことが多く、少数の「超優良顧客（外れ値）」が全体の誤差(RMSE)を大きく歪めます
  。
   * 目的変数（購入金額）を log(x + 1) で対数変換して学習させ、予測後に exp(x) - 1
     で戻すことで、外れ値の影響を抑え、モデルの安定性を高めます。

  3. モデルの変更
   * RandomForest も強力ですが、テーブルデータでは Gradient Boosting Decision Trees (GBDT) (例:
     scikit-learnの HistGradientBoostingRegressor, XGBoost, LightGBMなど)
     がより高い精度を出しやすい傾向があります。

  4. ゼロ過剰への対策 (Zero-Inflation)
   * 次の30日間で「購入しない（0円）」顧客が多数を占めます。
       * 2段階モデル:
         「買うか買わないか(分類)」×「いくら買うか(回帰)」に分けて予測する手法もありますが、まずは対数変換とG
         BDTで対応するのが手軽で効果的です。

  ---

  これらの中で効果が高く、すぐに実装可能な「特徴量の追加（期間拡張・Tenure）」と「対数変換」を実装して効果を
  検証しましょうか？
```
:::

この回答をもとに、まずは特徴量追加と対数変換を実装してもらいました。これにより、MAEが改善しました。
ただ本来は、様々なアプローチの精度影響を一貫性を持って評価するために、目的に応じて検証方法をきちんと定義し（例えば交差検証を）実装することが最優先になります。
続いて、Gemini からの提案通り、GBDT への変更、そしてハイパラチューニングとゼロ過剰モデル(購入有無の事前推定)の追加を指示しました。

:::details 生成されたコード（GBDT + Zero-Inflated Model）
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

結果として、RMSEが大幅に改善しました（965.4 -> 807.6）。購入金額0円の顧客を分類モデルでフィルタリングできた効果と考えられます。
しかしこの方法は推論を2回行う必要があり、パイプラインが複雑になるため注意が必要です。ただ当初のルールの通り、LLM を信じて次に進みます。

### さらなる改善と仕様の整理

さらなる改善案を聞いたところ、トレンド特徴量や購買間隔統計値の追加が提案されました。しかし前述のように様々なアプローチの有効性を比較するために、先に適切な検証方法を固める必要があります。
このまま続けても現在の holdout のままで続ける様子なので、当初のLLMを信じるというルールは少し脇に置いて、一度会話をリセットして再開しました。
変更内容と今後の指針を明確にするため、AIとの対話内容を一度仕様書 (`05_spec.txt`) としてまとめさせ、それを元に再度提案を求めました。

```
まずは @05_spec.txt を確認してください。
この内容に対してさらに汎化性能を改善するためにどのようなことが必要と考えられますか。
優先度と合わせて教えてください。
```

:::details 返答
```
✦ 05_spec.txt を確認しました。購入予測モデルの汎化性能を改善するために、以下の点が考えられます。優先度順に説明します。

  ---

  汎化性能改善のための提案

  優先度：高

   1. 特徴量エンジニアリングの強化（未実装の改善案を優先的に実装）
       * 内容: 仕様書で「今後の改善案」として挙げられている「トレンド特徴量」や「購買間隔統計」を優先的に実装します。
           * トレンド特徴量: 顧客の直近と長期の購買行動の変化率を捉える特徴量（例: 過去30日間の平均購入金額 /
             過去180日間の平均購入金額）。
           * 購買間隔統計:
             顧客の購入間隔（平均、標準偏差、中央値など）を特徴量として追加し、定期購入者などのパターンを識別します。
       * 理由:
         これらの特徴量は顧客の購買行動パターンをより詳細に捉え、モデルが未知のデータに対しても、個々の顧客の将来の行動
         をより正確に予測できるようになるため、汎化性能向上に直接的に寄与すると考えられます。

   2. 時系列交差検証 (Time Series Cross-Validation) の導入
       * 内容:
         現在の評価方法が固定のテストセットであるため、これを時系列交差検証に置き換えます。学習データとテストデータを時
         系列順に分割し、複数の期間でモデルの評価を行うことで、特定の期間に過学習することなく、より信頼性の高い汎化性能
         の評価が可能になります。
       * 理由:
         時系列データでは、特定の期間のデータに依存しないロバストな予測性能が重要です。時系列交差検証は、モデルが未来の
         データに対してどの程度安定して機能するかを客観的に評価するために不可欠です。

  優先度：中

   3. ハイパーパラメータ探索の改善と効率化
       * 内容: RandomizedSearchCV に加えて、より効率的なハイパーパラメータ最適化手法（例: ベイズ最適化ライブラリ Optuna
         や Hyperopt
         など）の導入を検討します。これにより、限られた計算リソースで、より短時間で最適なハイパーパラメータを見つけ、モ
         デルのポテンシャルを最大限に引き出すことが期待されます。
       * 理由:
         最適なハイパーパラメータはモデルの予測精度と汎化性能に大きく影響するため、探索の質を高めることが重要です。

   4. エラー分析とドメイン知識に基づく特徴量追加
       * 内容:
         モデルの予測誤差が大きい顧客や期間を特定し、その原因を詳細に分析します。その結果から、ビジネスにおけるドメイン
         知識（例: 季節性イベント、大規模セール、新製品発売など）を反映した新たな特徴量（例: カテゴリ/NLP
         情報、顧客のデモグラフィック情報など、利用可能であれば）を追加します。
       * 理由:
         単純なデータドリブンなアプローチだけでなく、ビジネスの背景を特徴量に落とし込むことで、より現実世界の複雑な購買
         パターンをモデルが学習できるようになり、汎化性能と解釈性の両面で改善が見込めます。

  優先度：低

   5. 代替モデルおよびアンサンブル学習の検討
       * 内容: 現在の HistGradientBoosting モデルに加え、他の強力な勾配ブースティングモデル（LightGBM, XGBoost
         など）や、仕様書にも記載のある Tweedie回帰
         のような代替モデルを試します。さらに、これらの複数のモデルを組み合わせるアンサンブル学習の導入も検討します。
       * 理由:
         複数のモデルの予測を組み合わせるアンサンブル学習は、単一モデルの弱点を補完し、予測の安定性と汎化性能を高める効
         果が期待できます。特に Tweedie回帰 は Two-Stage
         モデルの代替として、単一モデルでゼロと金額の両方を予測できる可能性があります。

  ---

  これらの改善策は、モデルの精度向上だけでなく、未知のデータに対するロバスト性を高め、より実用的な購入予測モデルへと進化
  させるために重要であると考えます。
```
:::

ここでようやく交差検証のアイデアが出てきました。このまま、優先度が高いと判断された交差検証と特徴量追加を実装します。

### 最終的なコード

交差検証と追加特徴量を実装した最終的なコードです。

:::details 最終的なコード
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

このコードには、当初やりたかったことが要素レベルではほぼ一通り入っています。細かい改善点（回帰の特徴量重要度表示、fold数の増加など）は残っていますが、ここまでとします。
この進め方では、LLM の提案を基本的に受け入れながら進めましたが、一見的確に思える提案でも最も重要な検証設計・実装が後回しになることがあり、方針レベルで全て鵜呑みにするのは危険だと感じました。


## おわりに

今回の実験を通して感じたことをまとめます。

**指示出しのバランス**
一部誘導的な会話もありましたが、当初の目的は概ね達成できました。
細かい修正は自分で直接編集する方が早い一方、変更内容は明確だがコード量が多い場合（交差検証の実装など）は、AIに任せることで効率的に進められると感じました。

**レビューの重要性と負担**
レビューは非常に重要ですが、同時に負担も大きいと感じました。
仕様が曖昧な部分をLLMが補完してくれることもありますが、必ずしも意図通りとは限らないため、入念なチェックが欠かせません。
また、会話を重ねるごとに変更箇所の特定やレビュー作業自体が辛くなり、心理的な負担が増しました。変更箇所の特定はツールのUI改善で緩和される可能性がありますが、レビュー自体の負担については、慣れや運用上の工夫が必要そうです。

**コンテキストの管理**
会話形式で修正を繰り返すと、徐々にLLMの提案の幅が狭まる傾向が見られました。適度なタイミングで要件や仕様を整理・保存し、コンテキストをリセットすることが有効だと考えられます。

**全体的な感想**
生成されたコードが全く動かないことはほぼなく、その点での体験は非常に良好でした。
しかし、会話を通じて徐々にコードを改善していくスタイルは、筆者にとっては~~かなり疲れました~~負担が大きく感じられました。やりたいことが明確な場合は、前記事のように最初に詳細な要件定義を行い、一括で生成してレビューや会話の回数を減らすアプローチの方が、効率的で合っていると感じました。

今後もツールの進化に合わせて、最適な使い方を探っていきたいと思います。
