---
title: "Gemini CLI と Gemini 3.0 Pro でテーブルデータの Vibe Modeling (のような何か) を試す"
emoji: "🍀"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["googlecloud", "vertexai", "gemini"]
published: true
---

[JP_Google Developer Experts Advent Calendar 2025](https://adventar.org/calendars/11658) の 2 日目の記事です。

## はじめに

最近、生成AIを活用した開発手法が各所で活用されています。その中でも特に自然言語で指示を出すだけ、場合によっては手放し運転状態で構築する、いわゆる Vibe Coding も試されています。

筆者の業務では、テーブルデータを用いた機械学習モデルの学習や推論を行う機会が少なくありません。プロジェクトごとにデータは異なりますが、前処理やパイプラインの構築など、毎回似たようなコードを書いているなと感じたり、「このあたりのカスタマイズを生成AIに任せてもっと効率化できない？ Vibe Modeling できない？」と思ったりします。

一般的に、ソフトウェア開発における生成AI支援では指示を具体的にすればするほど成果物の品質が高まる傾向にあります。この記事では、プロンプトの指示レベルを変化させて機械学習パイプラインのコードを生成し（本記事ではこれを勝手に Vibe Modeling と呼びます）、生成されたコードがどのように変化するか、その品質を確認したいと思います。

なお、本記事では生成されたコードに対して対話しながら修正するようなアプローチは取らず、最初の指示による一発出しや LLM の自律的な試行錯誤でどこまでいけるかを実験します。

![](/images/articles/vibe-ml-gemini/vibe_modeling.jpg)
*Vibe Modeling のイメージ(本当か?)*

## 実験の設定

本記事で使ったデータセットは [Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) で、データ冒頭はこのような内容になっています。事前にデータを確認した上で以降のコード生成を試しています。

![](/images/articles/vibe-ml-gemini/data.png)
*入力データ*

コーディング支援のツールとしては [Gemini CLI](https://github.com/google-gemini/gemini-cli) を使用し、LLM には [Gemini 3.0 Pro](https://deepmind.google/models/gemini/pro/) を設定しています。なお、実行ログを確認したところ、一部の処理では自動的に Gemini 2.5 Flash に切り替わって処理されている場面もありました。
プロンプトの指示レベルとしては次の 3 段階で変化させて、生成されるモデルの学習・推論コードがどのように変化するか、その品質を確認します。

- ① 目的と問題設定、重要な注意点のみを記述

    :::details 01_simple_requirements.txt
    ```text
    # モデル学習・推論notebookの要件
    - 使用データ
        - ./data.csv
    - 次のような問題設定とする
        - 予測日を基準に、前日までの30日間のデータを使って向こう30日間の購入金額を予測する
        - CustomerIDごとに集計を行い、特徴量と目的変数を作成する
    - 事前にデータ確認を行い、必要と考えられる前処理を行う
    - 検証設計
        - 学習・検証データの分割方法は、データ特性に合わせて決めて
        - リークは絶対に避けるよう徹底的に確認して
    - その他
        - コードは `01_simple.py` に記述
        - 動作確認は `uv run 01_simple.py` で実行
    ```
    :::

- ② ①に加えて、試行錯誤させて改善を促す

    :::details 02_try_and_error_requirements.txt
    ```text
    # モデル学習・推論notebookの要件
    - 使用データ
        - ./data.csv
    - 次のような問題設定とする
        - 予測日を基準に、前日までの30日間のデータを使って向こう30日間の購入金額を予測する
        - CustomerIDごとに集計を行い、特徴量と目的変数を作成する
    - 事前にデータ確認を行い、必要と考えられる前処理を行う
    - 検証設計
        - 学習・検証データの分割方法は、データ特性に合わせて決めて
        - リークは絶対に避けるよう徹底的に確認して
    - その他
        - 設計、実装したところで、その内容が適切かどうかをよく考え、動作確認して、必要に応じて改善・修正する
        - コードは `02_try_and_error.py` に記述
        - 動作確認は `uv run 02_try_and_error.py` で実行

    ```
    :::

- ③ ①に加えて、パイプラインや特徴量について詳しく記述（=筆者がやりたいことに相当）

    :::details 03_detail_requirements.txt
    ```text
    # モデル学習・推論notebookの要件
    - 使用データ
        - ./data.csv
    - 次のような問題設定とする
        - 予測日を基準に、「前日までの30日間のデータを使って向こう30日間の購入金額を予測する」という設定とする
            - この設定は、データの性質や業務プロセス、運用方法などによって変化することもあるので、実際の制約にもとづいてより良いやり方を考えてみてください
        - 例えば、2011-03-01 に予測日とする場合
            - InvoiceDateが2011-02-28以前の90日間のレコードを集計して特徴量を作成する
            - InvoiceDateが2011-03-01以降の30日間の購入金額の合計を目的変数とする
        - 予測日をシフトすることでこのような特徴量、目的変数のセットを複数作れるので、それらを組み合わせて学習データ、検証データを用意する
    - 事前に行ったデータ確認の結果をもとに、次のような前処理を行う
        - `pd.read_csv` でデータを読み込む際、`encoding='shift-jis'` を指定してエンコードエラーを回避
        - `CustomerID` で型指定（`str` ）
        - `InvoiceDate` カラムをdatetime型に変換
        - `CustomerID` が欠損しているレコードを削除
        - `Quantity` がマイナス値のレコードを削除
    - まずは簡単な特徴量で、精度評価を行えるようにする
        - 特徴量、目的変数とも、集計はCustomerIDとprediction_dateで一意とする
        - 特徴量
            - レコード数
            - InvoiceNoのnunique
            - StockCodeのnunique
            - InvoiceDateのnunique
            - Countryのmode
            - これまでの購入金額のsum, max, min, median
        - 目的変数
            - (UnitPrice * Quantity) の合計
        - 交差検証
            
            | Fold | 学習データ | 検証データ |
            | --- | --- | --- |
            | 1 | ”2011-03-01”, “2011-04-01”, “2011-05-01” | “2011-06-01” |
            | 2 | “2011-04-01”, “2011-05-01”, “2011-06-01” | “2011-07-01” |
            | 3 | “2011-05-01”, “2011-06-01”, “2011-07-01” | “2011-08-01” |
            | 4 | “2011-06-01”, “2011-07-01”, ”2011-08-01” | “2011-09-01” |
            | 5 | “2011-07-01”, ”2011-08-01”, “2011-09-01” | “2011-10-01” |
            | test | ”2011-08-01”, “2011-09-01”, “2011-10-01” | “2011-11-01” |

        - 目的関数、評価指標は回帰でよく使われるRMSEを用いる
        - モデルはLightGBMを使う
            - scikit learnラッパーではなく、LightGBM の Python APIを使用
            - モデルパラメータはKaggleの公開notebookで使用される設定を使う
        - パイプライン
            - カテゴリ値の特徴量は、モデルに入力する前にLabelEncoderで変換
            - 学習によって得られたartifacts（lgbモデル、label encoder、特徴量カラム）は保存して、推論時にロードできるようにする
            - 推論のみでも実行するような作りにして
    - その他
        - 設計、実装したところで、その内容が適切かどうかをよく考えたり動作確認して、必要に応じて自ら改善してください
        - コードは `03_detail.py` に記述
        - 動作確認は `uv run 03_detail.py` で実行
    ```
    :::

Gemini CLI では別途作成した要件ファイルを指定しながらコードの実装を指示するようにしました。基本となるプロンプトは以下の通りで、`@01_simple_requirements.txt` の部分を各条件に合わせて変更しています。

```text
まず @01_simple_requirements.txt を確認してください。
このドキュメントをもとにモデルの学習、推論コードを実装してください。
```

## ① 目的と問題設定、重要な注意点のみの場合

まずは、一発出しでどのようなコードが出てくるかを確認します。要件には目的と最低限の注意点だけを記載しました。
生成されたコードは以下の通りです。

:::details ① 目的と問題設定のみの場合のコード
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

# 警告を無視（FutureWarningなど）
warnings.filterwarnings('ignore')

def load_and_preprocess(filepath):
    """データの読み込みと基本前処理"""
    print("Loading data...")
    # 日付解析を含めて読み込み
    try:
        df = pd.read_csv(filepath, parse_dates=['InvoiceDate'], encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, parse_dates=['InvoiceDate'], encoding='cp1252')
    
    print(f"Original data shape: {df.shape}")
    
    # CustomerIDがないデータは顧客ごとの予測に使えないため削除
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    # 金額計算
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    # 日付のみのカラム作成（時間は切り捨て）
    df['Date'] = df['InvoiceDate'].dt.normalize()
    
    print(f"Data shape after dropping null CustomerID: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df

def create_features_and_target(df, cutoff_date, target_days=30, feature_days=30):
    """
    指定された基準日(cutoff_date)に基づいて特徴量と目的変数を作成する
    
    特徴量: [cutoff_date - feature_days, cutoff_date) の期間のデータ
    目的変数: [cutoff_date, cutoff_date + target_days) の期間のデータ
    """
    
    # 期間定義
    feature_start = cutoff_date - pd.Timedelta(days=feature_days)
    feature_end = cutoff_date # 基準日の前日まで（基準日は含まない、または基準日の00:00まで）
    
    target_start = cutoff_date
    target_end = cutoff_date + pd.Timedelta(days=target_days)
    
    # データのフィルタリング
    # 特徴量用データ
    feat_df = df[(df['Date'] >= feature_start) & (df['Date'] < feature_end)]
    
    # 目的変数用データ
    target_df = df[(df['Date'] >= target_start) & (df['Date'] < target_end)]
    
    # ベースとなる顧客リスト（特徴量期間に存在した顧客をベースにするか、全顧客にするか）
    # ここでは「特徴量期間にアクションがあった顧客」または「ターゲット期間にアクションがあった顧客」の両方を考慮すべきだが、
    # 「予測日を基準に...購入金額を予測する」なので、
    # 実際には「予測時点でアクティブな顧客」あるいは「全顧客」が対象。
    # 今回はシンプルに、特徴量期間またはターゲット期間にデータが存在する顧客を対象とする。
    # ただし、学習時は「特徴量が得られる顧客」に対して「未来の売上」を紐付ける形式が一般的。
    # もし特徴量期間に購入がなくても、過去に登録があれば予測対象かもしれないが、
    # このデータセットには顧客マスタがないため、トランザクションがある顧客しかわからない。
    # よって、feat_df に存在する顧客をベースとする（直近30日購入なし=0円とみなすなら、ベースを広げる必要がある）。
    
    # 今回は「直近30日間のデータを使って」という要件なので、
    # 特徴量期間にデータがある顧客をメインの母集団としつつ、
    # ターゲット期間のみにデータがある（新規・復帰）は、過去データがないため予測困難として今回は学習データから除外、
    # あるいは特徴量0として扱うことが考えられる。
    # シンプル化のため、「特徴量期間に購入実績がある顧客」を対象とする。
    
    base_customers = feat_df['CustomerID'].unique()
    
    # 特徴量集計
    features = feat_df.groupby('CustomerID').agg({
        'TotalAmount': ['sum', 'mean', 'count'],
        'Quantity': ['sum'],
        'InvoiceDate': ['max'] # 最終購入日計算用
    })
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    
    # Recency（基準日 - 最終購入日）
    features['Recency'] = (cutoff_date - features['InvoiceDate_max'].dt.normalize()).dt.days
    features = features.drop('InvoiceDate_max', axis=1)
    
    # カラム名のリネーム
    features = features.add_prefix('Past30_')
    
    # 目的変数集計
    targets = target_df.groupby('CustomerID')['TotalAmount'].sum().rename('Target_Next30_Amount')
    
    # 結合 (Left Join: 特徴量がある顧客に対して、正解を紐付ける。正解がない場合は0円とする)
    data = pd.DataFrame(index=base_customers)
    data.index.name = 'CustomerID'
    
    data = data.join(features, how='left')
    data = data.join(targets, how='left')
    
    # 欠損値埋め
    # 特徴量は base_customers が feat_df から来ているので欠損はないはずだが念のため
    data = data.fillna(0)
    
    return data

def main():
    # 1. データ読み込み
    df = load_and_preprocess('data.csv')
    
    # データの最大日付を確認
    max_date = df['Date'].max()
    min_date = df['Date'].min()
    print(f"Data covers from {min_date} to {max_date}")
    
    # 2. 検証設計
    # 未来の情報をリークさせないため、時系列で分割する。
    # テスト用基準日: データセットの最後から30日前（これにより最後の30日間をテストデータの正解として使える）
    test_cutoff = max_date - pd.Timedelta(days=30)
    
    print(f"\n--- Preparing Test Set (Cutoff: {test_cutoff}) ---")
    test_data = create_features_and_target(df, test_cutoff)
    print(f"Test data shape: {test_data.shape}")
    
    # 学習用基準日: テスト期間とかぶらないように設定する。
    # テスト期間の特徴量期間: [test_cutoff - 30, test_cutoff)
    # テスト期間の目的変数期間: [test_cutoff, test_cutoff + 30)
    # 
    # 学習データは、テストデータの「特徴量期間」以前の情報のみを使うのが安全だが、
    # 予測モデル自体の学習には「過去の特徴量」と「過去の正解（その時点での未来）」があればよい。
    # リークを防ぐには、学習データの「目的変数期間」が、テストデータの「目的変数期間」と重ならない（あるいはテストデータの特徴量計算時点より未来の情報を含まない）ようにする等の配慮が必要。
    # 最も安全なのは、学習データの目的変数期間の終了日が、テストデータの基準日以前であること。
    # つまり Train_Target_End <= Test_Cutoff
    # Train_Cutoff + 30 <= Test_Cutoff
    # Train_Cutoff <= Test_Cutoff - 30
    
    # データを増やすために、30日スライドでいくつか学習データを作成する
    train_cutoffs = []
    current_cutoff = test_cutoff - pd.Timedelta(days=60) # 余裕を持って60日前からスタート
    
    while current_cutoff > min_date + pd.Timedelta(days=30):
        train_cutoffs.append(current_cutoff)
        current_cutoff -= pd.Timedelta(days=30) # 30日ずつ遡る
    
    print(f"\n--- Preparing Train Set (Cutoffs: {[t.date() for t in train_cutoffs]}) ---")
    
    train_dfs = []
    for cutoff in train_cutoffs:
        train_dfs.append(create_features_and_target(df, cutoff))
    
    if not train_dfs:
        print("Error: Not enough data to create training set.")
        return

    train_data = pd.concat(train_dfs)
    print(f"Train data shape: {train_data.shape}")
    
    # 3. モデル学習
    X_train = train_data.drop('Target_Next30_Amount', axis=1)
    y_train = train_data['Target_Next30_Amount']
    
    X_test = test_data.drop('Target_Next30_Amount', axis=1)
    y_test = test_data['Target_Next30_Amount']
    
    print("\n--- Training Model ---")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 4. 評価
    print("\n--- Evaluation ---")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # 特徴量重要度
    print("\n--- Feature Importances ---")
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances)
    
    # 具体的な予測例
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

### 結果の評価

コードの中身を確認してみると、いくつかポジティブな点が見つかりました。まず最初の生成で `pd.read_csv` 実行時にエンコードエラーが発生することを確認した上で、適切なエンコーディングを指定するように修正していました。また、学習データとテストデータの期間をずらして分割しており、リーク対策も意識されています。さらに、マーケティング分析で一般的なRFM分析（Recency, Frequency, Monetary）の観点を取り入れ、RecencyとMonetaryを考慮した特徴量が作成されていました。
一方で、改善すべき点もいくつか見受けられます。まず、データが少なくないにも関わらず、汎化性能を高めるために交差検証が含まれていませんでした。また、モデルには Random Forest が使用されており、テーブルデータにおいて [Kaggle コミュニティでも初手として推奨](https://upura.hatenablog.com/entry/2019/10/29/184617) されている LightGBM などの勾配ブースティング系のモデルを使ってほしいところです。
データの前処理では、事前のデータ確認より Quantity にマイナス値があることがわかっており、その理由や説明が見当たらないためマイナス値のレコードは削除しておくのが無難と考えられる一方で、生成されたコードではマイナス値の考慮がなく、そのまま集計されています。Frequency に関する特徴量もありませんでした。推論のみ行う処理がなかった点ももう一歩というところです。

## ② 設計概要を伝えて試行錯誤で改善を促す

次は、①の内容に加えて LLM 自身で試行錯誤して改善するように指示を出すような条件です。LLMが自分でデータを分析し、より良いコードを書いてくれることを期待します。Gemini はまず、データを分析するコードを生成・実行して確認を行いました。

:::details ② 設計概要を伝えて試行錯誤で改善を促す場合のコード（データ確認）
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
    # 日付パースを指定して読み込む
    try:
        df = pd.read_csv('./data.csv', parse_dates=['InvoiceDate'])
    except Exception as e:
        print(f"Failed to read csv with date parsing: {e}")
        # フォーマットが特殊かもしれないので普通に読む
        df = pd.read_csv('./data.csv')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    print("\nData Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nData Range:")
    print(f"Start: {df['InvoiceDate'].min()}")
    print(f"End: {df['InvoiceDate'].max()}")

    # CustomerIDがないデータは顧客ごとの予測ができないので確認
    missing_customer = df['CustomerID'].isnull().sum()
    print(f"\nMissing CustomerID count: {missing_customer} ({missing_customer/len(df):.2%})")

    # 統計量
    print("\nDescriptive Stats:")
    print(df.describe())

if __name__ == "__main__":
    analyze_data()
```
:::

この分析結果を踏まえて生成された学習コードがこちらです。

:::details ② 設計概要を伝えて試行錯誤で改善を促す場合のコード（学習・推論）
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

### 結果の評価

期待していたほど試行錯誤してくれなかったのですが、最初の出力で以下のような改善が行われ、最終的なコードになっていました。
- コメント追加
- LightGBM のパラメータを追加
- `main` 関数内の処理も適切に関数化・リファクタリング

しかし、ポジティブな点や要改善な点の大筋は ① とあまり変わりませんでした。「試行錯誤して」と指示を含めても、具体的な実装ゴールや重視すべき特徴量などの要件が定義されていないと明らかな改善には繋がりにくいようです。


## ③ 詳細なパイプラインと特徴量を記述

最後は、事前に人間がデータを確認した上で、その内容をもとに具体的に指示を記述するパターンです。③の要件ドキュメントには、パイプラインの構成や作成すべき特徴量を、思い当たるだけ記述しました。

生成されたコードは以下の通りです。

:::details ③ 詳細なパイプラインと特徴量を記述
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

# 設定
INPUT_FILE = './data.csv'
ARTIFACTS_DIR = './artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# 特徴量生成の設定
# 要件書には「前日までの30日間のデータ」とあるが、例示では「90日間のレコードを集計」とあるため、
# 情報量を考慮し例示の90日を採用する。
FEATURE_WINDOW_DAYS = 90
TARGET_WINDOW_DAYS = 30

def load_and_preprocess(file_path):
    print("Loading data...")
    # pd.read_csv でデータを読み込む際、encoding='shift-jis' を指定
    # CustomerIDは読み込み時にstr指定できない場合がある(NAがあるとfloatになる)ため後で変換
    df = pd.read_csv(file_path, encoding='shift-jis')
    
    # InvoiceDate カラムをdatetime型に変換
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # CustomerID が欠損しているレコードを削除
    df = df.dropna(subset=['CustomerID'])
    
    # CustomerIDをstrに変換 (例えば 12345.0 -> "12345")
    df['CustomerID'] = df['CustomerID'].astype(float).astype(int).astype(str)
    
    # Quantity がマイナス値のレコードを削除
    df = df[df['Quantity'] > 0]
    
    # 金額列を追加
    df['Amount'] = df['Quantity'] * df['UnitPrice']
    
    return df.reset_index(drop=True)

def create_features(df, prediction_date):
    """
    指定された prediction_date を基準に特徴量を作成する
    対象データ: [prediction_date - FEATURE_WINDOW_DAYS, prediction_date)
    """
    start_date = prediction_date - pd.Timedelta(days=FEATURE_WINDOW_DAYS)
    end_date = prediction_date
    
    mask = (df['InvoiceDate'] >= start_date) & (df['InvoiceDate'] < end_date)
    target_df = df[mask]
    
    if target_df.empty:
        return pd.DataFrame()

    # Countryのmodeを取得する関数
    def get_mode(x):
        m = x.mode()
        if not m.empty:
            return m.iloc[0]
        return np.nan

    # 集計定義
    # InvoiceNo: nunique
    # StockCode: nunique
    # InvoiceDate: nunique
    # Country: mode
    # Amount: sum, max, min, median
    # count: レコード数として InvoiceNo の count を利用
    agg_funcs = {
        'InvoiceNo': ['count', 'nunique'], 
        'StockCode': ['nunique'],
        'InvoiceDate': ['nunique'],
        'Country': [get_mode],
        'Amount': ['sum', 'max', 'min', 'median']
    }
    
    grouped = target_df.groupby('CustomerID').agg(agg_funcs)
    
    # カラム名の平坦化
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
    
    # prediction_date を付与
    grouped['prediction_date'] = prediction_date
    
    return grouped

def create_target(df, prediction_date):
    """
    指定された prediction_date を基準に目的変数を作成する
    対象データ: [prediction_date, prediction_date + TARGET_WINDOW_DAYS)
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
        
        # 特徴量作成
        feats = create_features(df, pred_date)
        if feats.empty:
            continue
            
        if is_train:
            # 目的変数作成
            targets = create_target(df, pred_date)
            # Left join: 特徴量があるユーザーに対してターゲットを結合
            merged = pd.merge(feats, targets, on='CustomerID', how='left')
            # 期間内に購買がないユーザーは0円とする
            merged['target_amount'] = merged['target_amount'].fillna(0)
            features_list.append(merged)
        else:
            features_list.append(feats)
            
    if not features_list:
        return pd.DataFrame()
        
    return pd.concat(features_list, ignore_index=True)

def main():
    # 1. データロード
    df = load_and_preprocess(INPUT_FILE)
    print(f"Data loaded. Shape: {df.shape}")
    
    # 2. Fold定義
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

    # ラベルエンコーディングの準備
    print("Preparing label encoder...")
    le = LabelEncoder()
    # 欠損はない前提だが、念のためastype(str)
    le.fit(df['Country'].astype(str).unique())
    
    # 保存
    with open(os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    
    # モデルパラメータ
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

    # 交差検証ループ
    models = []
    rmse_scores = []
    feature_columns = None

    print("Starting cross-validation...")
    for i, fold in enumerate(folds):
        print(f"\n=== Fold {i+1} ===")
        
        # 学習データ作成
        train_df = make_dataset(df, fold['train'], is_train=True)
        valid_df = make_dataset(df, fold['valid'], is_train=True)
        
        if train_df.empty or valid_df.empty:
            print("Skipping fold due to empty data.")
            continue
            
        # 特徴量カラムの特定
        drop_cols = ['CustomerID', 'prediction_date', 'target_amount']
        features = [c for c in train_df.columns if c not in drop_cols]
        feature_columns = features # 保存用に保持
        
        X_train = train_df[features].copy()
        y_train = train_df['target_amount']
        X_valid = valid_df[features].copy()
        y_valid = valid_df['target_amount']
        
        # カテゴリ変数の変換
        if 'Country_mode' in X_train.columns:
            X_train['Country_mode'] = le.transform(X_train['Country_mode'].astype(str))
            X_valid['Country_mode'] = le.transform(X_valid['Country_mode'].astype(str))
            categorical_feats = ['Country_mode']
        else:
            categorical_feats = []

        # LGBM Dataset
        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feats)
        lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_feats)
        
        # 学習
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
        
        # 評価
        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        rmse_scores.append(rmse)
        print(f"Fold {i+1} RMSE: {rmse:.4f}")

    if rmse_scores:
        print(f"\nAverage RMSE: {np.mean(rmse_scores):.4f}")
    
    # Artifactsの保存
    if models:
        print("\nSaving artifacts...")
        best_model = models[-1] # 簡易的に最後のモデルを使用
        best_model.save_model(os.path.join(ARTIFACTS_DIR, 'lgb_model.txt'))
        
        with open(os.path.join(ARTIFACTS_DIR, 'feature_columns.pkl'), 'wb') as f:
            pickle.dump(feature_columns, f)
        print(f"Artifacts saved to {ARTIFACTS_DIR}")

    # 推論用テスト
    print("\n=== Inference on Test Set (Simulation) ===")
    # Test setの日付で予測を行う
    test_date_str = test_set_def['test'][0]
    print(f"Predicting for date: {test_date_str}")
    
    test_features_df = make_dataset(df, [test_date_str], is_train=False)
    
    if not test_features_df.empty and os.path.exists(os.path.join(ARTIFACTS_DIR, 'feature_columns.pkl')):
        # ロード
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
        
        # 表示
        print("Top 5 predictions:")
        print(test_features_df[['CustomerID', 'prediction_date', 'predicted_amount']].head())
        
        # 評価 (実際の値と比較)
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

### 結果の評価

要件ドキュメントに記述した内容は概ね実装に反映されているものの、細かく見ると改善点は残っています。例えば、LabelEncoder の `fit` を学習データと検証データの分割前に行っているため、リークのおそれがあります。また、学習の進捗ログが出力されない設定になっており、個人的には学習中の進捗を確認できると嬉しいなと思います（この点は好みが分かれる部分かと思います）。
ただ、このコードであれば手直し程度で精度改善の試行錯誤などの次ステップの作業に移れそうです。①②の場合には追加で必要な手直しが多く、結果、会話を継続したり自身で追加実装する必要があり、個人的には（度々の会話の往復がそこまで好きでない背景もあり）③の方法が最も趣味に合いそうでした。このあたりの匙加減は個々人の趣味嗜好に左右されるとも思います。


## おわりに

今回の実験では、機械学習モデルの学習・推論コードの生成AIによる Vibe Modeling (勝手に呼称) は、最低限の指示でも大きく外したコードにはならないものの、「もっとこうしてほしい」という部分が残る結果になりました。やりたいことを詳細に記述すると概ね実装に反映されるのですが、やりたいことを記述できるまでに人間側でデータ確認を地道に行う必要があり、当たり前ですが事前準備の手間と生成物の品質のトレードオフを感じました。
実装そのものの手間が省けることや、データ確認でもコード生成を活用できるので、ゼロからの実装よりはかなり楽になるものの、AIと人間の作業分担のバランスはまだ迷うところがあるなというのが現在の正直な感想です。
今後もモデルの性能は改善されるでしょうし、引き続き心地よい作業フローや分担方針を探っていければと思います。

明日の [JP_Google Developer Experts Advent Calendar 2025](https://adventar.org/calendars/11658) は岩尾さんの[【Gemini Canvas】小 1 娘の漢検対策に「恐怖の漢字鬼ごっこ」を爆速開発したら効果てきめんだった話](https://zenn.dev/mbk_digital/articles/14e29841551dbc)です！お楽しみに！