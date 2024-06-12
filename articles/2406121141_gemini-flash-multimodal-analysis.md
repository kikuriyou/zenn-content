---
title: "Gemini 1.5 Flashでマルチモーダル分析を試す"
emoji: "🍀"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["googlecloud", "vertexai", "gemini", "vlm", "ai"]
published: true
publication_name: hogeticlab
---

# はじめに

ここ数ヶ月の間にLLM（Large Language Model）とともにVLM（Vision Language Model）が普及し始めており、画像データにテキスト情報を付与すること（Image-to-Text）が容易になりました。これにより、画像データを用いた要因分析がより手軽に行えるようになりつつあります。本記事では、VLMとしてVertex AI Gemini 1.5 Flashを利用して、映画ポスターの画像データがユーザー評価に及ぼす影響の分析を試みます。

## 本記事の対象読者
本記事ではGemini, BERT, SHAPを組み合わせて分析するので、自然言語処理にある程度慣れていたり関心をお持ちの方におすすめです。それぞれの手法や実装の詳細については、Geminiは[こちら](https://zenn.dev/harappa80/articles/vertexai_gemini)、BERTは[こちら](https://zenn.dev/robes/articles/5c1599615290ed)、SHAPは[こちら](https://qiita.com/shin_mura/items/cde01198552eda9146b7)などをご覧ください。


## 分析の概要

テキストデータに対する要因分析はすでに多く目にします。例えば、テキストを単語に分割して回帰分析を行ったり、BERTなどの言語モデルとSHAPを組合せて各特徴量の影響度を探る、といったアプローチです。本記事では、Vertex AI Gemini 1.5 Flashを用いて画像データからキャプションテキストを生成し、そのテキストデータに対してBERTおよびSHAPによる分析を行い、特定指標への影響度の解釈を試みます。具体的には、[IMDB Movies Dataset](https://www.kaggle.com/datasets/amanbarthwal/imdb-movies-data)を用いて、映画ポスター画像のどの要素がユーザー評価に影響しているかを分析します。

![](/images/articles/gemini-flash-multimodal-analysis/analytical_overview.png)
*分析イメージ*


# 分析
以降では、実際に使ったコードと合わせて分析の流れを説明していきます。

## データの確認と画像データのダウンロード
データのプレビューは次の通りです。データ中には予測に有用そうなカラムが多く含まれていますが、今回はマルチモーダルの変換を試すことを目的とするためにポスター画像(PosterカラムのURLから得られる画像データ)とユーザー評価(Rating, 0~10の範囲)のみを使うこととします。

![](/images/articles/gemini-flash-multimodal-analysis/df_head.png)
*データのプレビュー*


画像データは、次のコードにより事前にローカルにダウンロードしておきます。

```python
import os
from pathlib import Path
import urllib
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import pandas as pd

class CFG:
    input_dir = Path(f"../input")
    image_dir = input_dir / "images"
    csv_path = input_dir / "imdb-movies-dataset.csv"
    concurrent = True
    num_concurrent = 50

def download_file(url, dst_path):
    try:
        with urllib.request.urlopen(url) as web_file:
            with open(dst_path, 'wb') as local_file:
                local_file.write(web_file.read())
    except Exception as e:
        print(f"Error: {e} in URL: {url}")

imdb_df = pd.read_csv(CFG.csv_path)
if not os.path.exists(CFG.image_dir):
    os.makedirs(CFG.image_dir)
urls = imdb_df["Poster"].tolist()

with ThreadPoolExecutor(max_workers=CFG.num_concurrent) as executor:
    futures = [
        executor.submit(
            download_file, url=url, dst_path=str(CFG.image_dir / url.split("/")[-1])
        ) for url in urls
    ]
    [f.result() for f in tqdm(futures, total=len(futures))]
```
    

## 画像データからのキャプション生成(Image-to-Text)

ダウンロードした画像に対してGemini 1.5 Flashによって画像キャプションを生成します。今回試したデータでは1画像あたり2-3秒でキャプションを生成できたため、処理はサクサク進みました。一方で、並行処理を行おうとすると意外に早くquotaに達してしまうため、大量の画像に対して処理を行いたい場合はquotaや所要時間の配慮は出てきそうです。とはいえマルチモーダルでこの速さはかなり良い体感だったので、色々な方におすすめできると思います。
一点気になったこととして、APIの安全フィルターによって、人の目では問題ないと感じられた画像でテキスト生成が行えないことがありました。扱いたいデータによってその影響が目立つこともあり得るため、この点は注意が必要と感じました。

**参考**
- [Vertex AI での生成 AI の割り当て上限](https://cloud.google.com/vertex-ai/generative-ai/docs/quotas?hl=ja)
- [Gemini モデルの長所と制限事項](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/strengths-limits?hl=ja)
- [Vertex AI の料金](https://cloud.google.com/vertex-ai/generative-ai/pricing?hl=ja)


```python
import os
import re
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

class CFG:
    model_name = "gemini-1.5-flash-preview-0514"
    temperature = 1
    max_tokens = 1000
    input_dir = Path(f"../input")
    image_dir = input_dir / "images"
    csv_path = input_dir / "imdb-movies-dataset.csv"
    pickle_path = input_dir / "image_captions.pkl"
    concurrent = True
    num_concurrent = 5

def generate(prompt, content_path: str):
    for _ in range(3):
        try:
            vertexai.init(project=os.environ["GOOGLE_CLOUD_PROJECT"], location="us-central1")
            with open(content_path, "rb") as f:
                content = Part.from_data(data=f.read(), mime_type="image/jpeg")
            model = GenerativeModel(CFG.model_name)
            response = model.generate_content(
                [content, prompt],
                generation_config={
                    "max_output_tokens": CFG.max_tokens,
                    "temperature": CFG.temperature,
                    "top_p": 0.95,
                },
                stream=False,
            )
            return response.text
        except Exception as e:
            print(f"Error: {e} in the content of {content_path}")
    return ""
        

imdb_df = pd.read_csv(CFG.csv_path)
image_files = [str(CFG.image_dir / url.split("/")[-1]) for url in imdb_df["Poster"].tolist()]

prompt = textwrap.dedent(
    """\
    Please provide a concise description of the image, focusing on the following:

    - Avoid proper nouns and numbers.
    - Mention the image's atmosphere, appeal, and distinctive features.
    """
)

with ThreadPoolExecutor(max_workers=CFG.num_concurrent) as executor:
    futures = [
        executor.submit(generate, prompt=prompt, content_path=image_file)
        for image_file in image_files
    ]
    image_captions = [f.result() for f in tqdm(futures, total=len(futures))]

with open(CFG.pickle_path, mode='wb') as f:
    pickle.dump(image_captions, f)
```


元のポスター画像と生成されたテキストをいくつか確認してみます。生成されたテキストはいずれも、特に問題なく画像の描写や特徴を的確に説明できていると感じられます。


![](/images/articles/gemini-flash-multimodal-analysis/blown_away.jpg)
*The image is a movie poster with a dark and intense atmosphere. The poster features two close-up images of men's faces, one with a yellow hue, the other with a red hue. The men appear to be yelling and engaged in a tense confrontation. The title of the movie is large and bold, printed in white letters on a black background, highlighting the dramatic nature of the film. \n"*


![](/images/articles/gemini-flash-multimodal-analysis/minari.jpg)
*The image portrays a family of four walking hand-in-hand across a grassy field, bathed in soft sunlight. They are casually dressed in comfortable attire, suggesting a sense of warmth and togetherness. A faint American flag in the background adds a subtle note of patriotism. The image exudes a heartwarming and nostalgic atmosphere, appealing to viewers with its portrayal of familial love and simple pleasures. The focus on the family's interconnectedness through their shared walk and the presence of a young child in their midst add a sense of innocence and hope to the composition. \n*


## キャプションのテキストデータからユーザー評価を予測するモデルを作成

次に、Gemini 1.5 Flashによって得られたテキストデータを使ってユーザー評価を予測するモデルを作成します。事前学習済みモデルとしてBERT([google-bert/bert-base-cased](https://huggingface.co/google-bert/bert-base-cased))を用いて、5epochsほどFinetuningを行います。参考のためにコードも載せていますが長いので読み飛ばして構いません。各パラメータをきちんと探索したわけではありませんが、Finetuningによって検証用データにおけるRMSEが1.07程度になったため学習できたものと判断し次のプロセスに進みます。


:::details コード
```python
import os
import re
from pathlib import Path
import time
import datetime
import pytz
import pickle
import random

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import root_mean_squared_error
import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BertTokenizer, BertModel, BertConfig, AdamW
import wandb

class CFG:
    input_dir = Path("../input")
    image_dir = input_dir / "images"
    csv_path = input_dir / "imdb-movies-dataset.csv"
    pickle_path = input_dir / "image_captions.pkl"
    model_dir = "../model"
    num_sample = None  # None, 100
    text_col = "image_caption"
    target_col = "Rating"
    model_name = "google-bert/bert-base-cased"
    num_workers = 4
    batch_size = 32
    n_epoch = 5
    lr = 5e-5
    max_length = 500
    num_warmup_steps = 0
    n_fold = 5
    val_folds = [0]
    use_fp16 = False
    wandb = True
    seed = 42

def class_to_dict(obj):
    return {key: value for key, value in obj.__dict__.items() if not key.startswith('__')}
    
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class NLPDataset(Dataset):
    def __init__(self, df, tokenizer, is_train=True, max_len=128):
        self.texts = df[CFG.text_col].tolist()
        self.targets = df[CFG.target_col].values
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, ix):
        sentence = str(self.texts[ix])
        target = self.targets[ix]
        text_inputs = self.tokenizer(
            sentence,
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True,
        )
        data = {
            "input_ids": torch.tensor(text_inputs["input_ids"], dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float),
        }
        return data

class NLPModel(nn.Module):
    def __init__(self):
        super(NLPModel, self).__init__()
        self.bert = BertModel.from_pretrained(CFG.model_name)
        self.fc = nn.Linear(768, 1)
        torch.nn.init.normal_(self.fc.weight, std=0.02)
    
    def forward(self, input_ids):
        output = self.bert(
            input_ids=input_ids,
            output_attentions=True,
        )
        last_hidden_state = output["last_hidden_state"]
        emb = last_hidden_state[:, 0, :]
        emb = emb.view(-1, 768)

        output = self.fc(emb)
        return output

def train_one_epoch(model, loss_fn, data_loader, optimizer, device, scheduler, epoch, scaler=None):
    epoch_loss = 0
    epoch_data_num = len(data_loader.dataset)

    model.train()
    for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch = {k : v.to(device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        targets = batch["target"]

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            with amp.autocast(enabled=CFG.use_fp16):
                preds = model(input_ids)
                loss = loss_fn(preds.squeeze(), targets.squeeze())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()

        if CFG.wandb:
            wandb.log({
                'step': step + epoch*len(data_loader),
                'train_loss': loss,
                'lr': scheduler.get_lr()[0],
            })
    
    epoch_loss_per_data = epoch_loss / epoch_data_num
    return epoch_loss_per_data

def valid_one_epoch(model, loss_fn, data_loader, device):
    epoch_loss = 0
    epoch_data_num = len(data_loader.dataset)
    pred_list = []
    target_list = []

    model.eval()
    for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch = {k : v.to(device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        targets = batch["target"]
        
        with torch.no_grad():
            preds = model(input_ids)
            loss = loss_fn(preds.squeeze(), targets.squeeze())
            epoch_loss += loss.item()

        pred_list.append(preds.detach().cpu().numpy())
        target_list.append(targets.detach().cpu().numpy())

    epoch_loss_per_data = epoch_loss / epoch_data_num
    val_preds = np.concatenate(pred_list, axis=0)
    val_targets = np.concatenate(target_list, axis=0)
    return epoch_loss_per_data, val_preds, val_targets

def train(train_df, valid_df, exec_time, fold):
    set_seed(CFG.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"train run device : {device}")

    model = NLPModel()
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(CFG.model_name)
    scaler = amp.GradScaler(enabled=CFG.use_fp16)

    train_dataset = NLPDataset(train_df, tokenizer, max_len=CFG.max_length, is_train=True)
    valid_dataset = NLPDataset(valid_df, tokenizer, max_len=CFG.max_length, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    num_training_steps = int(len(train_loader) * CFG.n_epoch)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CFG.num_warmup_steps,
        num_training_steps=num_training_steps
    )
    loss_fn = nn.MSELoss()

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    valid_period = 1

    results_list = []
    val_preds_list = []
    old_model_save_path = None
    score_old = 9999

    for epoch in range(CFG.n_epoch):
        train_epoch_loss = train_one_epoch(
            model, loss_fn, train_loader, optimizer, device, scheduler, epoch=epoch, scaler=scaler
        )
        valid_epoch_loss, val_preds, val_targets = valid_one_epoch(
            model, loss_fn, valid_loader, device
        )
        val_score = root_mean_squared_error(val_targets, val_preds)
        print(f"{epoch=}, {val_score=}")

        lr = optimizer.param_groups[0]['lr']
        results = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_epoch_loss,
            "valid_loss": valid_epoch_loss,
            "score": val_score
        }
        print(results)
        results_list.append(results)
        
        if CFG.wandb:
            wandb.log(results)

        msg = f"[Epoch: {epoch+1}/{CFG.n_epoch}] val_loss={valid_epoch_loss:.4f}, val_score={val_score:.2f}"
        print(msg)

        if val_score < score_old:
            msg = "val_score is updated, save the model."
            print(msg)
            model_save_path = f'{CFG.model_dir}/{exec_time}/model_best_score_fold-{fold}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
            }, model_save_path)
            score_old = val_score

    return val_score, results_list, val_preds

df = pd.read_csv(CFG.csv_path)
with open(CFG.pickle_path, mode='br') as f:
    image_captions = pickle.load(f)

# テキスト生成できなかった画像や欠損データを除去
df["image_caption"] = image_captions
df = df[df["image_caption"]!=""].reset_index(drop=True)
df = df[df["Rating"].notnull()].reset_index(drop=True)

exec_time = datetime.datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d-%H%M%S')
ckpt_path = Path(CFG.model_dir) / Path(exec_time)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
print(f'exec_time: {exec_time}')

if CFG.wandb:
    wandb.login()

folds = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
oof_preds = np.zeros((len(df), 1))
val_scores = []

print(f"CV starts, val_folds: {CFG.val_folds}")
for fold, (trn_idx, val_idx) in enumerate(folds.split(df)):
        # 簡略化のため1foldのみ
    if fold not in CFG.val_folds:
        break
    print(f"Fold: {fold}")

    if CFG.wandb:
        project = "image-shap"
        wandb.init(
            project=project,
            config=class_to_dict(CFG),
            name=f"{exec_time}_fold_{fold}",
            group=f'{exec_time}_{project}',
            job_type="train",
            anonymous="must",
        )

    train_df = df.iloc[trn_idx].reset_index(drop=True)
    valid_df = df.iloc[val_idx].reset_index(drop=True)
    val_score, score_list, val_preds = train(train_df, valid_df, exec_time, fold)
    val_scores.append(val_score)
    oof_preds[val_idx] = val_preds

    if CFG.wandb:
        wandb.finish()

print(f"CV end.")
```
:::

## 作成したモデルとSHAPを用いて要因分析

作成したモデルに対して下記のコードを実行するとSHAP値が計算され、文章ごとのユーザー評価への寄与度を表す図が表示されます。SHAP値の計算には時間がかかるため今回は簡易的に10件に絞っていますが、件数を増やすことで結果や精度にも影響が出ることにご注意ください。

```python
import shap

def f(sentences):
    input_ids = torch.tensor([tokenizer.encode(text, max_length=CFG.max_length, padding="max_length", truncation=True,) for text in sentences]).to(device)
    with torch.no_grad():
        output = model(input_ids)
    return output.detach().cpu()

explainer = shap.Explainer(model=f, masker=tokenizer, output_names=["Rating"])

shap_values = explainer(df["image_caption"].sample(n=10, random_state=CFG.seed))
shap.plots.text(shap_values)
```


表示される図では、文章ごとにユーザー評価に対して正の寄与がある場合に赤、負の寄与がある場合に青、さらにその寄与度が大きいほど長く表示される、といった見方になります。図の下部には元の文章も表示されており、寄与度の大きい箇所に赤色または青色が付与されます。

まず1つ目の結果について、`which is both jarring and powerful , drawing attention to the man ' s suffering and the film ' s unsettling subject matter .` という部分が正の寄与があることがわかります。一方で、`The atmosphere of the image is dark` は負の寄与があるとの説明になっています。暗い雰囲気がありつつも男性のワイルドな描写がユーザー評価に効いているということでしょうか。

![](/images/articles/gemini-flash-multimodal-analysis/shap_1.png)
*文章の各箇所のユーザー評価への寄与度①*

![](/images/articles/gemini-flash-multimodal-analysis/dog_pound.jpg)
*元画像①*


次の結果に移ります。この画像および文章においては、`A small , metallic object rests on the ground in front of them , adding to the mysterious and eerie atmosphere` 、さらには `dread , and vulnerability . The young person ' s posture and the ominous symbol behind them create a sense of impending danger` という箇所が正の寄与を示していて、不気味な雰囲気がユーザー評価に効いていることが伺えます。

![](/images/articles/gemini-flash-multimodal-analysis/shap_2.png)
*文章の各箇所のユーザー評価への寄与度②*

![](/images/articles/gemini-flash-multimodal-analysis/wish_upon.jpg)
*元画像②*


# 終わりに

本記事では分析のざっくりとしたイメージを伝えるために実装や設定を簡単に留めていますが、実際の業務適用にあたってはいくつか意識するべき点があります。

**Image-to-Textで適切な情報を抽出するためのプロンプト調整**
今回は簡単なプロンプトで試しましたが、分析の目的や観点に応じてプロンプトを練ることで必要な情報を抽出できるようになります。その結果、SHAPによる要因分析で現れる文章群も業務で活用しやすい内容になることが期待されます。
**画像以外のデータの利用**
冒頭でも触れたように、IMDBのデータにはユーザー評価に関わる有用と思われるカラムが多く含まれます。今回はImage-to-Textのイメージをお伝えする目的で省きましたが、実際にはこれらを考慮して分析する必要があります。
**文章同士の相互作用**
上記で紹介した可視化は1次元に投影されたものになっていますが、BERTはアテンション機構や全結合が含まれるため、文章や単語同士の相互作用を無視することはできません。NNモデルの内部を完全に理解するのは難しいものの、shapパッケージ内の他の可視化手法を併用することも役立つかもしれません。
**データに含まれるバイアスの考慮**
モデルの解釈結果に意図せずバイアスが含まれる場合があります。詳しくは[Responsible AI practices](https://ai.google/responsibility/responsible-ai-practices/)等を参照いただきたいですが、分析結果をまとめる際にはバイアス可能性の考慮が必要です。

また、過去の似た事例として、広告クリエイティブの領域ではテキストデータの寄与度分析を行い、その結果をクリエイティブ作成に活用した[事例](https://www.thinkwithgoogle.com/intl/ja-jp/marketing-strategies/automation/ml_creative/)があります。今回は画像データの要素を抽出しているので、その結果を使って画像クリエイティブの作成やText-to-Imageのプロンプトに役立てることもできるかもしれません。その他の応用例として、Gemini 1.5 FlashはVideo-to-Textの変換も行えるため動画データでも同様の分析を試すのも可能かと思います。
以上、簡単ではありますが、Vertex AI Gemini 1.5 Flashを活用した画像データの要因分析を紹介しました。何かの参考になれば幸いです。

# 参考文献

- [IMDB Movies Dataset](https://www.kaggle.com/datasets/amanbarthwal/imdb-movies-data)
- [Vertex AI での生成 AI の割り当て上限](https://cloud.google.com/vertex-ai/generative-ai/docs/quotas?hl=ja)
- [Gemini モデルの長所と制限事項](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/strengths-limits?hl=ja)
- [Vertex AI の料金](https://cloud.google.com/vertex-ai/generative-ai/pricing?hl=ja)
- [SHAP(SHapley Additive exPlanation)についての備忘録](https://qiita.com/perico_v1/items/fbbb18681ecc362a4f9e)
- [Emotion classification multiclass example](https://shap.readthedocs.io/en/latest/example_notebooks/text_examples/sentiment_analysis/Emotion%20classification%20multiclass%20example.html)
- [機械学習モデルを解釈する指標SHAPを自然言語処理に対して使ってみた](https://qiita.com/m__k/items/87cf3e4acf414408bfed)
- [広告クリエイティブ分野にも進む機械学習の浸透](https://www.thinkwithgoogle.com/intl/ja-jp/marketing-strategies/automation/ml_creative/)
- [Responsible AI practices](https://ai.google/responsibility/responsible-ai-practices/)