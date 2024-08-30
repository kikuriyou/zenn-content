---
title: "Cloud Run で GPU が利用できるようになったので解説します"
emoji: "🍀"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["googlecloud", "cloudrun", "gpu"]
published: false
publication_name: hogeticlab
---


# はじめに
Google Cloud のサーバーレスコンピューティングプラットフォームである Cloud Run に、待望の GPU サポートがプレビュー公開されました。この新機能により、

- Google の Gemma（2B / 7B）や Meta の Llama 3（8B）などの軽量なオープンモデルを使用したリアルタイム推論
- ファインチューニングされたカスタム画像生成 AI モデルのサービング
- オンデマンドの画像認識、動画のコード変換とストリーミング、3D レンダリングといった演算負荷の大きな処理

などといった用途で Cloud Run 上で実行できるようになります。本記事では、このCloud Run の GPU サポートについて解説していきます。

- 公式ドキュメント: [GPU (services)](https://cloud.google.com/run/docs/configuring/services/gpu)

# Cloud Run の GPU サポートとは
Cloud Run はゼロからの自動スケーリングが可能なため、API のサービング等に多く用いられています。しかし、演算は CPU のみの対応であったため、GPU を必要とするワークロードのサービングは難しいのが現状でした(正確にはAnthosを使うと対応可能ではあります)。今回の GPU サポートによって、GPU を必要とするワークロードも含めた API のサービングを手軽に行えることが期待されます。しかしながら、推奨されるインスタンス要件には制約があり、CPU のみの場合と比べて、利用には注意が必要です。

# GPUサポートの技術的詳細
提供される GPU は NVIDIA L4 で、24GB の VRAM を利用可能です。この VRAM はインスタンスメモリとは別に確保されるため、メインメモリを圧迫することなく高速な計算処理が可能となります。現在、GPU サポートが利用可能なリージョンはus-central1（アイオワ）のみで、GPUを利用するにはいくつかの設定要件があります。まず、CPU 常に割り当てる設定が必須で、最小で 4 CPU (推奨 8 CPU)、と 16 GiB (推奨 32 GiB) のメモリが必要です。さらに、GPUやインスタンス数は、リージョンやプロジェクトごとの quota に収まる範囲の設定にすることが求められます。

# 料金体系の特徴
CPU のみの Cloud Run と同様に、インスタンスや GPU の稼働時間に応じて費用が発生します。ただし、前述のように CPU を常に割り当てる設定と、さらに最小 4 CPU (推奨 8 CPU) のインスタンスが必要になるため、アイドル状態でも最小インスタンス稼働分の費用がベースラインとなります。CPU のみの場合ではゼロインスタンスまでスケールインできることと比べると、コスト管理には注意が必要です。

# GPU の利用方法
GPU の設定は、CPU のみの場合と同様に、コンソールや gcloud CLI、YAML ファイルを使用して行えます。現時点では、[Cloud Run GPU sign up](https://services.google.com/fb/forms/cloudrungpusignup/) から利用申請を行う必要があり、この承認には時間を要することがあります。利用開始後は、ドライバなどの管理は不要でフルマネージド、オンデマンドで GPU を利用することができるようになります。
また、GPU を効率的に活用するためには、公式ドキュメント上に公開されている[ベストプラクティス](https://cloud.google.com/run/docs/configuring/services/gpu-best-practices)の内容を押さえておくのが良いでしょう。LLM のような大規模モデルをロードする際の、アプローチごとのトレードオフ、デプロイ時の注意点、インスタンスごとの並行リクエスト数に応じたオートスケーリングなどについて詳細に記載されています。


次のコードが GPU 利用時のコード例ですが、通常の Cloud Run の gcloud コマンドに、 `--gpu`, `--gpu-type` を追加することで GPU を利用できます。
```
  gcloud beta run deploy SERVICE \
    --image IMAGE_URL \
    --project PROJECT_ID \
    --region REGION \
    --port PORT \
    --cpu CPU \
    --memory MEMORY \
    --gpu GPU_NUMBER \
    --gpu-type GPU_TYPE
```


# おわりに
おそらく多くの方が待望の Cloud Run の GPU サポートについて解説しました。利用状況に応じてのスケーリングが可能な Cloud Run において GPU を利用できることで、LLM などを含む API やワークロードのサービングが柔軟に行えることが期待されます。一方で料金体系に少しクセがあったり、効率的に GPU 設定をハンドリングするためのコツがあるので、周辺情報を理解した上で開発などに取り入れるのが良いでしょう。

※ 本記事の情報は2024年8月時点のプレビューの内容です。最新の情報は[公式ドキュメント](https://cloud.google.com/run/docs/configuring/services/gpu)をご確認ください。
