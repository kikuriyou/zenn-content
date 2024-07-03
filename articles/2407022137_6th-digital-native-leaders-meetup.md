---
title: "第6回 Digital Native Leader’s Meetup に参加しました"
emoji: "🍀"
type: "idea" # tech: 技術記事 / idea: アイデア
topics: ["googlecloud", "google"]
published: false
publication_name: hogeticlab
---

こんにちは、Hogetic Lab の喜久里です。先日、Google さん主催の Digital Native Leader's Meetup に参加してきました。
Meetup 初心者の私は最初は様子を伺いながらの参加でしたが、参加しやすい雰囲気作りが素晴らしく、結果としてとても楽しむことができました。
私と同じようにMeetup参加に少し不安を感じている方に向けて、イベントの様子をレポートしたいと思います。


# Digital Native Leader’s Meetup とは
Digital Native Leader's Meetup は、Google Cloud を利用するエンジニア向けの交流イベントです。普段はあまり接点のない様々な企業のエンジニア同士がオフラインで集まり、ネットワーキングや情報交換を通じて、お互いのプロダクト開発を加速させることを目的としています。
今回のイベントでは Google さんとのNDAの上で1社1名までが参加条件になっており、次のようなコンテンツを通して参加者同士で技術的な課題や情報共有を行える場として開催されています。
- ユーザー企業事例に関する Lightning Talk
- Shane Gu 氏の話題提供
- 参加者同士のアンカンファレンス
- ネットワーキング


# イベント内容
## ユーザー企業による Lightning Talk
最初のコンテンツはレバレジーズさんによるLTで、Gemini Batch Prediction の利用事例が紹介されました。LLM のAPIで大量データに対して処理を行う場合、逐次処理では時間がかかり、並行処理ではrate limitをケアする必要が出てきます。こういったケースに対処するためのサービスが Gemini Batch Prediction で、リアルタイム性が求められないケースであれば rate limit なしで処理を行うことができます。処理時間については、逐次処理のおよそ半分の時間で処理が行えることが紹介されていました。
弊社でも BigQuery上のデータに対して分析を行いたいケースやデータパイプラインの処理においてAPIのrate limitが気になることがあったので、こういったケースにおいて Gemini Batch Predictionを活用できそうに感じました。

## Shane Gu 氏による話題提供
[Shane Gu](https://x.com/shaneguml) 氏は、GeminiとGPT両方の開発に携わった経験を持つ研究者です。今回の発表内容の公開はNGになっている詳細には触れませんが、生成AIの最近の状況の見方や今後の展望について興味深く拝聴しました。

## 参加者によるアンカンファレンス
続いて参加者同士で行うアンカンファレンスです。アンカンファレンスは参加者が3,4人ずつのグループに分かれて議論を深めていく形式ですが、初対面の人と率直な意見を交わすのは誰でも多少の勇気がいるものです。このアンカンファレンスでは各グループにGoogle社員が1人ずつ入って自然な形で会話を繋いでくれたので、とても話しやすかったです。おかげでどのグループも活発な議論が行われていたように思います。

ディスカッションでは、参加者それぞれの業務や視点をふまえて、LLM (大規模言語モデル) のユースケースや私見、Google Cloud の各サービスの活用方法や成功事例などを情報交換しました。普段、社内以外のAI/MLエンジニアと関わる機会も多くないので、様々な領域でのAI/MLの活用事例を聞けたのは刺激的な時間となりました。

さらに嬉しいことに、私のいたグループに Shane Gu氏も途中から加わり、公共の場では話しづらい話をいくつもさせていただきました。普段はニュースやX(旧Twitter)など拝見するばかりですが、実際にお話ししてみると気さくな `おしゃべり研究者` という印象で、「googleは難しいことも簡単なことも同じ速さでやる」というジョークが色々な意味で個人的にツボだったのでした。

## ネットワーキング
アンカンファレンスの後はグループの枠を超えた自由な交流の時間が設けられました。ここでは、日頃取り組んでいることにとどまらず、今後取り組みたい(が追いついていない)ことについても、その道のプロフェッショナルの方と情報交換することができ、短い時間ではありましたが、今後の活動の大きな励みになる大変有意義な時間となりました。


# おわりに
冒頭にもあるように私はGoogleさんが主催するMeetupに初めて参加したもので、周囲の様子をうかがいながら参加していたのですが、蓋を開けてみると思ったより近しい方々が参加されていることがわかり、さらには様々なお話をさせていただくことができて、刺激のある貴重な時間になりました。その過程では主催のGoogleさんの多大な配慮があったことは言うまでもないですが、初めて参加する方にもおすすめできるものと感じました。
この記事が、今後の Digital Native Leader's Meetup への参加を検討されている方の参考になれば幸いです。


TODO:
- 写真追加したい
  - ポーチ、YouTube Retro Duffel
  - 参加者がたくさんうつっている全体の写真(iwakiさん相談)
  - ごはん(iwakiさん相談)
- iwakiさんにレビュー