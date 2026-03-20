---
title: "OpenClawを規約違反なく使いたい — 各社の対応と認証方式の整理（2026年3月）"
emoji: "🍀"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["openclaw", "antigravity", "claude", "openai", "kiro"]
published: true
---

## はじめに

自律型AIエージェントである[OpenClaw](https://openclaw.ai/)はパワフルで技術的にも興味深いことが多く、個人的にも色々試したり調査していたところ、2026年2月頃にAnthropicやGoogleがOpenClaw経由のアクセスを規約違反として対処し始めたというニュースや発表が相次ぎました。アカウント停止のリスクは避けたいため、どのプロバイダーでどのような使い方が許容されるのかを一通り調べることにしました。この記事はその備忘録であり、同じ課題感を持つ方の参考になれば幸いです。

:::message
本記事は2026年3月時点の内容です。各社の規約・対応は変化が速いため、利用前に必ず公式ドキュメントを確認してください。また、この調査はAIを活用した情報収集を含むため、誤りが混入している可能性があります。お気づきの点はご一報いただけると幸いです。重要な判断の前には一次情報を必ずご確認ください。
:::


## 各社の対応：一覧表と要点

OpenClawから各サービスをサブスクリプションのOAuth認証で使うことへの各社の対応をまとめます。

| サービス | OAuth流用 | 備考 |
|---|---|---|
| Anthropic（Claude） | ❌ 規約違反・BAN実例あり | 2026年1月から技術的ブロック、2月に公式明文化 |
| Google Antigravity | ❌ 規約違反・即時BAN・返金なし | 対応が最も苛烈。Gmailまで停止されたケースも |
| Google Gemini CLI | ❌ 規約違反 | 無料枠・AI Pro・AI Ultraいずれも対象。公式ドキュメントにOpenClawが名指しで記載 |
| OpenAI Codex | ✅ 公式に許可 | ChatGPT Plus/ProのOAuthが外部ツールで明示的に使用可 |
| GitHub Copilot | ✅ 公式SDK対応済み | Claude・GPT・Geminiなど複数モデルが利用可能 |
| AWS Kiro | ✅ 無料・フラットレート | AWS Builder IDで認証。Claude Sonnet等が利用可 |
| iFlow | ✅ 無料・無制限 | Kimi K2.5・GLM・MiniMax等のOAuthアクセスが可 |

### 各社の補足

**Anthropic** は2026年1月9日に技術的なブロックを静かに実施し、[同年2月に公式ドキュメントで明文化](https://www.theregister.com/2026/02/20/anthropic_clarifies_ban_third_party_claude_access/)しました。「Consumer ToS上、OAuth認証はClaude CodeとClaude.aiにのみ使用可能」というものです。アカウントは即時停止ではなく、[ブロック時にエラーメッセージを返す](https://blog.devgenius.io/you-might-be-breaking-claudes-tos-without-knowing-it-228fcecc168c)形での対応でした。なお、ドキュメント更新直後に混乱が生じた際、[AnthropicはAgent SDKの利用まで禁じているわけではないと補足説明](https://thenewstack.io/anthropic-agent-sdk-confusion/)しています。APIキー（従量課金）を使った接続は規約上問題ありません。

**Google** の対応はより苛烈でした。[2026年2月12日頃から一斉BANが始まり](https://winbuzzer.com/2026/02/23/google-bans-ai-subscribers-openclaw-no-refunds-xcxwbn/)、月額$250のAI Ultraユーザーも例外なく停止されました。返金なし・異議申し立てなし。さらに一部ユーザーはAntigravityだけでなくGmail・Workspaceまで利用不能になっています。その後Googleは[「ToSを知らなかったユーザー向けのアカウント復元プロセスを整備する」と発表](https://piunikaweb.com/2026/03/02/google-account-disabled-openclaw-antigravity-exec/)し軟化しましたが、方針自体は変わっていません。[公式Discussionでの経緯](https://github.com/google-gemini/gemini-cli/discussions/20632)も参考になります。Gemini APIキーまたはVertex AI APIキーを使った従量課金での接続は規約上問題ありません。

**OpenAI** は対照的に、[外部ツールからのOAuth利用を明示的に許可する立場](https://docs.openclaw.ai/concepts/oauth)をとっています。OpenClawの作者であるPeter Steinberger氏をOpenAIが採用したことも、この方針と無関係ではないでしょう。[OpenAI Codexの公式認証ドキュメント](https://developers.openai.com/codex/auth)にも、CLIやIDE拡張でのChatGPTサインインが正式にサポートされている旨が記載されています。

**GitHub Copilot** は[公式の`@github/copilot-sdk`を使ったSDK統合](https://github.com/openclaw/openclaw/pull/4469)がOpenClawに取り込まれており、Claude Opus・GPT-5系・Gemini 3など複数モデルが1つのサブスクリプションで利用できます。[OpenClawの公式ドキュメント](https://docs.openclaw.ai/providers/github-copilot)にセットアップ手順が記載されています。

**AWS Kiro** は[AWS Builder IDで認証するだけでClaude Sonnetなどが無制限・無料で利用できる](https://dev.to/gabrielkoo/max-out-your-openclaw-with-aws-credits-via-kiro-and-nova-models-mdp)プロバイダーです。AWS Creditsとの組み合わせでコストをほぼゼロにする使い方も紹介されています。

**iFlow** は[Kimi K2.5・GLM・MiniMax・DeepSeek R1など8モデル以上にOAuthで無制限アクセスできる](https://docs.openclaw.ai/concepts/model-providers)プロバイダーです。OpenClawのバンドルプロバイダーとしてデフォルトで有効化されています。

**Gemini CLI** の注意点として、「Log in with Google（無料枠）」「Google AI Pro」「AI Ultra」、いずれの認証方式でも外部ツールからの利用は規約違反になります。無料枠なら問題ないのでは、という誤解が生まれやすいため注意が必要です。[Gemini CLIの公式ドキュメント（Terms and Privacy）](https://geminicli.com/docs/resources/tos-privacy/)には次の記述があります。

> Directly accessing the services powering Gemini CLI (e.g., the Gemini Code Assist service) using third-party software, tools, or services (for example, using OpenClaw with Gemini CLI OAuth) is a violation of applicable terms and policies. Such actions may be grounds for suspension or termination of your account.
>
> （訳）サードパーティのソフトウェア・ツール・サービスを使ってGemini CLIの基盤サービス（例：Gemini Code Assistサービス）に直接アクセスすること（例：OpenClawでGemini CLIのOAuthを使用すること）は、適用される利用規約およびポリシーの違反にあたります。このような行為は、アカウントの停止または終了の根拠となる場合があります。

各社の可否の分かれ目を見ると、「**OAuthによるサブスクリプション認証か、APIキーによる従量課金か**」が主な判断軸になっていることがわかります。次章でその仕組みを整理します。


## なぜ問題なのか — OAuthとAPIキーの違い

### 2種類の認証方式

OpenClawをAIプロバイダーに接続する方法には、大きく2種類あります。

**OAuth認証（サブスクリプション流用）**
「Login with Claude」「Login with Google」のようなボタンを押してサインインする方式です。月額固定のサブスクリプションをそのまま利用できます。Claude Maxなら月$200の定額でClaudeを使い放題、という形になります。

**APIキー（従量課金）**
各社のコンソールで発行するAPIキーをOpenClawに設定する方式です。使ったトークン分だけ課金されます。

### なぜOAuth流用が問題になるのか

定額サブスクリプションは、**人間がIDEやチャット画面で対話的に使う**という前提で価格設計されています。

OpenClawのようなエージェントツールでは、一晩中エージェントがループしてコードを書き続けるといった使い方が普通に起きます。消費トークンは桁が変わります。[Claude Maxで月$200のサブスクリプションを使っていたユーザーが、API従量課金に切り替えたところ月$1,000以上になった](https://openclaw.rocks/blog/anthropic-oauth-ban)という事例も複数報告されています。

各社がOAuth流用を問題視するのは、**定額プランがエージェント用途では採算割れになるから**です。[PCWorldの解説記事](https://www.pcworld.com/article/3068842/whats-behind-the-openclaw-ban-wave.html)もこの構造を詳しく説明しています。


## 規約準拠でOpenClawから各サービスを使うには

上記を踏まえて、現実的な選択肢を絞ると以下になります。

**すでにChatGPT Plus/Proを契約している場合**は[Codex OAuthへの切り替え](https://docs.openclaw.ai/concepts/oauth)がもっとも手軽です。OpenClawのセットアップ時にOpenAI Codexを選ぶだけで移行できます。

**コストを抑えたい場合**は[GitHub Copilot](https://docs.openclaw.ai/providers/github-copilot)か[AWS Kiro](https://kiro.dev/)が有力候補です。GitHub Copilotは月$10からで、ClaudeやGeminiなど複数モデルを切り替えられる点が魅力です。AWS KiroはAWS Builder IDで認証でき、現在は無制限で無料となっています。

**Claudeを使いたい場合**は[AnthropicのAPIキー](https://console.anthropic.com/)（従量課金）が唯一の規約準拠な方法です。エージェントのループ回数や使い方次第でコストが膨らむため、上限設定を忘れずに。

### 規約準拠でOpenClawを使うための選択肢

| 方法 | コスト感 | 備考 |
|---|---|---|
| [OpenAI Codex OAuth](https://developers.openai.com/codex/auth) | ChatGPT Plus/Proに含まれる | 最もお手軽な乗り換え先 |
| [GitHub Copilot OAuth](https://docs.openclaw.ai/providers/github-copilot) | 月$10〜 | Claude・GPT・Geminiが使える |
| [AWS Kiro](https://kiro.dev/) | 現在無料 | Claude Sonnetが使える |
| [Anthropic APIキー](https://console.anthropic.com/) | 従量課金 | 使い過ぎに注意 |
| [Gemini APIキー](https://ai.google.dev/gemini-api/docs/rate-limits) | 従量課金 | Gemini CLIのOAuthとは別物 |
| [ローカルモデル（Ollama等）](https://ollama.com/) | ほぼ無料 | 性能とのトレードオフあり |


## おわりに

今回の調査を通じて、各社の対応方針の違いが改めて明確になりました。AnthropicとGoogleが相次いでOAuth流用を規制する一方、OpenAIが開放的な姿勢をとっているのは、それぞれの競争戦略を反映しているようで興味深いです。

OpenClawは面白いツールだと思います。ただ、モデルプロバイダーとの関係においては「どの認証方式を使うか」が利用継続の可否に直結します。各社の規約はまだ変化の途上にあるので、使い始める前に一度ドキュメントを確認することをおすすめします。

### 参考リンク

- [OpenClaw 公式ドキュメント — 認証方式](https://docs.openclaw.ai/concepts/oauth)
- [OpenClaw 公式ドキュメント — モデルプロバイダー一覧](https://docs.openclaw.ai/concepts/model-providers)
- [Anthropic — Claude Code Legal and Compliance](https://www.anthropic.com/legal/claude-code-legal-and-compliance)
- [Gemini CLI — Terms and Privacy](https://geminicli.com/docs/resources/tos-privacy/)
- [Gemini CLI — Quotas and Pricing](https://geminicli.com/docs/resources/quota-and-pricing/)
- [Addressing Antigravity Bans & Reinstating Access — gemini-cli GitHub Discussion](https://github.com/google-gemini/gemini-cli/discussions/20632)
- [What's behind the OpenClaw ban wave — PCWorld](https://www.pcworld.com/article/3068842/whats-behind-the-openclaw-ban-wave.html)
