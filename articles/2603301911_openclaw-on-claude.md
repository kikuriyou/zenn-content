---
title: "OpenClawの必要な部分だけをClaude Codeで再現する"
emoji: "🍀"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["openclaw", "claude", "mcp"]
published: false
---

## はじめに

自律型AIエージェントの[OpenClaw](https://github.com/openclaw/openclaw)、個人的にも壁打ち相手として気に入っていたのですが、実際に使っていくと、OpenClaw固有の設定体系やツール群の理解に時間がかかったり、モデル利用にAPIキーが必要であるために費用が気になってきます。

そこで、OpenClawの設計思想をClaude Codeや標準エコシステム上で必要な機能だけを再現するというアプローチを試みています。Claude Codeには CLAUDE.md、Skills、MCPなど、柔軟にカスタマイズするための仕組みが揃っています。これらはAnthropicの公式ドキュメントやコミュニティの情報が豊富で、新たにわかったことをそのまま他のプロジェクトにも転用できます。また、サブスクリプションプラン（Pro/Max）内で利用できるため、API費用を気にせずSonnetやOpusを使えます。

このような前提のもと、この記事では最小構成でOpenClaw的なAIアシスタントをClaude Code上に構築する方法を紹介します。

リポジトリ: https://github.com/kikuriyou/claw-on-claude

## 必要な機能の絞り込みと再現方法一覧

OpenClawは既存のチャットツールと比べて多くの便利な特徴や機能があるので、まずはじめに自分に必要な用途や機能を洗い出す・絞ることが必要です。自分の用途や希望としては主に下記が多いです。

- アイデアの壁打ちで、Web検索など積極的にツール利用してもらい自由に情報収集してもらう（積極的な人格、様々なツールの利用、ブラウザ操作）
  - 情報収集にはXでの情報収集を含み、ブラウザのログイン済みセッションでのブラウザ操作が必要になります
- 時間が経った過去のことを唐突に出すのではなく直近の会話を自然に重視してほしい（会話履歴の管理、定期タスク実行）
- スマホからチャットしたい（外部からのチャット接続）

よくある用途のうち自分があまり使わない、優先度が低いものは以下のようになります。

- 開発関連の指示（IDEなどを使う）
- 複数チャットツールへの対応（個人利用のため）
- 発話による指示（あまり行わない）

このような整理のもと、次のように機能を絞り、Claude Codeで再現を試みました。

| 必要な機能 | Claude Codeでの再現方法 |
|---|---|
| 積極的な人格 | SOUL.md / IDENTITY.mdを転用してCLAUDE.mdで読み込み |
| 日次+重要事項で会話履歴の管理 | 2層メモリ（日次ログ + 長期記憶）を転用|
| 定期タスク実行・報告 | CronCreate + HEARTBEAT |
| 様々なツールの利用 | MCP, Skills, Claude純正機能 |
| ブラウザ操作 | `claude --chrome` |
| スマホからの接続 | Discord Channel / Remote Control |

優先度の低い用途をふまえて、マルチチャンネル対応（Slack等）、音声インターフェース（TTS/STT）といった機能は個人的にはあまり使わないため対象外としました。長期記憶の検索機能は最後まで迷いましたが、今後必要性を感じた時点で追加するものとして今回はスコープ外としています。以降、各機能の実装を見ていきます。

## 積極的な人格

OpenClawの `SOUL.md` / `IDENTITY.md` による人格定義の仕組みをそのまま転用しています。Claude Codeではセッションごとにコンテキストがリセットされますが、`CLAUDE.md` で起動時にこれらを読み込む指示を書いておくことで同じ効果が得られます。

## 会話履歴の管理と定期タスク実行

OpenClawでは日次ログと長期記憶の2層構造で会話記録を管理しており、この設計をそのまま採用しています。日次ログ (`.claw/memory/YYYY-MM-DD.md`) は日毎の会話内容をそのまま記録する append-only のファイルで、長期記憶 (`.claw/MEMORY.md`) には日次ログから重要事項だけを蒸留して保持します。また、OpenClawには定期的にメモリをフラッシュする仕組みがあります。これらを、Claude Codeの `CronCreate`（セッション内で定期的にタスクを実行する機能）を使って、以下の2つの定期実行を設定することで模擬しています。

```
CLAUDE.md で定義しているジョブ:

| Job            | Schedule          | What it does                                    |
|----------------|-------------------|-------------------------------------------------|
| Memory Flush   | 7 */3 * * *       | 直近の会話から重要事項を日次ログに追記          |
| Daily Summary  | 5 0 * * *         | 前日の会話をサマリ化して保存、長期記憶に昇格    |
```

CronCreate はセッション終了で消えるため、この運用はセッションをつけっぱなしにすることを前提としていて、起動時に毎回再登録するようにしています（CLAUDE.md の Session Startup で指示）。また、CronCreate の設定は7日で期限切れになるため、毎日深夜に全ジョブを削除→再登録する「Health Check」ジョブを設けて期限をリセットしています。

```
### CronCreate Health Check (毎日 04:03)
- Schedule: `3 4 * * *`
- 全ジョブを CronDelete → HEARTBEAT.md を読んで全ジョブを CronCreate で再登録
  （この Health Check 自体も含む）
```

OpenClawの特徴的な機能のひとつが、AIが自発的にタスクを実行して報告する動作です。これをClaude Codeで再現するために、`HEARTBEAT.md` と `CronCreate` を組み合わせています。`HEARTBEAT.md` がジョブの定義を一元管理し、セッション起動時に `CronCreate` でそれらを登録するという関係です。先述のメモリフラッシュに加え、以下のようなジョブを定義しています。これによりGmailカレンダー、Xなどの最新状況を定期的な報告するようにできます。

```markdown
# HEARTBEAT.md から抜粋

### Morning Briefing (毎朝 05:00)
- Schedule: `3 5 * * *`
- Gmail未読メール一覧、今日のカレンダー予定、天気を報告

### X Timeline Check (毎朝 09:00)
- Schedule: `3 9 * * *`
- タイムラインの注目投稿とトレンドをサマリ報告
```

## 様々なツールの利用

OpenClawは独自のツール群を持っていますが、ここでは代わりにClaude Codeの標準エコシステムをそのまま活用しています。ファイル操作・Web検索・CronCreate・Agentといった Claude Code 純正機能はそのまま使えます。外部サービス連携には Skills と MCP を活用しています。Skills はGoogle Workspace（Gmail, Calendar）や画像生成などを登録しています。MCP は `.mcp.json` で接続設定するだけで外部サービスとの連携が可能です。

ブラウザ操作には [`claude --chrome`](https://code.claude.com/docs/en/chrome) を使います（[Claude in Chrome](https://chromewebstore.google.com/detail/claude-in-chrome/fcoeoabgfenejglbffodgkkbkcdhcgfn) 拡張機能が必要）。Playwright MCP 等はログイン済みのChromeセッションを引き継げませんが、`claude --chrome` はそのまま引き継いで操作できるため、Gmail や Calendar といった認証が必要なサービスの操作に適しています。ただし headless では動作しないため、Chrome は常時起動しておく必要があります。

## スマホからの接続

OpenClawではDiscordやSlack等のチャットプラットフォームとの連携が組み込まれています。Claude Codeでは [Channels](https://code.claude.com/docs/en/channels) 機能で以下のようなコマンドで起動することで、セッションが起動している限り、Discord Bot として接続してスマホなどから利用できます。

```bash
claude --channels plugin:discord@claude-plugins-official --chrome
```

Claude Code には [Remote Control](https://code.claude.com/docs/en/remote-control) 機能もあり、Claudeアプリからブラウザやスマホで接続できますが、現在はDiscordの方が接続が安定しているように感じるので、Discordをメインで使っています。

## まとめと感想

OpenClawのかわりに、自分に必要な機能だけをClaude Codeの標準エコシステム上で再現するというアプローチを試しました。

- セキュリティやツール利用がClaude Codeの標準に乗るため、情報が豊富でキャッチアップしやすい。新しく学んだことも他で転用できる
- サブスクプラン内で完結するため、API費用を気にせずSonnet/Opusが使える
- OpenClawの良さ（人格の一貫性、記憶の持続、プロアクティブ動作）はきちんと享受できる

といったメリットがある一方で、Claude Code側の機能追加が速く（Remote Control、Channels等）、OpenClawの各機能が今後もネイティブに吸収されていく印象があります。現時点ではこの構成が一番シンプルな出発点だと考えており、ネイティブ機能やOSSの動向を追いながら随時アップデートしていく予定です。

リポジトリ: https://github.com/kikuriyou/claw-on-claude
