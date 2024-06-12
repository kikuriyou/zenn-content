# Zenn Contents

[✍️ How to use](https://zenn.dev/zenn/articles/zenn-cli-guide)

# How to deploy

```sh
# 記事の作成
$ docker-compose up -d zenn-new-article

# プレビューの表示
$ docker-compose up zenn-preview

# デプロイ
$ bash deploy.sh
```

# cf.

- [Githubとの連携方法](https://zenn.dev/zenn/articles/connect-to-github)
- [Zenn CLIの使い方](https://zenn.dev/zenn/articles/install-zenn-cli)
- [ZennのMarkdown記法](https://zenn.dev/zenn/articles/markdown-guide)
- Markdownへ画像リンクを挿入したい場合は[ここ](https://zenn.dev/dashboard/uploader)から画像をアップロードして使う
```md
![altテキスト](https://画像のURL)

![altテキスト](https://画像のURL =250x)
```