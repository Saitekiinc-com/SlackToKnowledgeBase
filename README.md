# SlackToKnowledgeBase

Slack の `/knowledge` コマンドから入力した内容を Markdown ファイルとして自動保存するリポジトリです。

## フォルダ構成

```
knowledge/   ← Slack から登録したナレッジが保存される
.github/workflows/save-knowledge.yml  ← 自動保存ワークフロー
```

## 使い方

1. Slack で `/knowledge` と入力
2. フォームにタイトル・カテゴリ・本文・タグを入力
3. 送信すると `knowledge/` フォルダに Markdown ファイルが自動作成される
