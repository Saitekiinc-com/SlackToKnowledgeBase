# SlackToKnowledgeBase

Slack から投稿された AI 活用ナレッジを GitHub リポジトリに蓄積し、Neo4j（グラフデータベース）で関連性を管理・検索できるナレッジ管理システムです。

## 主要機能

### 1. ナレッジ投稿機能
Slack のスラッシュコマンド（`/knowledge`）からモーダルを開き、入力されたナレッジを Markdown ファイルとして `knowledge/` に保存し、PR を自動作成します。
👉 [詳細仕様](./docs/spec/ナレッジ投稿機能.md)

### 2. ナレッジグラフ登録機能
`knowledge/` 配下の Markdown が main ブランチにマージされると、LLM でキーワードを抽出し、Neo4j にナレッジ間の関連グラフを自動構築します。
👉 [詳細仕様](./docs/spec/ナレッジグラフ登録機能.md)

### 3. ナレッジ検索機能
Slack（`/knowledge-search`）や GitHub Issue コメント（`/search`）から自然言語で検索し、Neo4j + LLM による RAG で回答を生成・返信します。
👉 [詳細仕様](./docs/spec/ナレッジ検索機能.md)

### 4. ナレッジ陳腐化チェック処理機能
週次バッチで投稿日からの経過日数を判定し、確認対象への移動（Slack 通知）・無効化（Neo4j データ削除）を自動実行します。
👉 [詳細仕様](./docs/spec/ナレッジ陳腐化チェック処理機能.md)

> 📖 全機能の一覧は [docs/spec/機能一覧.md](./docs/spec/機能一覧.md) を参照してください。

## フォルダ構成

```
knowledge/                ← 有効なナレッジ（Markdown）
check/                    ← 確認対象ナレッジ（CHECK_DAYS 超過分）
invalid/                  ← 無効化ナレッジ（INVALID_DAYS 超過分）
  └── list/               ← 無効化リスト（週次で更新）
scripts/                  ← Python 処理スクリプト
prompts/                  ← LLM プロンプトテンプレート
docs/                     ← 設計ドキュメント
  └── spec/               ← 機能仕様・環境設定・データフロー
work/                     ← Cloudflare Worker ソース
.github/workflows/        ← GitHub Actions ワークフロー
  ├── save-knowledge.yml          ナレッジ投稿
  ├── register-graph.yml          ナレッジグラフ登録
  ├── search-knowledge-slack.yml  ナレッジ検索（Slack）
  ├── search-knowledge-github-issue.yml  ナレッジ検索（GitHub Issue）
  └── check-knowledge.yml         陳腐化チェック
```

## 使い方

### ナレッジの投稿
1. Slack のメッセージ入力欄で `/knowledge` を実行
2. 表示されたモーダルに入力して送信
3. 自動で PR が作成されるので、レビュー後にマージ

## 入力項目

| 項目 | 必須 | 内容 |
|---|---|---|
| タイトル (title) | ○ | ナレッジのタイトル（日本語対応） |
| 背景、課題 (why) | ○ | ナレッジの背景や解決したい課題 |
| 手段 (how) | ○ | 解決のためのプロンプトや各種アクション |
| 結果 (result) | ○ | 実行した結果や得られた知見 |
| ツールやAIのモデル (tools) | ○ | 使用したAIモデルや利用ツール |

※ 投稿者(`author`)と登録日(`date`)はシステム側で自動付与されます。

## 保存フォーマット

送信すると `knowledge/` フォルダに以下の形式で Markdown ファイルが自動生成されます。

- **ファイル名**: `YYYY-MM-DD-HHMMSS-タイトル.md`（日本語タイトルをそのまま使用）
- **タイムゾーン**: JST（日本標準時）

```markdown
---
date: "2026-03-05"
title: "ナレッジのタイトル"
author: "著者名"
---

# ナレッジのタイトル

## 背景、課題
背景や課題の内容がここに入ります。

## 手段
手段の内容がここに入ります。

## 結果
結果がここに入ります。

## ツールやAIのモデル
使用したツールがここに入ります。
```

### ナレッジの検索
- **Slack**: `/knowledge-search` でモーダルを開き、検索キーワードを入力
- **GitHub Issue**: Issue のコメントに `/search 検索キーワード` と投稿

## 環境設定

本システムの運用には、GitHub・Cloudflare Workers・Slack の3つのプラットフォームで環境変数やアプリ設定が必要です。

### Repository Secrets（GitHub）

| Name | 説明 |
|---|---|
| `OPENAI_API_KEY` | OpenAI API キー |
| `NEO4J_URI` | Neo4j の接続先 URI |
| `NEO4J_USERNAME` | Neo4j のユーザー名 |
| `NEO4J_PASSWORD` | Neo4j のパスワード |
| `SLACK_WEBHOOK_URL` | Slack Incoming Webhook URL |
| `AUTHOR_SLACK_MAP` | 投稿者名→Slack User ID の JSON マッピング |

### Repository Variables（GitHub）

| Name | 説明 | デフォルト値 |
|---|---|---|
| `KNOWLEDGE_CHECK_DAYS` | `knowledge/` → `check/` 移動の日数閾値 | `30` |
| `KNOWLEDGE_INVALID_DAYS` | `check/` → `invalid/` 移動の日数閾値 | `90` |

> 📖 Cloudflare Workers の環境変数や Slack アプリの設定を含む、すべての環境設定の詳細は [docs/spec/環境設定.md](./docs/spec/環境設定.md) を参照してください。

## システム連携アーキテクチャ

本システムは以下の3つの外部サービスと連携して動作します。

- **Slack**: ユーザーインターフェース（投稿・検索の入力と結果の受信）
- **Cloudflare Worker**: Slack と GitHub Actions の中継役（`/work/cloudflare-worker-v2.js`）
- **GitHub Actions**: Markdown の保存、Neo4j へのグラフ登録/検索/削除処理
