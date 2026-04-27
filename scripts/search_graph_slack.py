"""
search_graph_slack.py
Slack 向けナレッジ検索スクリプト

search-knowledge-slack.yml から呼び出される。
共通処理（Neo4j 検索・記事生成）は search_graph_common を使用し、
このスクリプトは Slack 向けフォーマットと Webhook 送信に特化する。

環境変数:
  QUERY              : 検索クエリ
  OPENAI_API_KEY     : OpenAI APIキー
  NEO4J_URI          : Neo4j 接続URI
  NEO4J_USERNAME     : Neo4j ユーザー名
  NEO4J_PASSWORD     : Neo4j パスワード
  SLACK_WEBHOOK_URL  : Slack Incoming Webhook URL
  SLACK_CHANNEL_ID   : 送信先チャンネルID（任意）
  GITHUB_REPOSITORY  : リポジトリ名（ファイルリンク生成用）
"""

import os
import json
import sys
import urllib.request

# 共通モジュールを同ディレクトリから import
sys.path.insert(0, os.path.dirname(__file__))
from search_graph_common import (
    driver,
    GITHUB_REPO,
    interpret_query,
    search_neo4j,
    summarize_as_article,
)


# ─────────────────────────────────────────────────────────────────
# Step 3: Slack 向け出力整形
# ─────────────────────────────────────────────────────────────────
def format_slack(results: list[dict], query: str) -> str:
    """
    Slack メッセージ用テキストを生成する（記事形式）。
    結果が 0 件の場合はシンプルなメッセージを返す。
    """
    if not results:
        return f"*🔍 「{query}」*\n\n該当するナレッジが見つかりませんでした。"

    article = summarize_as_article(results, query, "slack")
    title = article.get("title", query)
    body = article.get("body", "")

    # 参照ナレッジリスト（上位5件）
    sources = []
    for r in results[:5]:
        file_path = r.get("file_path", "")
        label = r.get("title", file_path)
        if file_path and GITHUB_REPO:
            sources.append(f"<https://github.com/{GITHUB_REPO}/blob/main/{file_path}|{label}>")
        else:
            sources.append(label)

    lines = [
        f"📖 *{title}*\n",
        body,
        "",
        f"_参照ナレッジ: {' | '.join(sources)}_",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Slack Webhook 送信
# ─────────────────────────────────────────────────────────────────
def post_to_slack(message: str) -> None:
    """Slack Incoming Webhook に検索結果を送信する"""
    slack_webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    channel_id = os.environ.get("SLACK_CHANNEL_ID", "")

    if not slack_webhook:
        print("⚠️ SLACK_WEBHOOK_URL が未設定のため、Slack 投稿をスキップ")
        print(message)
        return

    payload: dict = {"text": message}
    if channel_id:
        payload["channel"] = channel_id

    req = urllib.request.Request(
        slack_webhook,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req)
    print("✅ Slack に検索結果を投稿しました")


# ─────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────
def main():
    query = os.environ.get("QUERY", "").strip()

    if not query:
        print("⚠️ QUERY が空です。")
        return

    print(f"🔍 検索クエリ: {query}")
    print("📤 出力形式: slack")

    # Step 1: クエリ解釈
    parsed = interpret_query(query)
    print(f"📊 解釈結果: {json.dumps(parsed, ensure_ascii=False)}")

    # Step 2: Neo4j 検索
    results = search_neo4j(parsed)
    print(f"📋 検索結果: {len(results)} 件ヒット")
    for r in results:
        print(f"  - [{r['score']:.1f}] {r['title']} ({'関連' if r['is_related'] else '直接'})")

    # Step 3: Slack 向けフォーマット（記事形式）
    message = format_slack(results, query)

    # Slack に送信
    post_to_slack(message)


if __name__ == "__main__":
    try:
        main()
    finally:
        driver.close()
