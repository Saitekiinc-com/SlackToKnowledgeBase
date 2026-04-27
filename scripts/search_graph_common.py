"""
search_graph_common.py
Neo4j ナレッジグラフ検索の共通処理モジュール

search_graph_slack.py / search_graph_github.py から import して使用する。

提供する機能:
  - load_prompt()         : prompts/ から .md プロンプトを読み込む
  - interpret_query()     : LLM でユーザークエリを構造化（Step 1）
  - search_neo4j()        : Neo4j Cypher でナレッジを検索・スコアリング（Step 2）
  - summarize_as_article(): 検索結果を LLM で記事形式に整形（Step 3.5）
"""

import os
import json
import re

from openai import OpenAI
from neo4j import GraphDatabase

# ── クライアント初期化 ────────────────────────────────────────────
client_oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

# GitHub リポジトリ情報（結果にリンクを含めるため）
GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY", "")


# ─────────────────────────────────────────────────────────────────
# プロンプトローダー
# ─────────────────────────────────────────────────────────────────
def load_prompt(name: str) -> str:
    """
    prompts/ ディレクトリから .md プロンプトファイルを読み込む。
    name: ファイル名（例: "search_article.md"）
    """
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", name)
    with open(path, encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────
# Step 1: LLM によるクエリ解釈
# ─────────────────────────────────────────────────────────────────
def interpret_query(query: str) -> dict:
    """
    ユーザーの自然言語クエリを LLM で構造化する。
    返り値: {"keywords": [...], "tools": [...], "author": str|null}
    """
    prompt = f"""以下のナレッジ検索クエリを解析し、検索に使うキーワードを抽出してください。

ルール:
- keywords: 検索に使う重要キーワードを3〜5個抽出（日本語・英語混在OK）
- tools: ツールやAIモデル名が含まれていればリストで抽出（なければ空配列）
- author: 投稿者名が指定されていれば文字列で抽出（なければnull）

JSON のみ出力してください。
例: {{"keywords": ["RAG", "ドキュメント検索"], "tools": ["LangChain"], "author": null}}

検索クエリ:
{query}
"""

    try:
        res = client_oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        raw = res.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return {
                "keywords": parsed.get("keywords", []),
                "tools": parsed.get("tools", []),
                "author": parsed.get("author"),
            }
    except Exception as e:
        print(f"⚠️ クエリ解釈エラー: {e}")

    # フォールバック: クエリをそのままキーワードとして使用
    return {"keywords": query.split(), "tools": [], "author": None}


# ─────────────────────────────────────────────────────────────────
# Step 2: Neo4j Cypher 検索
# ─────────────────────────────────────────────────────────────────
def search_neo4j(parsed: dict) -> list[dict]:
    """
    構造化クエリを元に Neo4j を検索し、スコア付きの結果を返す。
    検索戦略:
      A) Entity マッチ（スコア: 3点/hit）
      B) テキスト部分一致（スコア: 1点/hit）
      C) Tool マッチ（スコア: 2点/hit）
      D) RELATED_TO による関連ナレッジ（スコア: 0.5点）
    """
    keywords = parsed.get("keywords", [])
    tools = parsed.get("tools", [])
    author = parsed.get("author")

    if not keywords and not tools and not author:
        return []

    scored: dict[str, dict] = {}  # id -> {knowledge data + score}

    with driver.session() as session:

        # ── A) Entity マッチ ──────────────────────────────────────
        for kw in keywords:
            result = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.id) CONTAINS toLower($kw)
                WITH e
                MATCH (k:Knowledge)-[:CONTAINS_ENTITY]->(e)
                WHERE k.status = 'active'
                RETURN DISTINCT
                    k.id AS id, k.title AS title, k.why AS why,
                    k.how AS how, k.result AS result, k.author AS author,
                    k.date AS date, k.file_path AS file_path,
                    collect(e.id) AS matched_entities
            """, kw=kw)
            for r in result:
                kid = r["id"]
                if kid not in scored:
                    scored[kid] = _record_to_dict(r, score=0)
                scored[kid]["score"] += 3 * len(r["matched_entities"])
                scored[kid]["matched_entities"].extend(r["matched_entities"])

        # ── B) テキスト部分一致 ───────────────────────────────────
        for kw in keywords:
            result = session.run("""
                MATCH (k:Knowledge)
                WHERE k.status = 'active'
                  AND (
                    toLower(k.title)  CONTAINS toLower($kw) OR
                    toLower(k.why)    CONTAINS toLower($kw) OR
                    toLower(k.how)    CONTAINS toLower($kw) OR
                    toLower(k.result) CONTAINS toLower($kw)
                  )
                RETURN
                    k.id AS id, k.title AS title, k.why AS why,
                    k.how AS how, k.result AS result, k.author AS author,
                    k.date AS date, k.file_path AS file_path
            """, kw=kw)
            for r in result:
                kid = r["id"]
                if kid not in scored:
                    scored[kid] = _record_to_dict(r, score=0)
                scored[kid]["score"] += 1

        # ── C) Tool マッチ ────────────────────────────────────────
        for tool in tools:
            result = session.run("""
                MATCH (k:Knowledge)-[:USES_TOOL]->(t:Tool)
                WHERE k.status = 'active'
                  AND toLower(t.id) CONTAINS toLower($tool)
                RETURN
                    k.id AS id, k.title AS title, k.why AS why,
                    k.how AS how, k.result AS result, k.author AS author,
                    k.date AS date, k.file_path AS file_path,
                    t.id AS matched_tool
            """, tool=tool)
            for r in result:
                kid = r["id"]
                if kid not in scored:
                    scored[kid] = _record_to_dict(r, score=0)
                scored[kid]["score"] += 2
                if r.get("matched_tool"):
                    scored[kid].setdefault("matched_tools", [])
                    scored[kid]["matched_tools"].append(r["matched_tool"])

        # ── Author フィルタ ───────────────────────────────────────
        if author:
            scored = {
                kid: data for kid, data in scored.items()
                if data.get("author") and author.lower() in data["author"].lower()
            }

        # ── D) RELATED_TO で関連ナレッジを追加 ───────────────────
        direct_ids = list(scored.keys())
        if direct_ids:
            result = session.run("""
                MATCH (k:Knowledge)-[r:RELATED_TO]->(other:Knowledge)
                WHERE k.id IN $ids
                  AND other.status = 'active'
                  AND NOT other.id IN $ids
                RETURN DISTINCT
                    other.id AS id, other.title AS title, other.why AS why,
                    other.how AS how, other.result AS result, other.author AS author,
                    other.date AS date, other.file_path AS file_path,
                    r.score AS relation_score
            """, ids=direct_ids)
            for r in result:
                kid = r["id"]
                if kid not in scored:
                    scored[kid] = _record_to_dict(r, score=0.5)
                    scored[kid]["is_related"] = True

        # ── 各ナレッジに紐づく Entity / Tool を取得 ──────────────
        all_ids = list(scored.keys())
        if all_ids:
            # Entity
            result = session.run("""
                MATCH (k:Knowledge)-[:CONTAINS_ENTITY]->(e:Entity)
                WHERE k.id IN $ids
                RETURN k.id AS kid, collect(e.id) AS entities
            """, ids=all_ids)
            for r in result:
                if r["kid"] in scored:
                    scored[r["kid"]]["entities"] = r["entities"]

            # Tool
            result = session.run("""
                MATCH (k:Knowledge)-[:USES_TOOL]->(t:Tool)
                WHERE k.id IN $ids
                RETURN k.id AS kid, collect(t.id) AS tools
            """, ids=all_ids)
            for r in result:
                if r["kid"] in scored:
                    scored[r["kid"]]["tools"] = r["tools"]

    # スコア順ソート
    results = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
    return results[:10]  # 上位10件


def _record_to_dict(record, score: float = 0) -> dict:
    """Neo4j レコードを辞書に変換"""
    return {
        "id": record["id"],
        "title": record.get("title", ""),
        "why": record.get("why", ""),
        "how": record.get("how", ""),
        "result": record.get("result", ""),
        "author": record.get("author", ""),
        "date": record.get("date", ""),
        "file_path": record.get("file_path", ""),
        "score": score,
        "matched_entities": [],
        "matched_tools": [],
        "entities": [],
        "tools": [],
        "is_related": False,
    }


# ─────────────────────────────────────────────────────────────────
# Step 3.5: LLM による記事生成
# ─────────────────────────────────────────────────────────────────
def _generate_article(prompt: str) -> dict | None:
    """記事生成プロンプトを LLM に送り、{title, body} を返す。失敗時は None。"""
    try:
        res = client_oai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.7,
        )
        raw = res.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"⚠️ 記事生成エラー: {e}")
    return None


def review_article(article: dict) -> list[str]:
    """
    生成記事を prompts/search_article_review.md のプロンプトでレビューする。
    返り値: 指摘リスト（問題なければ空リスト）
    """
    review_template = load_prompt("search_article_review.md")
    prompt = (
        review_template
        .replace("{title}", article.get("title", ""))
        .replace("{body}", article.get("body", ""))
    )
    try:
        res = client_oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.1,
        )
        raw = res.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            if result.get("ok"):
                return []
            return result.get("issues", [])
    except Exception as e:
        print(f"⚠️ レビューエラー（スキップ）: {e}")
    return []  # レビュー失敗時はそのまま通す


def summarize_as_article(results: list[dict], query: str, output_format: str) -> dict:
    """
    LLM を使い、Neo4j 検索結果を記事（ブログ）形式に整形する。
    生成後に自動レビューを実行し、問題があれば 1 度だけ再生成する。
    output_format: "slack" or "github"
    返り値: {"title": str, "body": str}
    """
    prompt_template = load_prompt("search_article.md")

    # ナレッジリストを文字列化
    knowledge_text = ""
    for i, r in enumerate(results[:5], 1):
        knowledge_text += f"\n### ナレッジ {i}: {r.get('title', '')}\n"
        knowledge_text += f"- 背景: {r.get('why', '')}\n"
        knowledge_text += f"- 手段: {r.get('how', '')}\n"
        knowledge_text += f"- 結果: {r.get('result', '')}\n"

    length_hint = "300〜500字" if output_format == "slack" else "500〜800字"

    prompt = (
        prompt_template
        .replace("{query}", query)
        .replace("{knowledge_list}", knowledge_text)
        .replace("{length_hint}", length_hint)
    )

    # 生成 → レビュー → 問題あれば1回再生成
    article = _generate_article(prompt)
    if article:
        issues = review_article(article)
        if issues:
            print(f"📝 レビュー指摘あり（再生成）: {issues}")
            retried = _generate_article(prompt)
            if retried:
                article = retried
                print("✅ 再生成完了")
        else:
            print("✅ レビューOK")
        return article

    # フォールバック: 最高スコアナレッジのタイトルと why を返す
    print("⚠️ 記事生成に失敗しました。フォールバック結果を使用します。")
    top = results[0] if results else {}
    return {
        "title": top.get("title", query),
        "body": top.get("why", "（記事生成に失敗しました）"),
    }
