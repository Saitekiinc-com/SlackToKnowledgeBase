"""
register_graph.py
pushトリガーで起動し、変更された.mdをNeo4jにグラフ登録する

ノード設計:
  Knowledge  : ナレッジ本体（title/why/how/result/tools/author/date）
  Entity     : 本文から抽出したキーワード
  Tool       : 使用ツール・AIモデル
  Author     : 投稿者

エッジ設計:
  CONTAINS_ENTITY : Knowledge → Entity
  USES_TOOL       : Knowledge → Tool
  WRITTEN_BY      : Knowledge → Author
  RELATED_TO      : Knowledge → Knowledge（エンティティ共有スコア>=2）
"""

import os
import sys
import json
import re
import random

import frontmatter
from openai import OpenAI
from neo4j import GraphDatabase

# ── 定数 ──────────────────────────────────────────────────────────
MAX_EXISTING_ENTITIES_IN_PROMPT = 200  # プロンプトに含める既存Entityの上限

# ── クライアント初期化 ────────────────────────────────────────────
client_oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)


# ─────────────────────────────────────────────────────────────────
# ⓪ 既存 Entity 取得
# ─────────────────────────────────────────────────────────────────
def get_existing_entities() -> list[str]:
    """Neo4j から登録済みの全 Entity.id を取得する"""
    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN e.id AS name ORDER BY name")
        return [r["name"] for r in result]


# ─────────────────────────────────────────────────────────────────
# ① .mdのパース
# ─────────────────────────────────────────────────────────────────
def parse_md(file_path: str) -> dict:
    """
    frontmatter（title/date/author）と
    本文セクション（why/how/result/tools）を取得する

    .mdの本文構造（save-knowledge.ymlが生成する形式）:
      ## 背景、課題
      {why}
      ## 手段
      {how}
      ## 結果
      {result}
      ## ツールやAIのモデル
      {tools}
    """
    post = frontmatter.load(file_path)
    meta = post.metadata
    body = post.content

    def extract_section(heading: str, text: str) -> str:
        """## 見出し 〜 次の ## or # までを抽出"""
        pattern = rf"##\s+{re.escape(heading)}\s*\n([\s\S]*?)(?=\n##|\n#\s|\Z)"
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""

    def extract_tools_list(text: str) -> list[str]:
        """ツールセクションからリストアイテム(- xxx)を抽出する"""
        tools_section = extract_section("ツールやAIのモデル", text)
        if not tools_section:
            # 旧形式のセクション名にもフォールバック
            tools_section = extract_section("使用ツール", text)
        if not tools_section:
            return []
        # "- ツール名" の形式を抽出
        items = re.findall(r"^\s*[-*]\s+(.+)", tools_section, re.MULTILINE)
        # リストがなければカンマ区切り or 読点区切りを試す
        if not items:
            items = re.split(r"[、,，]", tools_section)
        return [item.strip() for item in items if item.strip()]

    # frontmatterにtools配列がある場合はそちらを優先、なければ本文から抽出
    tools_from_meta = meta.get("tools", [])
    tools = tools_from_meta if tools_from_meta else extract_tools_list(body)

    return {
        "file_path": file_path,
        # file_pathをidとして使用（スラッシュ・拡張子を正規化）
        "id":        file_path.replace("/", "__").replace(".md", ""),
        "title":     meta.get("title",  ""),
        "date":      str(meta.get("date", "")),
        "author":    meta.get("author", ""),
        # save-knowledge.yml が生成する見出しに合わせる
        "why":       extract_section("背景、課題", body) or extract_section("背景", body),
        "how":       extract_section("手段", body) or extract_section("やり方", body),
        "result":    extract_section("結果", body),
        "tools_raw": extract_section("ツールやAIのモデル", body) or extract_section("使用ツール", body),
        "tools":     tools,
    }


# ─────────────────────────────────────────────────────────────────
# ② エンティティ抽出（LLM） ─ 既存Entityによる語彙統一付き
# ─────────────────────────────────────────────────────────────────
def extract_entities(doc: dict, existing_entities: list[str]) -> list[str]:
    """
    why/how/result/titleを結合してLLMにエンティティ抽出させる。
    既存Entity一覧をプロンプトに含め、語彙の統一を促す。
    """
    combined = " ".join([
        doc["title"],
        doc["why"],
        doc["how"],
        doc["result"],
    ]).strip()

    if not combined:
        return []

    # 既存Entityリストの構築（上限を超えたらランダムサンプリング）
    if len(existing_entities) > MAX_EXISTING_ENTITIES_IN_PROMPT:
        sampled = random.sample(existing_entities, MAX_EXISTING_ENTITIES_IN_PROMPT)
    else:
        sampled = existing_entities

    existing_block = ""
    if sampled:
        entity_list_json = json.dumps(sampled, ensure_ascii=False)
        existing_block = f"""
【重要ルール: 語彙の統一】
以下は既に登録済みのキーワード一覧です。
意味的に同じ、またはほぼ同義のキーワードがこの一覧に存在する場合は、
新しい表記を作らず、既存の表記をそのまま使用してください。
一覧にない新しい概念の場合のみ、新規キーワードとして出力してください。

既存キーワード一覧:
{entity_list_json}
"""

    prompt = f"""以下のAI活用ナレッジから重要キーワードを5〜10個抽出してください。
{existing_block}
抽出の優先順位:
1. AI手法・アプローチ名（例: few-shot, RAG, プロンプトチェーン）
2. ツール・サービス名（例: LangChain, Supabase, Cursor）
3. 目的・用途（例: コード生成, 要約, テスト自動化）
4. 状態・性質（例: 高精度, 低コスト, 日本語対応）

JSON配列のみ出力してください。例: ["few-shot", "プロンプト設計", "ChatGPT"]

テキスト:
{combined[:800]}
"""

    try:
        res = client_oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        raw = res.choices[0].message.content.strip()
        # JSONの前後にテキストが混入した場合に備えて抽出
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        # GitHub Actions のエラーアノテーションとして出力
        print(f"::error::エンティティ抽出失敗（OpenAI API）: {e}")
        print(f"⚠️ エンティティ抽出エラー: {e}")
        raise  # 呼び出し元でキャッチしてエラーカウント

    return []


# ─────────────────────────────────────────────────────────────────
# ③ Neo4j登録
# ─────────────────────────────────────────────────────────────────
def register_to_neo4j(doc: dict, entities: list[str]):
    with driver.session() as session:

        # ── Knowledgeノード ──────────────────────────────────────
        session.run("""
            MERGE (k:Knowledge {id: $id})
            SET
                k.title     = $title,
                k.why       = $why,
                k.how       = $how,
                k.result    = $result,
                k.tools_raw = $tools_raw,
                k.author    = $author,
                k.date      = $date,
                k.file_path = $file_path,
                k.status    = 'active'
        """,
            id=doc["id"],
            title=doc["title"],
            why=doc["why"],
            how=doc["how"],
            result=doc["result"],
            tools_raw=doc["tools_raw"],
            author=doc["author"],
            date=doc["date"],
            file_path=doc["file_path"],
        )
        print(f"  ✅ Knowledgeノード: {doc['title']}")

        # ── Authorノード + WRITTEN_BYエッジ ─────────────────────
        if doc["author"]:
            session.run("""
                MERGE (a:Author {id: $author})
                WITH a
                MATCH (k:Knowledge {id: $id})
                MERGE (k)-[:WRITTEN_BY]->(a)
            """,
                author=doc["author"],
                id=doc["id"],
            )
            print(f"  ✅ Authorノード: {doc['author']}")
        else:
            print("  ⚠️ Author が空のためスキップ")

        # ── Toolノード + USES_TOOLエッジ ────────────────────────
        tools = doc["tools"] if doc["tools"] else []
        for tool in tools:
            if not tool:
                continue
            session.run("""
                MERGE (t:Tool {id: $tool})
                WITH t
                MATCH (k:Knowledge {id: $id})
                MERGE (k)-[:USES_TOOL]->(t)
            """,
                tool=str(tool),
                id=doc["id"],
            )
        if tools:
            print(f"  ✅ Toolノード: {tools}")
        else:
            print("  ⚠️ Tool が空のためスキップ")

        # ── Entityノード + CONTAINS_ENTITYエッジ ────────────────
        for entity in entities:
            if not entity:
                continue
            session.run("""
                MERGE (e:Entity {id: $name})
                WITH e
                MATCH (k:Knowledge {id: $id})
                MERGE (k)-[:CONTAINS_ENTITY]->(e)
            """,
                name=str(entity),
                id=doc["id"],
            )
        if entities:
            print(f"  ✅ Entityノード: {entities}")
        else:
            print("  ⚠️ Entity が空のためスキップ（OpenAI APIエラーの可能性あり）")

        # ── RELATED_TOエッジ（エンティティ共有数>=2で自動生成） ──
        result = session.run("""
            MATCH (k:Knowledge {id: $id})-[:CONTAINS_ENTITY]->(e:Entity)
                  <-[:CONTAINS_ENTITY]-(other:Knowledge)
            WHERE other.id <> $id
              AND other.status = 'active'
            WITH k, other, count(e) AS shared
            WHERE shared >= 2
            MERGE (k)-[r:RELATED_TO]->(other)
            SET r.score = shared
            RETURN other.title AS related_title, shared
        """,
            id=doc["id"],
        )
        related = result.data()
        if related:
            for r in related:
                print(f"  🔗 RELATED_TO: {r['related_title']} (共有エンティティ: {r['shared']})")
        else:
            print("  ℹ️ RELATED_TOエッジ: 該当なし（共有エンティティ不足）")


# ─────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────
def main():
    # パイプ区切りでファイルパスを分割（スペース入りファイル名対応）
    raw_files = os.environ.get("CHANGED_FILES", "")
    changed_files = [f for f in raw_files.split("|") if f.strip()]
    target_files  = [f for f in changed_files if f.endswith(".md") and f.startswith("knowledge/")]

    if not target_files:
        print("対象ファイルなし。処理を終了します。")
        return

    print(f"📄 対象ファイル数: {len(target_files)}")

    # ── 既存Entity一覧を事前取得（語彙統一のため） ──
    existing_entities = get_existing_entities()
    print(f"📚 既存 Entity 数: {len(existing_entities)}")
    if existing_entities:
        print(f"  📋 一覧（先頭10件）: {existing_entities[:10]}")

    error_files = []  # エンティティ抽出エラーが発生したファイル

    for file_path in target_files:
        print(f"\n── 処理開始: {file_path} ──")

        if not os.path.exists(file_path):
            print(f"  ⚠️ ファイルが存在しません: {file_path}")
            continue

        # ① パース
        doc = parse_md(file_path)
        print(f"  📝 タイトル: {doc['title']}")
        print(f"  👤 Author:  {doc['author']}")
        print(f"  📅 Date:    {doc['date']}")
        print(f"  ❓ Why:     {doc['why'][:50]}..." if len(doc['why']) > 50 else f"  ❓ Why:     {doc['why']}")
        print(f"  🔧 How:     {doc['how'][:50]}..." if len(doc['how']) > 50 else f"  🔧 How:     {doc['how']}")
        print(f"  📊 Result:  {doc['result'][:50]}..." if len(doc['result']) > 50 else f"  📊 Result:  {doc['result']}")
        print(f"  🛠️ Tools:   {doc['tools']}")

        # ② エンティティ抽出（既存Entityリストを渡して語彙統一）
        try:
            entities = extract_entities(doc, existing_entities)
        except Exception:
            entities = []
            error_files.append(file_path)
        print(f"  🔍 抽出エンティティ: {entities}")

        # ③ Neo4j登録（エンティティ空でもKnowledge/Author/Toolは登録する）
        register_to_neo4j(doc, entities)
        print(f"  ✅ 登録完了: {file_path}")

    # ── エラーサマリ ──
    if error_files:
        print(f"\n::error::エンティティ抽出に失敗したファイル: {error_files}")
        print("⚠️ Knowledge/Author/Toolノードは登録済みですが、Entityノード・RELATED_TOエッジが未生成です。")
        print("   再実行でEntityを補完できます。")
        sys.exit(1)

    print("\n🎉 全ファイルの処理が完了しました")


if __name__ == "__main__":
    try:
        main()
    finally:
        driver.close()
