"""
check_knowledge.py
ナレッジ風化チェック: 週次スケジュールで実行し、
古いナレッジの移動・Slack通知・Neo4j削除を行う。

Phase 1: knowledge/ → check/ (KNOWLEDGE_CHECK_DAYS 日超過)
Phase 2: check/ 移動分の Slack 再確認依頼通知
Phase 3: check/ → invalid/ (KNOWLEDGE_INVALID_DAYS 日超過)
Phase 4: invalid 後処理
  - invalid/list/ の前回リスト削除
  - 最新週の invalid リスト生成
  - Neo4j の該当 Knowledge ノード削除
"""

import os
import re
import json
import glob
import shutil
from datetime import datetime, timezone, timedelta
from urllib.request import Request, urlopen
from urllib.error import URLError

# Neo4j ドライバはオプション（インストールされていれば使う）
try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


# ── 定数 ──────────────────────────────────────────────────────────
JST = timezone(timedelta(hours=9))


# ── 設定ロード ────────────────────────────────────────────────────
def load_config() -> dict:
    """環境変数から設定を読み込む"""
    return {
        "check_days": int(os.environ.get("KNOWLEDGE_CHECK_DAYS", "30")),
        "invalid_days": int(os.environ.get("KNOWLEDGE_INVALID_DAYS", "90")),
        "webhook_url": os.environ.get("SLACK_WEBHOOK_URL", ""),
        "author_slack_map": _parse_author_map(
            os.environ.get("AUTHOR_SLACK_MAP", "{}")
        ),
        "neo4j_uri": os.environ.get("NEO4J_URI", ""),
        "neo4j_username": os.environ.get("NEO4J_USERNAME", ""),
        "neo4j_password": os.environ.get("NEO4J_PASSWORD", ""),
    }


def _parse_author_map(raw: str) -> dict:
    """AUTHOR_SLACK_MAP の JSON をパースする"""
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        print("⚠️ AUTHOR_SLACK_MAP の JSON パースに失敗しました。メンションなしで通知します。")
        return {}


# ── frontmatter パーサ ────────────────────────────────────────────
def parse_frontmatter(filepath: str) -> tuple:
    """
    frontmatter から date, author, title を抽出する。
    戻り値: (date_str, author, title)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None, None, None

    fm = match.group(1)
    date_m = re.search(r'^date:\s*["\']?(\d{4}-\d{2}-\d{2})["\']?', fm, re.MULTILINE)
    author_m = re.search(r'^author:\s*["\']?(.*?)["\']?\s*$', fm, re.MULTILINE)
    title_m = re.search(r'^title:\s*["\']?(.*?)["\']?\s*$', fm, re.MULTILINE)

    date_val = date_m.group(1) if date_m else None
    author_val = author_m.group(1).strip() if author_m else None
    title_val = title_m.group(1).strip() if title_m else None

    return date_val, author_val, title_val


# ── ナレッジ ID 生成 ──────────────────────────────────────────────
def make_knowledge_id(original_filepath: str) -> str:
    """
    register_graph.py と同じ ID 生成ロジック。
    knowledge/ パスベースで ID を生成する。
    例: knowledge/2026-04-01-010457-AI任せ開発の実践.md
      → knowledge__2026-04-01-010457-AI任せ開発の実践
    """
    # ファイル名部分を取得
    filename = os.path.basename(original_filepath)
    # 元の knowledge/ パスを復元
    knowledge_path = f"knowledge/{filename}"
    return knowledge_path.replace("/", "__").replace(".md", "")


# ── Slack 通知 ────────────────────────────────────────────────────
def send_slack_notification(
    config: dict,
    author: str,
    title: str,
    filename: str,
    action_type: str,
) -> None:
    """Slack に通知を送信する"""
    webhook_url = config["webhook_url"]
    if not webhook_url:
        print(f"⚠️ SLACK_WEBHOOK_URL 未設定。通知スキップ: {filename}")
        return

    author_slack_map = config["author_slack_map"]
    slack_id = author_slack_map.get(author, "") if author else ""

    if slack_id:
        mention = f"<@{slack_id}>"
    elif author:
        mention = f"@{author}"
    else:
        mention = "（投稿者不明）"

    check_days = config["check_days"]
    invalid_days = config["invalid_days"]

    if action_type == "check":
        emoji = "🔔"
        message = (
            f"{emoji} {mention} さんのナレッジ「{title or filename}」が "
            f"{check_days} 日以上経過したため、確認対象に移動しました。\n"
            f"内容を確認し、まだ有効であれば `knowledge/` フォルダに戻してください。"
        )
    else:
        emoji = "⚠️"
        message = (
            f"{emoji} {mention} さんのナレッジ「{title or filename}」が "
            f"無効化されました（{invalid_days} 日以上経過）。\n"
            f"メンテナンスが必要な場合は内容を更新して `knowledge/` に戻してください。"
        )

    payload = json.dumps({"text": message}, ensure_ascii=False).encode("utf-8")
    req = Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
    try:
        urlopen(req)
        print(f"✅ Slack 通知送信: {filename}")
    except URLError as e:
        print(f"❌ Slack 通知失敗: {filename} - {e}")


# ── Neo4j 削除 ────────────────────────────────────────────────────
def delete_from_neo4j(config: dict, knowledge_id: str, title: str) -> None:
    """
    Neo4j から Knowledge ノードと関連エッジを削除する。
    DETACH DELETE により全リレーションシップ
    (WRITTEN_BY, USES_TOOL, CONTAINS_ENTITY, RELATED_TO) も削除される。
    Entity / Tool / Author ノード自体は残す。
    """
    if not HAS_NEO4J:
        print(f"⚠️ neo4j パッケージ未インストール。Neo4j 削除スキップ: {title}")
        return

    uri = config["neo4j_uri"]
    username = config["neo4j_username"]
    password = config["neo4j_password"]

    if not uri or not username or not password:
        print(f"⚠️ Neo4j 接続情報が未設定。削除スキップ: {title}")
        return

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            # 削除前に存在確認
            result = session.run(
                "MATCH (k:Knowledge {id: $id}) RETURN k.title AS title",
                id=knowledge_id,
            )
            record = result.single()

            if record:
                # Knowledge ノードと全エッジを削除
                session.run(
                    "MATCH (k:Knowledge {id: $id}) DETACH DELETE k",
                    id=knowledge_id,
                )
                print(f"🗑️ Neo4j 削除完了: {title} (id={knowledge_id})")
            else:
                print(f"ℹ️ Neo4j にレコードなし: {title} (id={knowledge_id})")
    except Exception as e:
        print(f"❌ Neo4j 削除失敗: {title} - {e}")
    finally:
        if driver:
            driver.close()


# ── Phase 1: knowledge/ 走査 ─────────────────────────────────────
def phase1_process_knowledge(config: dict, today) -> tuple[list, list]:
    """
    knowledge/ 配下の .md を走査する。
    - age >= KNOWLEDGE_INVALID_DAYS: invalid/ へ移動
    - age >= KNOWLEDGE_CHECK_DAYS:   check/ へ移動
    戻り値: (moved_to_check, moved_to_invalid)
      各リスト要素: (filename, author, title, date_str, age_days)
    """
    check_days = config["check_days"]
    invalid_days = config["invalid_days"]
    os.makedirs("check", exist_ok=True)
    os.makedirs("invalid", exist_ok=True)
    
    moved_to_check = []
    moved_to_invalid = []

    for filepath in sorted(glob.glob("knowledge/*.md")):
        date_str, author, title = parse_frontmatter(filepath)
        if not date_str:
            print(f"⏭️ date なし、スキップ: {filepath}")
            continue
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            print(f"⏭️ date パース失敗、スキップ: {filepath}")
            continue

        age_days = (today - file_date).days
        filename = os.path.basename(filepath)

        if age_days >= invalid_days:
            dest = os.path.join("invalid", filename)
            shutil.move(filepath, dest)
            moved_to_invalid.append((filename, author, title, date_str, age_days))
            print(f"🗑️ invalid/ へ直接移動 ({age_days}日経過): {filename}")
        elif age_days >= check_days:
            dest = os.path.join("check", filename)
            shutil.move(filepath, dest)
            moved_to_check.append((filename, author, title, date_str, age_days))
            print(f"📦 check/ へ移動 ({age_days}日経過): {filename}")

    return moved_to_check, moved_to_invalid


# ── Phase 2: Slack 再確認依頼通知 ─────────────────────────────────
def phase2_notify_check(config: dict, moved_to_check: list) -> None:
    """check/ に移動されたナレッジについて Slack 通知を送る"""
    for filename, author, title, date_str, age_days in moved_to_check:
        send_slack_notification(config, author, title, filename, "check")


# ── Phase 3: check/ → invalid/ ───────────────────────────────────
def phase3_move_to_invalid(config: dict, today) -> list:
    """
    check/ 配下の .md を走査し、
    KNOWLEDGE_INVALID_DAYS 日以上経過したものを invalid/ に移動する。
    戻り値: [(filename, author, title, date_str, age_days), ...]
    """
    invalid_days = config["invalid_days"]
    os.makedirs("invalid", exist_ok=True)
    moved = []

    for filepath in sorted(glob.glob("check/*.md")):
        date_str, author, title = parse_frontmatter(filepath)
        if not date_str:
            continue
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        age_days = (today - file_date).days
        if age_days >= invalid_days:
            filename = os.path.basename(filepath)
            dest = os.path.join("invalid", filename)
            shutil.move(filepath, dest)
            moved.append((filename, author, title, date_str, age_days))
            print(f"🗑️ invalid/ へ移動 ({age_days}日経過): {filename}")
            # 通知はここで行う
            send_slack_notification(config, author, title, filename, "invalid")

    return moved


# ── Phase 4: invalid 後処理 ───────────────────────────────────────
def phase4_post_invalid(config: dict, moved_to_invalid: list, today) -> None:
    """
    invalid 後処理:
    1. invalid/list/ の前回リストを削除
    2. 最新週の invalid リストを生成 (invalid/配下の全ファイル対象)
    3. Neo4j から該当 Knowledge ノードを削除 (今回移動分のみ)
    """
    os.makedirs("invalid/list", exist_ok=True)

    # ── 4-1: 前回リスト削除（.gitkeep は保持） ──
    for old_list in glob.glob("invalid/list/*.md"):
        os.remove(old_list)
        print(f"🧹 前回リスト削除: {old_list}")

    # ── 4-2: 最新リスト生成 (全 invalid ファイルを走査) ──
    invalid_files = sorted(glob.glob("invalid/*.md"))
    if invalid_files:
        list_date = today.strftime("%Y-%m-%d")
        list_filename = f"invalid/list/{list_date}-invalid-list.md"

        lines = [
            f"# 無効化ナレッジ一覧 ({list_date})\n",
            "| ファイル名 | 投稿者 | 投稿日 | 経過日数 |",
            "|---|---|---|---|",
        ]
        
        for filepath in invalid_files:
            filename = os.path.basename(filepath)
            date_str, author, title = parse_frontmatter(filepath)
            
            age_days = "?"
            if date_str:
                try:
                    file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    age_days = str((today - file_date).days)
                except ValueError:
                    pass
            
            display = title or filename
            lines.append(
                f"| {display} | {author or '不明'} | {date_str or '不明'} | {age_days}日 |"
            )

        lines.append(f"\n合計: {len(invalid_files)} 件\n")

        with open(list_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"📝 無効化リスト生成: {list_filename} ({len(invalid_files)}件)")

    # ── 4-3: Neo4j 削除 (今回移動分のみ) ──
    for filename, author, title, date_str, age_days in moved_to_invalid:
        knowledge_id = make_knowledge_id(filename)
        delete_from_neo4j(config, knowledge_id, title or filename)


# ── メイン ────────────────────────────────────────────────────────
def main():
    config = load_config()
    today = datetime.now(JST).date()

    print(f"📅 実行日: {today}")
    print(f"⚙️ CHECK_DAYS={config['check_days']}, INVALID_DAYS={config['invalid_days']}")
    print(f"🔗 Neo4j: {'接続あり' if config['neo4j_uri'] else '未設定'}")
    print()

    # Phase A: check/ -> invalid/ (先に行うことで、この回で check に入ったものは除外される)
    print("=" * 50)
    print("Phase A: check/ -> invalid/")
    print("=" * 50)
    moved_from_check_to_invalid = phase3_move_to_invalid(config, today)

    # Phase B: knowledge/ -> check/ OR invalid/
    print()
    print("=" * 50)
    print("Phase B: knowledge/ -> check/ or invalid/")
    print("=" * 50)
    moved_to_check, moved_to_invalid_direct = phase1_process_knowledge(config, today)

    # 通知: check 移動分
    phase2_notify_check(config, moved_to_check)
    
    # 通知: knowledge -> invalid 直接移動分
    for filename, author, title, date_str, age_days in moved_to_invalid_direct:
        send_slack_notification(config, author, title, filename, "invalid")

    # Phase C: invalid 後処理
    print()
    print("=" * 50)
    print("Phase C: invalid 後処理 (Neo4j削除等)")
    print("=" * 50)
    all_moved_to_invalid = moved_from_check_to_invalid + moved_to_invalid_direct
    phase4_post_invalid(config, all_moved_to_invalid, today)

    # Summary
    print()
    total_changes = len(moved_to_check) + len(all_moved_to_invalid)
    if total_changes == 0:
        print("✅ 移動対象のナレッジはありませんでした。")
    else:
        print(
            f"📊 処理結果: check移動={len(moved_to_check)}件, "
            f"invalid移動={len(all_moved_to_invalid)}件"
        )


if __name__ == "__main__":
    main()
