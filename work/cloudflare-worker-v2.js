/**
 * Cloudflare Worker - Slack ナレッジBot（投稿 + 検索）
 *
 * 環境変数（Cloudflare Dashboard > Workers > Settings > Variables）:
 *   SLACK_BOT_TOKEN      : xoxb-xxxx
 *   SLACK_SIGNING_SECRET : Slack App の Signing Secret
 *   GITHUB_TOKEN         : GitHub Personal Access Token
 *   GITHUB_OWNER         : GitHubユーザー名 or Org名
 *   GITHUB_REPO          : リポジトリ名
 *
 * Slackコマンド:
 *   /knowledge          → ナレッジ投稿 Modal を開く
 *   /knowledge-search   → 検索 Modal を開く
 */

// ─────────────────────────────────────────────────────────────────
// Slack Rich Text 解析 (AST -> Markdown)
// ─────────────────────────────────────────────────────────────────
function parseRichText(richTextObj) {
  if (!richTextObj || !richTextObj.elements) return "";
  let md = "";
  for (const block of richTextObj.elements) {
    if (block.type === "rich_text_section") {
      md += parseRichElements(block.elements) + "\n";
    } else if (block.type === "rich_text_list") {
      const isOrdered = block.style === "ordered";
      block.elements.forEach((item, idx) => {
        const prefix = isOrdered ? `${idx + 1}. ` : "- ";
        const indent = "  ".repeat(block.indent || 0);
        md += indent + prefix + parseRichElements(item.elements) + "\n";
      });
    } else if (block.type === "rich_text_quote") {
      md += "> " + parseRichElements(block.elements).replace(/\n/g, "\n> ") + "\n";
    } else if (block.type === "rich_text_preformatted") {
      md += "```\n" + parseRichElements(block.elements) + "\n```\n";
    }
  }
  return md.trim();
}

function parseRichElements(elements) {
  if (!elements) return "";
  let text = "";
  for (const el of elements) {
    if (el.type === "text") {
      let content = el.text;
      if (el.style?.bold) content = `**${content}**`;
      if (el.style?.italic) content = `*${content}*`;
      if (el.style?.strike) content = `~~${content}~~`;
      if (el.style?.code) content = `\`${content}\``;
      text += content;
    } else if (el.type === "link") {
      text += el.text ? `[${el.text}](${el.url})` : el.url;
    } else if (el.type === "emoji") {
      text += `:${el.name}:`;
    } else if (el.type === "user") {
      text += `<@${el.user_id}>`;
    } else if (el.type === "channel") {
      text += `<#${el.channel_id}>`;
    }
  }
  return text;
}

// ─────────────────────────────────────────────────────────────────
// Slack 署名検証
// ─────────────────────────────────────────────────────────────────
async function verifySlackSignature(request, body, signingSecret) {
  const timestamp = request.headers.get("x-slack-request-timestamp");
  const slackSig = request.headers.get("x-slack-signature");

  if (Math.abs(Date.now() / 1000 - Number(timestamp)) > 300) return false;

  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(signingSecret),
    { name: "HMAC", hash: "SHA-256" },
    false, ["sign"]
  );
  const sig = await crypto.subtle.sign(
    "HMAC", key,
    new TextEncoder().encode(`v0:${timestamp}:${body}`)
  );
  const hex = "v0=" + Array.from(new Uint8Array(sig))
    .map(b => b.toString(16).padStart(2, "0")).join("");

  return hex === slackSig;
}

// ─────────────────────────────────────────────────────────────────
// ナレッジ投稿 Modal
// ─────────────────────────────────────────────────────────────────
async function openPostModal(triggerId, botToken) {
  await fetch("https://slack.com/api/views.open", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${botToken}` },
    body: JSON.stringify({
      trigger_id: triggerId,
      view: {
        type: "modal",
        callback_id: "knowledge_submit",
        title: { type: "plain_text", text: "ナレッジ投稿" },
        submit: { type: "plain_text", text: "投稿する" },
        close: { type: "plain_text", text: "キャンセル" },
        blocks: [
          {
            type: "input", block_id: "title_block",
            label: { type: "plain_text", text: "タイトル" },
            element: {
              type: "rich_text_input", action_id: "title",
              placeholder: { type: "plain_text", text: "ナレッジを要約すると？" }
            }
          },
          {
            type: "input", block_id: "why_block",
            label: { type: "plain_text", text: "背景、課題 (why)" },
            element: {
              type: "rich_text_input", action_id: "why",
              placeholder: { type: "plain_text", text: "困りごとや気づきなど、 ナレッジを残す経緯を端的に" }
            }
          },
          {
            type: "input", block_id: "how_block",
            label: { type: "plain_text", text: "手段 (how)" },
            element: {
              type: "rich_text_input", action_id: "how",
              placeholder: { type: "plain_text", text: "どのようにしてAIを使用したか" }
            }
          },
          {
            type: "input", block_id: "result_block",
            label: { type: "plain_text", text: "結果 (result)" },
            element: {
              type: "rich_text_input", action_id: "result",
              placeholder: { type: "plain_text", text: "良かったこと、できなかったことなど" }
            }
          },
          {
            type: "input", block_id: "tools_block",
            label: { type: "plain_text", text: "ツールやAIのモデル (tools)" },
            hint: { type: "plain_text", text: "箇条書きで縦に並べて記載すると見やすくなります。（複数行入力可能）" },
            element: {
              type: "rich_text_input", action_id: "tools",
              placeholder: { type: "plain_text", text: "利用したツール(Cursorなど)やAIモデル(Claude Opus 4.6など)" }
            }
          }
        ]
      }
    })
  });
}

// ─────────────────────────────────────────────────────────────────
// 検索 Modal
// ─────────────────────────────────────────────────────────────────
async function openSearchModal(triggerId, botToken, privateMetadata = "") {
  await fetch("https://slack.com/api/views.open", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${botToken}` },
    body: JSON.stringify({
      trigger_id: triggerId,
      view: {
        type: "modal",
        callback_id: "knowledge_search",
        private_metadata: privateMetadata,
        title: { type: "plain_text", text: "ナレッジ検索" },
        submit: { type: "plain_text", text: "検索する" },
        close: { type: "plain_text", text: "キャンセル" },
        blocks: [
          // 検索ワード
          {
            type: "input", block_id: "query_block",
            label: { type: "plain_text", text: "検索キーワード" },
            element: {
              type: "plain_text_input", action_id: "query",
              placeholder: { type: "plain_text", text: "例: プロンプト設計のコツ" }
            }
          }
        ]
      }
    })
  });
}


// ─────────────────────────────────────────────────────────────────
// GitHub Actions キック（投稿用）
// ─────────────────────────────────────────────────────────────────
async function triggerGitHubActions(eventType, payload, env) {
  const res = await fetch(
    `https://api.github.com/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/dispatches`,
    {
      method: "POST",
      headers: {
        "Accept": "application/vnd.github+json",
        "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "CloudflareWorker-SlackBot"
      },
      body: JSON.stringify({ event_type: eventType, client_payload: payload })
    }
  );
  return res.status; // 204 = 成功
}

// ─────────────────────────────────────────────────────────────────
// メインハンドラー
// ─────────────────────────────────────────────────────────────────
export default {
  async fetch(request, env) {
    if (request.method !== "POST") return new Response("Method Not Allowed", { status: 405 });

    const rawBody = await request.text();
    const isValid = await verifySlackSignature(request, rawBody, env.SLACK_SIGNING_SECRET);
    if (!isValid) return new Response("Unauthorized", { status: 401 });

    const contentType = request.headers.get("content-type") ?? "";
    if (!contentType.includes("application/x-www-form-urlencoded")) {
      return new Response("OK", { status: 200 });
    }

    const params = new URLSearchParams(rawBody);
    const payload = params.get("payload");

    // ── スラッシュコマンド ────────────────────────────────────────
    if (!payload) {
      const command = params.get("command");
      const triggerId = params.get("trigger_id");

      if (command === "/knowledge") {
        // 投稿 Modal
        await openPostModal(triggerId, env.SLACK_BOT_TOKEN);
        return new Response("", { status: 200 });
      }

      if (command === "/knowledge-search") {
        // 検索 Modal
        const channelId = params.get("channel_id") ?? "";
        await openSearchModal(triggerId, env.SLACK_BOT_TOKEN, channelId);
        return new Response("", { status: 200 });
      }

      if (command === "/search") {
        // 検索 Modal（エイリアス）
        const channelId = params.get("channel_id") ?? "";
        await openSearchModal(triggerId, env.SLACK_BOT_TOKEN, channelId);
        return new Response("", { status: 200 });
      }

      return new Response("", { status: 200 });
    }

    const data = JSON.parse(payload);

    // ── ショートカット ────────────────────────────────────────────
    if (data.type === "shortcut" || data.type === "message_action") {
      if (data.callback_id === "knowledge_search_shortcut") {
        const channelId = data.channel?.id ?? "";
        await openSearchModal(data.trigger_id, env.SLACK_BOT_TOKEN, channelId);
      } else {
        await openPostModal(data.trigger_id, env.SLACK_BOT_TOKEN);
      }
      return new Response("", { status: 200 });
    }

    // ── Modal 送信 ────────────────────────────────────────────────
    if (data.type === "view_submission") {

      // ① 投稿 Modal の送信
      if (data.view?.callback_id === "knowledge_submit") {
        const values = data.view.state.values;
        const title = parseRichText(values.title_block?.title?.rich_text_value);
        const why = parseRichText(values.why_block?.why?.rich_text_value);
        const how = parseRichText(values.how_block?.how?.rich_text_value);
        const result = parseRichText(values.result_block?.result?.rich_text_value);
        const tools = parseRichText(values.tools_block?.tools?.rich_text_value);
        const author = data.user?.name ?? "";
        const userId = data.user?.id ?? "";

        const status = await triggerGitHubActions(
          "slack-knowledge-submitted",
          { title, why, how, result, tools, author },
          env
        );

        await fetch("https://slack.com/api/chat.postMessage", {
          method: "POST",
          headers: { "Content-Type": "application/json", "Authorization": `Bearer ${env.SLACK_BOT_TOKEN}` },
          body: JSON.stringify({
            channel: userId,
            text: status === 204
              ? `✅ ナレッジ「${title}」を登録しました！GitHubに保存されます。`
              : `❌ ナレッジの登録に失敗しました（GitHub連携エラー）。管理者に連絡してください。`
          })
        });

        return new Response("", { status: 200 });
      }

      // ② 検索 Modal の送信
      if (data.view?.callback_id === "knowledge_search") {
        const values = data.view.state.values;
        const query = values.query_block?.query?.value ?? "";
        const userId = data.user?.id ?? "";
        const channelId = data.view?.private_metadata || "";
        const status = await triggerGitHubActions(
          "search-knowledge-slack",
          {
            query,
            channel_id: channelId,
            user_id: userId,
            result_target: "channel"
          },
          env
        );

        const notifyText = status === 204
          ? "🔎 検索を受け付けました。結果を取得できしだい投稿します。"
          : "❌ 検索リクエストの連携に失敗しました。管理者に連絡してください。";
        await fetch("https://slack.com/api/chat.postMessage", {
          method: "POST",
          headers: { "Content-Type": "application/json", "Authorization": `Bearer ${env.SLACK_BOT_TOKEN}` },
          body: JSON.stringify({ channel: userId, text: notifyText })
        });

        return new Response("", { status: 200 });
      }
    }

    return new Response("OK", { status: 200 });
  }
};
