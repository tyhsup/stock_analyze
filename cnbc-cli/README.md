# Cnbc Cli CLI

CNBC 全球財經新聞與即時行情 API（涵蓋 World News、國際市場、經濟、科技、個股報價與即時搜尋）。

Printed by [@ting-yu-hsu](https://github.com/ting-yu-hsu) (Ting-Yu Hsu).

## Install

The recommended path installs both the `cnbc-cli-pp-cli` binary and the `pp-cnbc-cli` agent skill in one shot:

```bash
npx -y @mvanhorn/printing-press install cnbc-cli
```

For CLI only (no skill):

```bash
npx -y @mvanhorn/printing-press install cnbc-cli --cli-only
```


### Without Node

The generated install path is category-agnostic until this CLI is published. If `npx` is not available before publish, install Node or use the category-specific Go fallback from the public-library entry after publish.

### Pre-built binary

Download a pre-built binary for your platform from the [latest release](https://github.com/mvanhorn/printing-press-library/releases/tag/cnbc-cli-current). On macOS, clear the Gatekeeper quarantine: `xattr -d com.apple.quarantine <binary>`. On Unix, mark it executable: `chmod +x <binary>`.

<!-- pp-hermes-install-anchor -->
## Install for Hermes

From the Hermes CLI:

```bash
hermes skills install mvanhorn/printing-press-library/cli-skills/pp-cnbc-cli --force
```

Inside a Hermes chat session:

```bash
/skills install mvanhorn/printing-press-library/cli-skills/pp-cnbc-cli --force
```

## Install for OpenClaw

Tell your OpenClaw agent (copy this):

```
Install the pp-cnbc-cli skill from https://github.com/mvanhorn/printing-press-library/tree/main/cli-skills/pp-cnbc-cli. The skill defines how its required CLI can be installed.
```

## Quick Start

### 1. Install

See [Install](#install) above.

### 2. Verify Setup

```bash
cnbc-cli-pp-cli doctor
```

This checks your configuration.

### 3. Try Your First Command

```bash
cnbc-cli-pp-cli quote-html-webservice --symbols example-value
```

## Usage

Run `cnbc-cli-pp-cli --help` for the full command reference and flag list.

## Commands

### quote-html-webservice

Manage quote html webservice

- **`cnbc-cli-pp-cli quote-html-webservice get-quotes`** - 取得 Dow Jones (.DJI)、S&P 500 (.SPX)、Nasdaq (.IXIC) 等主要指數或個股 (AAPL, NVDA, TSLA) 之即時行情。

### rs

Manage rs

- **`cnbc-cli-pp-cli rs get-news-feed`** - 取得 CNBC 的新聞 RSS 摘要列表，支援多種新聞分類：
- `100727362`: 國際全球新聞與深度分析 (World / International News & Analysis)
- `10001147`: 商業新聞 (Business News)
- `100003114`: 美國頭條新聞 (US Top News and Analysis)
- `15837362`: 美國新聞 (U.S. News)
- `20910258`: 總體經濟 (Economy)
- `10000664`: 金融市場 (Finance)
- `19854910`: 科技產業 (Tech)
- `19832390`: 亞洲新聞 (Asia News)
- `15839069`: 投資理財 (Investing)
- **`cnbc-cli-pp-cli rs search-news`** - 依關鍵字搜尋 CNBC 新聞文章與報導。


## Output Formats

```bash
# Human-readable table (default in terminal, JSON when piped)
cnbc-cli-pp-cli quote-html-webservice --symbols example-value

# JSON for scripting and agents
cnbc-cli-pp-cli quote-html-webservice --symbols example-value --json

# Filter to specific fields
cnbc-cli-pp-cli quote-html-webservice --symbols example-value --json --select id,name,status

# Dry run — show the request without sending
cnbc-cli-pp-cli quote-html-webservice --symbols example-value --dry-run

# Agent mode — JSON + compact + no prompts in one flag
cnbc-cli-pp-cli quote-html-webservice --symbols example-value --agent
```

## Agent Usage

This CLI is designed for AI agent consumption:

- **Non-interactive** - never prompts, every input is a flag
- **Pipeable** - `--json` output to stdout, errors to stderr
- **Filterable** - `--select id,name` returns only fields you need
- **Previewable** - `--dry-run` shows the request without sending
- **Read-only by default** - this CLI does not create, update, delete, publish, send, or mutate remote resources
- **Offline-friendly** - sync/search commands can use the local SQLite store when available
- **Agent-safe by default** - no colors or formatting unless `--human-friendly` is set

Exit codes: `0` success, `2` usage error, `3` not found, `5` API error, `7` rate limited, `10` config error.

## Use with Claude Code

Install the focused skill — it auto-installs the CLI on first invocation:

```bash
npx skills add mvanhorn/printing-press-library/cli-skills/pp-cnbc-cli -g
```

Then invoke `/pp-cnbc-cli <query>` in Claude Code. The skill is the most efficient path — Claude Code drives the CLI directly without an MCP server in the middle.

<details>
<summary>Use as an MCP server in Claude Code (advanced)</summary>

If you'd rather register this CLI as an MCP server in Claude Code, install the MCP binary first:


Install the MCP binary from this CLI's published public-library entry or pre-built release.

Then register it:

```bash
claude mcp add cnbc-cli cnbc-cli-pp-mcp
```

</details>

## Use with Claude Desktop

This CLI ships an [MCPB](https://github.com/modelcontextprotocol/mcpb) bundle — Claude Desktop's standard format for one-click MCP extension installs (no JSON config required).

To install:

1. Download the `.mcpb` for your platform from the [latest release](https://github.com/mvanhorn/printing-press-library/releases/tag/cnbc-cli-current).
2. Double-click the `.mcpb` file. Claude Desktop opens and walks you through the install.

Requires Claude Desktop 1.0.0 or later. Pre-built bundles ship for macOS Apple Silicon (`darwin-arm64`) and Windows (`amd64`, `arm64`); for other platforms, use the manual config below.

<details>
<summary>Manual JSON config (advanced)</summary>

If you can't use the MCPB bundle (older Claude Desktop, unsupported platform), install the MCP binary and configure it manually.


Install the MCP binary from this CLI's published public-library entry or pre-built release.

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "cnbc-cli": {
      "command": "cnbc-cli-pp-mcp"
    }
  }
}
```

</details>

## Health Check

```bash
cnbc-cli-pp-cli doctor
```

Verifies configuration and connectivity to the API.

## Configuration

Config file: `~/.config/cnbc-news-market-pp-cli/config.toml`

Static request headers can be configured under `headers`; per-command header overrides take precedence.

## Troubleshooting
**Not found errors (exit code 3)**
- Check the resource ID is correct
- Run the `list` command to see available items

---

Generated by [CLI Printing Press](https://github.com/mvanhorn/cli-printing-press)
