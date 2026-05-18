# Cnyes Cli CLI

非官方的鉅亨網新聞列表 API。

Printed by [@ting-yu-hsu](https://github.com/ting-yu-hsu) (Ting-Yu Hsu).

## Install

The recommended path installs both the `cnyes-cli-pp-cli` binary and the `pp-cnyes-cli` agent skill in one shot:

```bash
npx -y @mvanhorn/printing-press install cnyes-cli
```

For CLI only (no skill):

```bash
npx -y @mvanhorn/printing-press install cnyes-cli --cli-only
```


### Without Node

The generated install path is category-agnostic until this CLI is published. If `npx` is not available before publish, install Node or use the category-specific Go fallback from the public-library entry after publish.

### Pre-built binary

Download a pre-built binary for your platform from the [latest release](https://github.com/mvanhorn/printing-press-library/releases/tag/cnyes-cli-current). On macOS, clear the Gatekeeper quarantine: `xattr -d com.apple.quarantine <binary>`. On Unix, mark it executable: `chmod +x <binary>`.

<!-- pp-hermes-install-anchor -->
## Install for Hermes

From the Hermes CLI:

```bash
hermes skills install mvanhorn/printing-press-library/cli-skills/pp-cnyes-cli --force
```

Inside a Hermes chat session:

```bash
/skills install mvanhorn/printing-press-library/cli-skills/pp-cnyes-cli --force
```

## Install for OpenClaw

Tell your OpenClaw agent (copy this):

```
Install the pp-cnyes-cli skill from https://github.com/mvanhorn/printing-press-library/tree/main/cli-skills/pp-cnyes-cli. The skill defines how its required CLI can be installed.
```

## Quick Start

### 1. Install

See [Install](#install) above.

### 2. Verify Setup

```bash
cnyes-cli-pp-cli doctor
```

This checks your configuration.

### 3. Try Your First Command

```bash
cnyes-cli-pp-cli media get-news-list-by-category mock-value
```

## Usage

Run `cnyes-cli-pp-cli --help` for the full command reference and flag list.

## Commands

### media

Manage media

- **`cnyes-cli-pp-cli media get-news-list-by-category`** - 取得鉅亨網的新聞列表，支援多種分類：
- `headline` (頭條新聞)
- `tw_stock` (台股新聞)
- `us_stock` (美股新聞)
- `cn_stock` (陸港股新聞)
- `forex` (外匯新聞)
- `wd_macro` (全球/總經新聞)
- **`cnyes-cli-pp-cli media get-news-list-by-symbol`** - 取得鉅亨網特定個股的新聞列表，支援台股/美股等股票。
例如台積電為 `TWS:2330:STOCK`。


## Output Formats

```bash
# Human-readable table (default in terminal, JSON when piped)
cnyes-cli-pp-cli media get-news-list-by-category mock-value

# JSON for scripting and agents
cnyes-cli-pp-cli media get-news-list-by-category mock-value --json

# Filter to specific fields
cnyes-cli-pp-cli media get-news-list-by-category mock-value --json --select id,name,status

# Dry run — show the request without sending
cnyes-cli-pp-cli media get-news-list-by-category mock-value --dry-run

# Agent mode — JSON + compact + no prompts in one flag
cnyes-cli-pp-cli media get-news-list-by-category mock-value --agent
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
npx skills add mvanhorn/printing-press-library/cli-skills/pp-cnyes-cli -g
```

Then invoke `/pp-cnyes-cli <query>` in Claude Code. The skill is the most efficient path — Claude Code drives the CLI directly without an MCP server in the middle.

<details>
<summary>Use as an MCP server in Claude Code (advanced)</summary>

If you'd rather register this CLI as an MCP server in Claude Code, install the MCP binary first:


Install the MCP binary from this CLI's published public-library entry or pre-built release.

Then register it:

```bash
claude mcp add cnyes-cli cnyes-cli-pp-mcp
```

</details>

## Use with Claude Desktop

This CLI ships an [MCPB](https://github.com/modelcontextprotocol/mcpb) bundle — Claude Desktop's standard format for one-click MCP extension installs (no JSON config required).

To install:

1. Download the `.mcpb` for your platform from the [latest release](https://github.com/mvanhorn/printing-press-library/releases/tag/cnyes-cli-current).
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
    "cnyes-cli": {
      "command": "cnyes-cli-pp-mcp"
    }
  }
}
```

</details>

## Health Check

```bash
cnyes-cli-pp-cli doctor
```

Verifies configuration and connectivity to the API.

## Configuration

Config file: `~/.config/cnyes-news-pp-cli/config.toml`

Static request headers can be configured under `headers`; per-command header overrides take precedence.

## Troubleshooting
**Not found errors (exit code 3)**
- Check the resource ID is correct
- Run the `list` command to see available items

---

Generated by [CLI Printing Press](https://github.com/mvanhorn/cli-printing-press)
