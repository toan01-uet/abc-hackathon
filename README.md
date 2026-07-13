# abc-hackathon-123

# MeetingTasksAgent

## Overview
The Meeting Tasks Agent extracts action items from meeting transcripts and helps users convert them into structured tasks in the appropriate project management system.

The agent bridges the gap between discussion and execution by ensuring that no action items are lost and that follow-up becomes part of the operational workflow.

> **Note:** The Problem/Solution/Key Features/Architecture/Architectural Choices/Demo/Scenario sections below describe the *original* Copilot Studio/Power Platform prototype (Planner + Azure DevOps connectors) and are kept for historical context. **The actual code in this repository is a from-scratch Python reimplementation targeting Notion instead** — see [Technologies Used](#technologies-used) onward for what's actually here and how to run it.

---

## Problem
After meetings, action items are often:
- Not clearly captured
- Manually recreated in different systems
- Spread across multiple tools

This leads to inefficiencies, inconsistencies, and missed follow-ups.

---

## Solution
The Meeting Tasks Agent:
- Extracts action items from meeting transcripts
- Structures them into clear tasks
- Identifies owners, due dates, and dependencies
- Allows the user to decide where tasks should be created
- Creates tasks directly in the selected system

---

## Key Features
- Task extraction from unstructured meeting transcripts
- Detection of owner, due date, and dependencies  
- Human-in-the-loop validation before execution  
- Support for multiple project management systems  
- Direct task creation via integrated connectors  

---

## Architecture

The solution follows a structured, multi-step workflow:

1. A meeting transcript is provided (manual input or trigger)
2. The agent extracts and structures action items
3. Tasks are presented to the user for validation
4. The user selects the target system and project/plan
5. Tasks are created via system connectors

---

## Architectural Choices

### Human-in-the-loop
The agent does not automatically create tasks. Instead:
- The user validates extracted tasks
- The user selects the target system (Planner or Azure DevOps)
- The user selects the correct project or plan

This ensures:
- Higher accuracy
- Better user trust
- Flexibility across teams and tools

---

### Instruction-based orchestration
The agent uses structured instructions rather than rigid topic flows:
- Enables flexibility for different types of meetings
- Handles unstructured transcript input effectively
- Reduces dependency on predefined conversation paths

---

### Modular design with child agents
The solution separates orchestration and execution:

- Main agent → handles task extraction and user interaction  
- Child agents → handle system-specific task creation  

This allows:
- Clear separation of concerns
- Easier maintenance
- Scalable extension (e.g. adding Jira later)

---

### Tool-based execution
The agent integrates directly with external systems using connectors:

- Microsoft Planner → task creation  
- Azure DevOps → work item creation  

This enables seamless transition from meeting output to operational tools.

---

## Demo

https://youtu.be/hPbb13zlAx4

This demo shows:
- Task extraction from a meeting transcript
- Identification of dependencies
- User selection of the target system
- Task creation in Azure DevOps

---

## Scenario

The demo is based on a recruitment scenario for a shoe store:

- A meeting transcript is analysed  
- Action items are extracted  
- Dependencies between tasks are identified  
- The user selects first Planner, and then Azure DevOps  
- Tasks are created in the **Recruitment DEMO** project and plan

---

## Technologies Used

This repository contains a from-scratch Python reimplementation of the concept above (the original Copilot Studio/Power Platform prototype description is preserved above for context):

- **Chainlit** — chat UI, with its native MCP client support for connecting to external tool servers
- **LangChain / LangGraph** — LLM wrapper, structured output, and the agentic tool-calling loop used to talk to Notion's MCP tools
- **Notion**, via Notion's official hosted MCP server (`https://mcp.notion.com/mcp`) — task/page creation, connected live through Chainlit's MCP integration (`stdio`, bridged with `mcp-remote` so the browser OAuth sign-in works even though Chainlit itself has no native OAuth support for MCP)
- **An OpenAI-compatible LLM endpoint** (FPT Cloud AI Marketplace, `DeepSeek-V4-Flash` by default) — transcript extraction and structured-output/tool-calling

---

## Setup

### Prerequisites

- Python >= 3.13 and [`uv`](https://docs.astral.sh/uv/)
- Node.js + `npx` (used to run `mcp-remote`, the bridge that gives Notion's hosted MCP server browser-based OAuth sign-in) — see [Installing Node.js/npx](#installing-nodejsnpx-for-mcp-remote) below if you don't have it yet
- An API key for an OpenAI-compatible LLM endpoint (default setup uses FPT Cloud AI Marketplace)
- A Notion account with edit access to whatever page/database you want tasks created in

#### Installing Node.js/npx (for `mcp-remote`)

Connecting to Notion (step 3 below) runs `npx -y mcp-remote https://mcp.notion.com/mcp`, which requires Node.js (npx ships with it). If `npx --version` already works in your terminal, skip this.

**Ubuntu / Debian:**

```bash
# Recommended: NodeSource repo (gets a recent LTS version; the Ubuntu repo's default nodejs is often too old)
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

node --version
npx --version
```

Alternative via [nvm](https://github.com/nvm-sh/nvm) (no `sudo` needed, easier to manage multiple versions):

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
# restart your shell, then:
nvm install --lts
```

**Windows:**

- Easiest: download and run the LTS installer from [nodejs.org](https://nodejs.org/) (includes npm/npx), then open a **new** terminal (PowerShell or Command Prompt) so `PATH` picks it up.
- Or via [winget](https://learn.microsoft.com/windows/package-manager/winget/): `winget install OpenJS.NodeJS.LTS`
- Or via [nvm-windows](https://github.com/coreybutler/nvm-windows): install it, then `nvm install lts` followed by `nvm use <version>`.

Verify in a new terminal:

```powershell
node --version
npx --version
```

If you're running this project's app itself under WSL (Windows Subsystem for Linux), follow the Ubuntu/Debian instructions inside your WSL distro instead — Node.js installed on native Windows won't be visible from within WSL.

**Notes/caveats — nothing else needs installing, but two environment quirks to know about:**
- `npx -y mcp-remote ...` downloads and runs `mcp-remote` on the fly along with its own dependencies (`express`, `open`, etc.) — you don't need to `npm install` anything yourself.
- `mcp-remote` tries to auto-open your default browser for the Notion OAuth sign-in. If you're on a headless machine (SSH session, remote server, container with no desktop/browser), that auto-open will silently fail, but the sign-in link is still printed to the terminal (`Please authorize this client by visiting: https://mcp.notion.com/authorize?...`) — copy that URL into any browser (even on a different device) to complete sign-in; the token gets cached locally afterward.
- This repo's `.chainlit/config.toml` already allowlists `npx` for stdio MCP servers (`features.mcp.stdio.allowed_executables`), so no config change is needed here — only relevant if you're wiring this up in a fresh Chainlit project.

### Install & configure

```bash
uv sync
cp .env.example .env
```

Edit `.env`:

```
FPT_API_KEY=your-api-key-here
FPT_BASE_URL=https://mkp-api.fptcloud.com/v1
LLM_MODEL=DeepSeek-V4-Flash
```

Never commit the real `.env` (it's gitignored). Optional env vars for debugging: `LOG_LEVEL=DEBUG` (verbose logging of every LLM/MCP call) and `LOG_FILE=<path>` (also write logs to a file).

### Run the app

```bash
uv run chainlit run app.py -w
```

Open the printed URL (default `http://localhost:8000`). `-w` enables auto-reload on code changes — note it only reloads `app.py` reliably; if you edit a file under `meeting_agent/`, restart the process to be safe.

### Quick extraction-only test (no Chainlit, no Notion)

Useful for iterating on the extraction prompt or verifying the LLM endpoint works at all:

```bash
uv run python scripts/smoke_extract.py samples/transcript_en.txt
uv run python scripts/smoke_extract.py samples/transcript_vi.txt
# or your own file:
uv run python scripts/smoke_extract.py path/to/your_transcript.txt
```

Prints the extracted `TaskList` as JSON. No Notion connection needed.

---

## Using the agent, step by step

### 1. Provide a meeting transcript

In the chat, either:
- **Paste** the transcript text directly, or
- **Attach** a `.txt`/`.md` file (drag-and-drop or the 📎 icon in the composer)

The agent extracts action items — title, description, owner, due date, dependencies, and an inferred progress status — and shows them as a numbered list.

### 2. Review and correct

Reply with free-text corrections as needed, e.g.:
- `"merge tasks 2 and 3"`
- `"John is the owner of task 1"`
- `"task 4 is already done"`

Each correction re-runs extraction against the current list + your feedback and re-displays it. Repeat as many times as needed.

Click **✅ Looks good, proceed** when the list is ready.

### 3. Connect Notion (can be done any time before this point too)

Click the 🔌 icon in the message composer → **Add MCP server**:

| Field | Value |
|---|---|
| Type | `stdio` |
| Command | `npx -y mcp-remote https://mcp.notion.com/mcp` |

The first time, this opens a browser window for you to sign in to Notion and grant access — no manual token/integration setup needed. Subsequent connections reuse the cached session. Once connected, Chainlit shows "Connected MCP server `notion` — N tools available."

If you proceed without connecting first, the agent will just prompt you to connect before continuing.

### 4. Pick (or create) the target Notion database

After proceeding, the agent searches your Notion workspace and lists candidate databases as buttons:

- Click one of the listed databases, **or**
- Click **➕ Create a new database** — you'll be asked for a name and which existing Notion page it should live in; the agent creates it there with a title, Owner, and Due date property, **or**
- Click **❌ None of these / Cancel** to abort

### 5. Confirm before anything is written

The agent reads the chosen database's real schema and maps your task fields (title/owner/due date/status) onto its actual property names. You'll see:

- How many tasks are brand-new vs. already exist in the database (see below)
- The field mapping it resolved
- **✅ Create tasks** / **✏️ Keep editing** / **❌ Cancel**

**Nothing is written to Notion before you click "Create tasks."**

### 6. Duplicate detection

Before writing, the agent checks whether each task already exists in the target database (by title match). For tasks that already exist, it **updates only their progress/status property** instead of creating a duplicate page; brand-new tasks are created as new pages. The final result lists each task as `created`, `progress updated`, or `already up to date`.

---

## Module layout

- `app.py` — Chainlit entry point: chat flow/state machine, MCP connect/disconnect handlers, all human-in-the-loop confirmation gates
- `meeting_agent/config.py` — loads `.env` (`FPT_API_KEY`, `FPT_BASE_URL`, `LLM_MODEL`)
- `meeting_agent/models.py` — `Task`/`TaskList` Pydantic models (the extraction domain model)
- `meeting_agent/state.py` — per-session `SessionState` (stage, tasks, MCP tool cache, resolved mapping, etc.)
- `meeting_agent/langchain_llm.py` — `FptStructuredRunnable`, a LangChain `Runnable` that gets structured JSON output from the LLM (plain-prompt JSON is the primary strategy, with `response_format=json_schema` as a fallback — see note below)
- `meeting_agent/extraction.py` — `extract_tasks`/`revise_tasks`, LCEL chains built on `FptStructuredRunnable`
- `meeting_agent/agent.py` — `build_notion_agent`/`run_agent`, a LangGraph ReAct agent wrapper used for all Notion tool-calling
- `meeting_agent/mcp_tools.py` — wraps Chainlit's raw MCP `ClientSession` into LangChain tools via `langchain-mcp-adapters`; `call_tool_directly` for one-off tool calls outside the agent loop
- `meeting_agent/notion_models.py` — Pydantic models for the Notion-specific flow (`PropertyMapping`, `DataSourceCandidates`, `NotionCreatePagesArgs`, etc.)
- `meeting_agent/notion_mapping.py` — Notion-specific logic: database discovery, property mapping resolution, duplicate-task detection, and the actual page create/update calls
- `scripts/smoke_extract.py` — fast extraction-only test loop against `samples/*.txt`, no Chainlit or Notion required

### A note on the LLM quirk this is built around

The FPT Cloud endpoint (`DeepSeek-V4-Flash`) sometimes returns empty or lower-quality JSON when forced into `response_format=json_schema` mode, especially on longer transcripts. `FptStructuredRunnable` therefore asks for JSON via a plain prompt instruction first (proven more reliable through repeated testing) and only falls back to `response_format=json_schema` if that fails to parse. If you swap in a different LLM provider/model, you may be able to simplify or remove this fallback — but verify with real inputs first.

---

## Future Improvements
- Jira integration (add a second connector module alongside `notion_mapping.py`)
- Relation-based Notion dependencies with topological creation order
- Owner-name → Notion user ID resolution for true `people` properties
- Improved dependency detection
- Enhanced task structuring and prioritization
- Triggered by the availability of transcription