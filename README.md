# abc-hackathon

# MeetingTasksAgent

## Overview
The Meeting Tasks Agent extracts action items from meeting transcripts and helps users convert them into structured tasks in the appropriate project management system.

The agent bridges the gap between discussion and execution by ensuring that no action items are lost and that follow-up becomes part of the operational workflow.

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
- **Notion**, via the official Notion MCP server — task/page creation, connected live through Chainlit's MCP integration (stdio `npx` server or Notion's hosted MCP)
- **An OpenAI-compatible LLM endpoint** (FPT Cloud AI Marketplace) — transcript extraction and structured-output/tool-calling

### Running it

1. `uv sync`
2. Copy `.env.example` to `.env` and fill in `FPT_API_KEY` (never commit the real key)
3. `uv run chainlit run app.py -w`
4. Paste/attach a transcript, review the extracted tasks, connect a Notion MCP server via the 🔌 icon, then confirm to create the tasks

### Module layout

- `app.py` — Chainlit entry point (chat flow, MCP connect/disconnect handlers, confirmation gate)
- `meeting_agent/extraction.py` — LLM-based task extraction/revision (structured JSON output)
- `meeting_agent/mcp_tools.py` — connector-agnostic MCP tool discovery, dispatch, and the agentic tool-calling loop
- `meeting_agent/notion_mapping.py` — Notion-specific: data-source schema fetch, field→property fuzzy matching, deterministic page creation
- `scripts/smoke_extract.py` — fast extraction-only test loop against `samples/*.txt`, no Chainlit required

---

## Repository Contents (original Power Platform prototype)

- Power Platform solution export (.zip)
- README with architecture and explanation

---

## Notes
The architectural description above (Human-in-the-loop, instruction-based orchestration, modular connectors, tool-based execution) was originally written for the Power Platform prototype and has been carried over as the design basis for this Python reimplementation.

---

## Future Improvements
- Jira integration (add a second connector module alongside `notion_mapping.py`)
- Relation-based Notion dependencies with topological creation order
- Owner-name → Notion user ID resolution for true `people` properties
- Improved dependency detection
- Enhanced task structuring and prioritization
- Triggered by the availability of transcription