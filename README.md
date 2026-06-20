# A.E.G.I.S

<div align="center">

# 🛡️ A.E.G.I.S
### Adaptive Executive General Intelligence System

*"The shield that thinks, learns, and strikes."*

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)
![Status](https://img.shields.io/badge/Status-Under%20Development-yellow)

</div>

---

# Overview

**A.E.G.I.S (Adaptive Executive General Intelligence System)** is a personal AI operating system built around a self-expanding architecture.

Unlike traditional AI assistants, AEGIS is designed to become the intelligent layer between the user and every digital task. It reasons, delegates, learns, manages knowledge, and executes long-horizon projects through specialized agents and atomic skills.

AEGIS follows a:

```text
Brain
 ↓
Agents
 ↓
Skills
 ↓
Knowledge
 ↓
Projects
```

architecture.

---

# Architecture

## 🧠 Brain

The Brain is the orchestrator.

Responsibilities:

- Task understanding
- Planning
- Capability routing
- Replanning on failures
- Session context management
- Knowledge-aware reasoning

The Brain never executes anything directly.

---

## 🤖 Agents

Agents are domain specialists.

Current agents:

- File Agent
- Search Agent
- Browser Agent
- Code Agent
- Project Agent
- Knowledge Agent
- Writing Agent

Each agent:

- Has reasoning capability
- Owns a specific domain
- Maintains a mutable skill list
- Tracks performance
- Evolves over time

---

## ⚡ Skills

Skills are atomic capabilities.

Examples:

### File Skills

- read_file()
- write_file()
- move_file()
- search_files()

### Search Skills

- web_search()
- semantic_search()

### Browser Skills

- fetch_page()
- screenshot()
- fill_form()

### Code Skills

- run_python()
- validate_syntax()

### Knowledge Skills

- add_entity()
- find_skill_gaps()
- multi_hop_traverse()

### Project Skills

- create_project()
- checkpoint_task()
- resume_task()

Skills:

- are versioned
- schema validated
- runtime monitored
- permission aware

---

## 📚 Knowledge Layer

AEGIS uses a hybrid KAG architecture:

### Knowledge Graph

**Kuzu**

Stores:

- preferences
- skills
- goals
- relationships
- task history
- structured facts

### Vector Store

**ChromaDB**

Stores:

- documents
- notes
- PDFs
- research papers

### Task Store

**MongoDB**

Stores:

- task history
- corrections
- performance signals

---

## 📂 Project Layer

Long-horizon execution.

Supports:

- Multi-session tasks
- Checkpointing
- Resume after interruption
- Dependency tracking
- Structured plans

Examples:

- Job applications
- Building software projects
- Deep research campaigns

---

# Factory System

AEGIS can expand itself.

## Skill Factory

Creates new skills when missing capabilities are detected.

Flow:

```text
Need detected
↓
Generate Python function
↓
Validate
↓
Sandbox test
↓
Register
```

---

## Agent Factory

Creates new agents dynamically.

Flow:

```text
Unknown domain
↓
Generate AgentSpec
↓
Validate
↓
Smoke test
↓
Register
```

---

# Tech Stack

## Backend

- Python 3.11+
- FastAPI
- LangGraph
- Celery
- Redis

## AI

- Ollama
- Claude
- Whisper
- ElevenLabs

## Knowledge

- Kuzu
- ChromaDB
- MongoDB

## Automation

- Playwright
- Docker Sandbox
- APScheduler

## Frontend

- React
- Tauri
- React Native
- WebSockets

---

# Project Structure

```text
aegis/
│
├── main.py
├── config.py
│
├── backend/
│   ├── brain/
│   ├── agents/
│   ├── skills/
│   ├── registry/
│   ├── factory/
│   ├── knowledge/
│   ├── projects/
│   ├── memory/
│   ├── perception/
│   └── sandbox/
│
├── registry_store/
│   ├── agents.json
│   ├── skills.json
│   ├── pipelines.json
│   └── projects.json
│
└── frontend/
```

---

# Build Roadmap

## Phase 1 — Foundation + Factories

- Brain
- Registries
- Agent Factory
- Skill Factory
- Persistent Task Object

---

## Phase 2 — Knowledge Layer

- Kuzu Graph
- Knowledge Agent
- Entity Resolution
- Context Assembly

---

## Phase 3 — Intelligence

- Memory
- Scoring
- Evolution
- Preference Learning

---

## Phase 4 — Project Layer

- Project Agent
- Checkpointing
- Resume Logic

---

## Phase 5 — Perception + Voice

- Whisper
- ElevenLabs
- Screen Vision
- Clipboard Monitor
- Calendar Awareness

---

## Phase 6 — Completion

- Desktop GUI
- Mobile App
- Offline Mode
- Proactive Intelligence

---

# Current Status

🚧 Under Development

Current Focus:

```text
Brain
↓
Agent Factory
↓
Skill Factory
↓
Registry System
```

---

# Guiding Principles

- Local-first
- Single-user
- Modular
- Self-expanding
- Safe by default
- Transparent
- Permission aware

---

# Future Expansion

- Full computer control
- Multi-user support
- Agent marketplace
- Smart home integration
- Advanced vision
- Enterprise connectors
- Autonomous workflows

---

# Vision

> A.E.G.I.S is not a chatbot.

It is a personal AI operating system that learns how you work, expands its own capabilities, and becomes the intelligent layer between you and every digital task.

---

<div align="center">

# 🛡️ A.E.G.I.S

### Adaptive Executive General Intelligence System

*"The shield that thinks, learns, and strikes."*

</div>
