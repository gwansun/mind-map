# Mind Map Project Modularization & Packaging Plan

## 1. Overview
The current project tightly couples data processing (Processing), storage/retrieval (RAG), and user interfaces (CLI/API). This plan outlines how to decouple them into three independent libraries and one integrated application to maximize reusability and maintainability.

## 2. Package Decomposition Structure (Proposed Hierarchy)

### 📦 `mind-map-core` (Common Base)
*   **Role**: Define data models and basic interfaces shared across all packages.
*   **Contents**:
    *   `schemas.py`: Data specifications like `GraphNode`, `Edge`, and `NodeType`.
    *   `llm_base.py`: Common abstraction classes for LLM providers (Gemini, Anthropic, Ollama, etc.).
*   **Dependencies**: None.

### 📦 `mind-map-processor` (Processing LLM Engine)
*   **Role**: Perform input data filtering, entity extraction, tagging, and summarization (Existing LLM-B logic).
*   **Contents**:
    *   `processing_llm.py`: Model management and Ollama/Cloud API provider logic.
    *   `filter_agent.py`: Logic for judging information value (Keep/Discard).
    *   `knowledge_processor.py`: Knowledge extraction and tagging engine.
*   **Dependencies**: `mind-map-core`.

### 📦 `mind-map-rag` (Graph-RAG Engine)
*   **Role**: Manage hybrid storage (ChromaDB + SQLite) and perform Knowledge Graph-based retrieval.
*   **Contents**:
    *   `graph_store.py`: Hybrid dual-write and graph traversal logic.
    *   `importance.py`: Importance scoring logic based on time-decay.
    *   `reasoning_llm.py`: LLM-A management for final response generation.
    *   `response_generator.py`: RAG-enhanced response synthesis agent.
*   **Dependencies**: `mind-map-core`.

### 🚀 `mind-map-app` (Integrated Application)
*   **Role**: Combine the above packages to provide end-user services (CLI/API).
*   **Contents**:
    *   `pipeline.py`: Workflow orchestration using LangGraph.
    *   `cli/main.py`: Typer-based terminal interface.
    *   `api/routes.py`: FastAPI endpoints.
*   **Dependencies**: `mind-map-processor`, `mind-map-rag`.

---

## 3. Implementation Steps

### Step 1: Interface Abstraction (Decoupling)
*   Remove direct dependencies between `ProcessingLLM` and `GraphStore`.
*   Establish abstract interfaces for data exchange.
*   Modularize `config.yaml` so each package only references relevant sections.

### Step 2: Monorepo or Individual Repository Setup
*   Use **Poetry Workspaces** to manage multiple packages within a single repository or split them into separate repositories for PyPI registration.

### Step 3: Namespace Packaging
*   Use namespaces like `mind_map.processor` and `mind_map.rag` to prevent installation conflicts.

---

## 4. Expected Benefits
1.  **Reusability**: Use `mind-map-processor` independently for data preprocessing in other projects.
2.  **Flexibility**: Easily swap the storage backend (e.g., SQLite to Neo4j) by only updating the `mind-map-rag` package.
3.  **Testability**: Independent unit testing for each engine ensures higher system stability.
