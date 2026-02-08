# Frontend Implementation Plan

## Overview
- Angular web application for providing user interface to visualize network graph that can be interacted with user activities.
- Providing web interface to get user question and display response for the question from backend (reasoning LLM).

## Architecture
- **Framework**: Angular 17+ (utilizing standalone components and Signals).
- **State Management**: Angular Signals for fine-grained reactivity (graph updates, streaming responses).
- **Service Layer**: 
    - `GraphService`: D3.js force simulation management and data binding.
    - `ApiService`: Standardized communication with the FastAPI backend.
- **Design Pattern**: Container - presenter design pattern for UI components.
    - Container components: responsible for data fetching and state management.
    - Presenter components: responsible for UI rendering and user interaction.
- **Visuals**: Modern aesthetics with Dark Mode by default, Glassmorphism for overlays, and smooth micro-animations.

## Components
- **GraphVisualizerComponent**: Main visualization engine.
    - **D3.js Force Layout**: Optimized simulation with zoom/pan support.
    - **Node Types**: Visual distinction between Concepts (large), Entities (medium), and Tags (small/pill-shaped).
    - **Interactions**:
        - Drag-and-drop to organize.
        - Click to focus/inspect node details.
        - Hover to highlight topological paths.
    - **Search**: Integrated autocomplete to jump to specific nodes.
- **ChatOverlayComponent**: Floating interface for LLM interaction.
    - Glassmorphism effect for background transparency.
    - **Markdown Support**: Syntax highlighting for code blocks and formatted text.
    - **Status Indicators**: Granular feedback ("Thinking...", "Processing...", "Ingesting...").
- **InspectorPanelComponent**: Sidebar for detailed metadata.
    - Displays importance scores ($S$), creation dates, and relationship lists.
    - Action buttons for manual node editing or deletion.

## API Integration
- Backend API running at `http://localhost:8000` (configurable via environment).
- Global HTTP Interceptor for error handling and base URL injection.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Health check |
| GET | `/graph` | Full graph (nodes + edges) |
| GET | `/node/{id}` | Node details with edges |
| GET | `/stats` | Graph statistics |
| POST | `/ask` | Query with RAG response |
| POST | `/memo` | Ingest memo via LangGraph pipeline |

## Testing & Verification
- **Unit Testing**: Jest for business logic and component logic.
- **E2E Testing**: Playwright for critical user flows (dragging nodes, sending questions).
- **Integration**: Serving Angular `dist` assets via FastAPI for unified deployment.
