"""Mind Map CLI - Typer-based command line interface."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="mind-map",
    help="Knowledge Graph-based Mind Map with intelligent context management",
    no_args_is_help=True,
)
console = Console()

# Model management subcommand group
model_app = typer.Typer(help="Manage Ollama models for processing LLM")
app.add_typer(model_app, name="model")


# ============== Help Command ==============


@app.command(name="help")
def show_help() -> None:
    """Show comprehensive help with all commands and options."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Mind Map CLI[/bold cyan]\n"
        "[dim]Knowledge Graph-based Mind Map with intelligent context management[/dim]",
        border_style="cyan",
    ))

    # Main Commands
    console.print("\n[bold yellow]MAIN COMMANDS[/bold yellow]\n")

    main_table = Table(show_header=True, header_style="bold magenta", box=None)
    main_table.add_column("Command", style="cyan", width=20)
    main_table.add_column("Description", style="white")

    main_table.add_row("init", "Initialize database and configuration")
    main_table.add_row("memo", "Ingest a note into the knowledge graph")
    main_table.add_row("ask", "Query the knowledge graph with RAG")
    main_table.add_row("stats", "Display knowledge graph statistics")
    main_table.add_row("serve", "Start FastAPI server for frontend")
    main_table.add_row("ollama-init", "Initialize Ollama processing model")
    main_table.add_row("model", "Manage Ollama models (subcommands)")
    main_table.add_row("help", "Show this help message")

    console.print(main_table)

    # Init Command Options
    console.print("\n[bold yellow]COMMAND OPTIONS[/bold yellow]")

    console.print("\n[bold green]init[/bold green] - Initialize database")
    init_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    init_table.add_column("Option", style="cyan", width=25)
    init_table.add_column("Short", style="dim", width=8)
    init_table.add_column("Description")

    init_table.add_row("--data-dir PATH", "-d", "Directory for database storage [default: ./data]")
    init_table.add_row("--with-ollama", "-o", "Also initialize Ollama with selected model")
    console.print(init_table)

    # Memo Command Options
    console.print("\n[bold green]memo[/bold green] TEXT - Ingest a note")
    memo_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    memo_table.add_column("Option", style="cyan", width=25)
    memo_table.add_column("Short", style="dim", width=8)
    memo_table.add_column("Description")

    memo_table.add_row("TEXT", "", "Text to ingest (required argument)")
    memo_table.add_row("--source TEXT", "-s", "Source identifier for the note")
    memo_table.add_row("--data-dir PATH", "-d", "Directory for database storage [default: ./data]")
    memo_table.add_row("--llm / --no-llm", "", "Use LLM for processing [default: --llm]")
    memo_table.add_row("--model TEXT", "-m", "Specific Ollama model to use")
    console.print(memo_table)

    # Ask Command Options
    console.print("\n[bold green]ask[/bold green] QUERY - Query the knowledge graph")
    ask_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    ask_table.add_column("Option", style="cyan", width=25)
    ask_table.add_column("Short", style="dim", width=8)
    ask_table.add_column("Description")

    ask_table.add_row("QUERY", "", "Question to ask (required argument)")
    ask_table.add_row("--depth INT", "-d", "Graph traversal depth [default: 2]")
    ask_table.add_row("--data-dir PATH", "", "Directory for database storage [default: ./data]")
    ask_table.add_row("--model TEXT", "-m", "Specific processing model to use")
    console.print(ask_table)

    # Stats Command Options
    console.print("\n[bold green]stats[/bold green] - Display statistics")
    stats_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    stats_table.add_column("Option", style="cyan", width=25)
    stats_table.add_column("Short", style="dim", width=8)
    stats_table.add_column("Description")

    stats_table.add_row("--data-dir PATH", "-d", "Directory for database storage [default: ./data]")
    console.print(stats_table)

    # Serve Command Options
    console.print("\n[bold green]serve[/bold green] - Start API server")
    serve_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    serve_table.add_column("Option", style="cyan", width=25)
    serve_table.add_column("Short", style="dim", width=8)
    serve_table.add_column("Description")

    serve_table.add_row("--host TEXT", "-h", "Host to bind to [default: 127.0.0.1]")
    serve_table.add_row("--port INT", "-p", "Port to bind to [default: 8000]")
    console.print(serve_table)

    # Ollama-init Command Options
    console.print("\n[bold green]ollama-init[/bold green] - Initialize Ollama model")
    ollama_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    ollama_table.add_column("Option", style="cyan", width=25)
    ollama_table.add_column("Short", style="dim", width=8)
    ollama_table.add_column("Description")

    ollama_table.add_row("--model TEXT", "-m", "Specific model to initialize")
    ollama_table.add_row("--interactive", "-i", "Interactive model selection")
    ollama_table.add_row("--pull", "", "Auto-pull model if not available")
    console.print(ollama_table)

    # Model Subcommands
    console.print("\n[bold yellow]MODEL SUBCOMMANDS[/bold yellow]")
    console.print("[dim]Usage: mind-map model <subcommand>[/dim]\n")

    model_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    model_table.add_column("Subcommand", style="cyan", width=25)
    model_table.add_column("Description")

    model_table.add_row("list", "List available Ollama models with recommendations")
    model_table.add_row("get", "Show currently selected processing model")
    model_table.add_row("set MODEL [--persist]", "Set processing model (use -p to save to config)")
    model_table.add_row("pull MODEL", "Download an Ollama model")
    model_table.add_row("select [--persist]", "Interactive model selection (use -p to save)")
    console.print(model_table)

    # Examples
    console.print("\n[bold yellow]EXAMPLES[/bold yellow]\n")

    examples = """[dim]# Initialize and setup[/dim]
mind-map init                           # Initialize database
mind-map init --with-ollama             # Initialize with Ollama model
mind-map model list                     # List available models
mind-map model set phi3.5 --persist     # Set and save model choice

[dim]# Ingesting notes[/dim]
mind-map memo "Python is great"         # Ingest with LLM processing
mind-map memo "Note" --no-llm           # Ingest without LLM (heuristic)
mind-map memo "Note" --model mistral    # Use specific model

[dim]# Querying[/dim]
mind-map ask "What is Python?"          # Query knowledge graph
mind-map ask "topic" --depth 3          # Query with deeper traversal

[dim]# Server[/dim]
mind-map serve                          # Start on localhost:8000
mind-map serve --port 3000              # Start on custom port"""

    console.print(examples)

    # Configuration
    console.print("\n[bold yellow]CONFIGURATION[/bold yellow]\n")
    config_info = """[dim]Config file:[/dim] config.yaml
[dim]Environment:[/dim] .env (API keys: GOOGLE_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY)
[dim]Data storage:[/dim] ./data/ (ChromaDB + SQLite)

[bold]Processing LLM (B):[/bold] Cloud APIs (Gemini/Anthropic/OpenAI) with Ollama fallback
[bold]Reasoning LLM (A):[/bold] Claude CLI / Cloud APIs - for response generation"""

    console.print(config_info)
    console.print()


@model_app.command(name="help")
def model_help() -> None:
    """Show help for model management commands."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Model Management[/bold cyan]\n"
        "[dim]Manage Ollama models for processing LLM (LLM-B)[/dim]",
        border_style="cyan",
    ))

    console.print("\n[bold yellow]SUBCOMMANDS[/bold yellow]\n")

    # List
    console.print("[bold green]model list[/bold green]")
    console.print("  List all available Ollama models with installation status")
    console.print("  Shows recommended models for processing tasks")
    console.print("  [dim]Example: mind-map model list[/dim]\n")

    # Get
    console.print("[bold green]model get[/bold green]")
    console.print("  Show the currently selected processing model")
    console.print("  [dim]Example: mind-map model get[/dim]\n")

    # Set
    console.print("[bold green]model set[/bold green] MODEL_NAME [OPTIONS]")
    set_table = Table(show_header=False, box=None, padding=(0, 2))
    set_table.add_column("Option", style="cyan", width=20)
    set_table.add_column("Description")
    set_table.add_row("MODEL_NAME", "Name of the model to set (required)")
    set_table.add_row("--persist, -p", "Save selection to config.yaml")
    console.print(set_table)
    console.print("  [dim]Example: mind-map model set phi3.5 --persist[/dim]\n")

    # Pull
    console.print("[bold green]model pull[/bold green] MODEL_NAME")
    console.print("  Download an Ollama model from the registry")
    console.print("  [dim]Example: mind-map model pull mistral[/dim]\n")

    # Select
    console.print("[bold green]model select[/bold green] [OPTIONS]")
    select_table = Table(show_header=False, box=None, padding=(0, 2))
    select_table.add_column("Option", style="cyan", width=20)
    select_table.add_column("Description")
    select_table.add_row("--persist, -p", "Save selection to config.yaml")
    console.print(select_table)
    console.print("  [dim]Example: mind-map model select --persist[/dim]\n")

    # Recommended Models
    console.print("[bold yellow]RECOMMENDED MODELS[/bold yellow]\n")

    rec_table = Table(show_header=True, header_style="bold", box=None)
    rec_table.add_column("Model", style="cyan")
    rec_table.add_column("Size", style="green")
    rec_table.add_column("Description")

    rec_table.add_row("phi3.5", "~2 GB", "Microsoft Phi-3.5 - Fast, good quality (default)")
    rec_table.add_row("phi3", "~2 GB", "Microsoft Phi-3 - Slightly older version")
    rec_table.add_row("llama3.2", "~2 GB", "Meta Llama 3.2 - Good general purpose")
    rec_table.add_row("mistral", "~4 GB", "Mistral 7B - Higher quality, slower")
    rec_table.add_row("gemma2:2b", "~1.5 GB", "Google Gemma 2 - Lightweight")
    rec_table.add_row("qwen2.5:3b", "~2 GB", "Alibaba Qwen 2.5 - Multilingual")
    console.print(rec_table)
    console.print()


@app.command()
def init(
    data_dir: Annotated[
        Path, typer.Option("--data-dir", "-d", help="Directory for database storage")
    ] = Path("./data"),
    with_ollama: Annotated[
        bool, typer.Option("--with-ollama", "-o", help="Also initialize Ollama with phi3.5")
    ] = False,
) -> None:
    """Initialize the Mind Map database and configuration."""
    from mind_map.core.graph_store import GraphStore

    data_dir.mkdir(parents=True, exist_ok=True)
    store = GraphStore(data_dir)
    store.initialize()
    console.print(Panel(f"[green]Mind Map initialized at {data_dir}[/green]"))

    if with_ollama:
        from mind_map.core.llm import initialize_ollama
        initialize_ollama()


@app.command(name="ollama-init")
def ollama_init(
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Specific model to initialize")
    ] = None,
    interactive: Annotated[
        bool, typer.Option("--interactive", "-i", help="Interactive model selection")
    ] = False,
    auto_pull: Annotated[
        bool, typer.Option("--pull", help="Auto-pull model if not available")
    ] = False,
) -> None:
    """Initialize Ollama and set up a processing model."""
    from mind_map.core.llm import (
        check_ollama_available,
        ensure_model_available,
        get_selected_model,
    )

    if not check_ollama_available():
        console.print("[red]Ollama is not running.[/red]")
        console.print("[dim]Start Ollama with: ollama serve[/dim]")
        raise typer.Exit(1)

    target_model = model or get_selected_model()
    console.print(f"[yellow]Ensuring {target_model} model is available...[/yellow]")

    result = ensure_model_available(
        target_model,
        auto_pull=auto_pull,
        interactive=interactive,
    )

    if result:
        console.print(Panel(f"[green]Ollama ready with model: {result}[/green]"))
    else:
        console.print("[red]Failed to initialize Ollama model[/red]")
        console.print("[dim]Try: mind-map model list[/dim]")
        raise typer.Exit(1)


# ============== Model Management Commands ==============


@model_app.command(name="list")
def model_list() -> None:
    """List available Ollama models."""
    from mind_map.core.llm import list_models

    list_models(show_recommended=True)


@model_app.command(name="set")
def model_set(
    model_name: Annotated[str, typer.Argument(help="Model name to set as default")],
    persist: Annotated[
        bool, typer.Option("--persist", "-p", help="Save to config.yaml for future sessions")
    ] = False,
) -> None:
    """Set the processing model to use."""
    from mind_map.core.llm import set_processing_model

    set_processing_model(model_name, persist=persist)


@model_app.command(name="get")
def model_get() -> None:
    """Show the currently selected processing model."""
    from mind_map.core.llm import get_selected_model

    model = get_selected_model()
    console.print(f"[bold]Current processing model:[/bold] {model}")


@model_app.command(name="pull")
def model_pull(
    model_name: Annotated[str, typer.Argument(help="Model name to pull")],
) -> None:
    """Pull/download an Ollama model."""
    from mind_map.core.llm import check_ollama_available, pull_model

    if not check_ollama_available():
        console.print("[red]Ollama is not running.[/red]")
        console.print("[dim]Start Ollama with: ollama serve[/dim]")
        raise typer.Exit(1)

    console.print(f"[yellow]Pulling model: {model_name}[/yellow]")
    if pull_model(model_name, show_progress=True):
        console.print(Panel(f"[green]Model {model_name} is ready[/green]"))
    else:
        console.print(f"[red]Failed to pull model: {model_name}[/red]")
        raise typer.Exit(1)


@model_app.command(name="select")
def model_select(
    persist: Annotated[
        bool, typer.Option("--persist", "-p", help="Save selection to config.yaml")
    ] = False,
) -> None:
    """Interactively select a processing model."""
    from mind_map.core.llm import check_ollama_available, select_model_interactive, set_processing_model

    if not check_ollama_available():
        console.print("[red]Ollama is not running.[/red]")
        console.print("[dim]Start Ollama with: ollama serve[/dim]")
        raise typer.Exit(1)

    selected = select_model_interactive()
    if selected:
        set_processing_model(selected, persist=persist)
    else:
        console.print("[yellow]No model selected[/yellow]")


@app.command()
def memo(
    text: Annotated[str, typer.Argument(help="Text to ingest into the knowledge graph")],
    source: Annotated[
        str | None, typer.Option("--source", "-s", help="Source identifier")
    ] = None,
    data_dir: Annotated[
        Path, typer.Option("--data-dir", "-d", help="Directory for database storage")
    ] = Path("./data"),
    use_llm: Annotated[
        bool, typer.Option("--llm/--no-llm", help="Use LLM for processing (requires Ollama)")
    ] = True,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Specific Ollama model to use")
    ] = None,
) -> None:
    """Ingest a note or thought into the knowledge graph."""
    from mind_map.agents.pipeline import ingest_memo
    from mind_map.core.graph_store import GraphStore

    if not data_dir.exists():
        console.print("[red]Database not initialized. Run 'mind-map init' first.[/red]")
        raise typer.Exit(1)

    store = GraphStore(data_dir)
    store.initialize()

    # Try to get LLM if requested
    llm = None
    if use_llm:
        from mind_map.core.llm import get_processing_llm
        llm = get_processing_llm(model_name=model)
        if not llm:
            console.print("[dim]LLM not available, using heuristic processing[/dim]")

    console.print("[yellow]Processing memo...[/yellow]")

    success, message, node_ids = ingest_memo(text, store, llm=llm, source_id=source)

    if success:
        console.print(f"[green]{message}[/green]")
        if node_ids:
            ids_display = ", ".join(node_ids[:3]) + ("..." if len(node_ids) > 3 else "")
            console.print(f"[dim]Node IDs: {ids_display}[/dim]")
    else:
        console.print(f"[yellow]{message}[/yellow]")


@app.command()
def ask(
    query: Annotated[str, typer.Argument(help="Question to ask the knowledge graph")],
    depth: Annotated[int, typer.Option("--depth", "-d", help="Graph traversal depth")] = 2,
    data_dir: Annotated[
        Path, typer.Option("--data-dir", help="Directory for database storage")
    ] = Path("./data"),
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Specific processing model to use")
    ] = None,
) -> None:
    """Query the knowledge graph with RAG-enhanced response."""
    from mind_map.agents.response_generator import ResponseGenerator
    from mind_map.core.graph_store import GraphStore
    from mind_map.core.llm import get_reasoning_llm

    if not data_dir.exists():
        console.print("[red]Database not initialized. Run 'mind-map init' first.[/red]")
        raise typer.Exit(1)

    store = GraphStore(data_dir)
    store.initialize()

    console.print("[yellow]Querying knowledge graph...[/yellow]")

    # Retrieve relevant nodes via similarity search
    nodes = store.query_similar(query, n_results=5)

    if not nodes:
        console.print(Panel("[dim]No relevant context found in knowledge graph.[/dim]"))
        return

    # Enrich with relation factors and re-sort by combined score
    nodes = store.enrich_context_nodes(nodes)

    console.print(f"[dim]Found {len(nodes)} relevant nodes[/dim]")

    # Get reasoning LLM for response generation
    llm = get_reasoning_llm()
    if not llm:
        console.print("[yellow]Reasoning LLM not available. Showing raw context only.[/yellow]")
        for i, node in enumerate(nodes, 1):
            console.print(f"[cyan][{i}][/cyan] {node.document[:200]}...")
        return

    console.print("[dim]Generating response with reasoning LLM...[/dim]")

    # Generate response using ResponseGenerator
    generator = ResponseGenerator(llm)
    response = generator.generate_sync(query, nodes)

    # Log the Q&A exchange
    console.print(Panel(query, title="[cyan]Question[/cyan]", border_style="cyan"))
    console.print(Panel(response, title="[green]Response (LLM-A)[/green]", border_style="green"))

    # --- LLM(B): interpret Q&A, extract tags/summary, persist to graph ---
    from mind_map.agents.pipeline import ingest_memo
    from mind_map.core.llm import get_processing_llm
    from mind_map.models.schemas import Edge

    # Update interaction timestamps on the context nodes that were used
    for node in nodes:
        store.update_interaction(node.id)

    # Get processing LLM (LLM-B); fall back to heuristic extraction if unavailable
    processing_llm = get_processing_llm(model_name=model)
    if processing_llm:
        console.print("[dim]Summarizing Q&A with processing LLM...[/dim]")
    else:
        console.print("[dim]Processing LLM not available, using heuristic extraction...[/dim]")

    # Ingest Q&A pair through the full pipeline: FilterAgent → KnowledgeProcessor → store
    qa_text = f"Q: {query}\nA: {response}"
    success, message, qa_node_ids = ingest_memo(
        text=qa_text,
        store=store,
        llm=processing_llm,
        source_id=f"qa_{query[:50]}",
    )

    if success and qa_node_ids:
        # Link the new Q&A concept node back to the context nodes it was derived from
        qa_concept_id = qa_node_ids[0]
        for context_node in nodes:
            store.add_edge(Edge(
                source=qa_concept_id,
                target=context_node.id,
                relation_type="derived_from",
            ))
        console.print(f"[dim]Stored: {message} (linked to {len(nodes)} context nodes)[/dim]")
    elif not success:
        console.print(f"[dim]Ingestion skipped: {message}[/dim]")


@app.command()
def stats(
    data_dir: Annotated[
        Path, typer.Option("--data-dir", "-d", help="Directory for database storage")
    ] = Path("./data"),
) -> None:
    """Display knowledge graph statistics."""
    from mind_map.core.graph_store import GraphStore

    if not data_dir.exists():
        console.print("[red]Database not initialized. Run 'mind-map init' first.[/red]")
        raise typer.Exit(1)

    store = GraphStore(data_dir)
    store.initialize()
    data = store.get_stats()

    table = Table(title="Knowledge Graph Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Nodes", str(data["total_nodes"]))
    table.add_row("Total Edges", str(data["total_edges"]))
    table.add_row("Concept Nodes", str(data["concept_nodes"]))
    table.add_row("Entity Nodes", str(data["entity_nodes"]))
    table.add_row("Tag Nodes", str(data["tag_nodes"]))
    table.add_row("Avg. Connections", str(data["avg_connections"]))

    console.print(table)


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", "-h", help="Host to bind to")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to bind to")] = 8000,
) -> None:
    """Start the FastAPI server for frontend integration."""
    import uvicorn

    console.print(f"[green]Starting Mind Map API server at http://{host}:{port}[/green]")
    uvicorn.run("mind_map.api.routes:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    app()
