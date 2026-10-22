import asyncio
import sys
import uuid
from langchain_core.messages import HumanMessage
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel

import config
from core.graph import ragobot_graph, set_stream_listeners
from core.tts import tts_engine
from ingestion.rag_engine import rag_engine
# pyrefly: ignore[missing-import]
from cli.cyber_console import (
    console,
    print_banner,
    render_token_dashboard,
    render_dual_columns,
    render_consensus_panel,
    render_abstention_panel,
    render_sources_table
)

async def run_query(query: str, thread_id: str, state_tracker: dict) -> None:
    """Executes a multi-agent RAG query through LangGraph with real-time live streaming."""
    buffer_a: list[str] = []
    buffer_b: list[str] = []
    
    # Register streaming listeners for live column updates
    def on_token_a(token: str):
        buffer_a.append(token)
        
    def on_token_b(token: str):
        buffer_b.append(token)
        
    set_stream_listeners(on_token_a, on_token_b)
    
    # Initial live display
    tokens = state_tracker.get("token_count", 0)
    was_trimmed = state_tracker.get("was_trimmed", False)
    
    dashboard = render_token_dashboard(tokens, config.MAX_CONTEXT_TOKENS, was_trimmed)
    dual_columns = render_dual_columns("Initiating parallel compute nodes...", "Waiting for LPU stream...")
    
    live_group = Group(dashboard, dual_columns)
    
    with Live(live_group, console=console, refresh_per_second=12) as live:
        # Background task to refresh the Live display while graph executes
        stop_refresh = False
        
        async def refresh_loop():
            while not stop_refresh:
                col_a = "".join(buffer_a)
                col_b = "".join(buffer_b)
                curr_cols = render_dual_columns(col_a, col_b)
                curr_dash = render_token_dashboard(
                    state_tracker.get("token_count", 0),
                    config.MAX_CONTEXT_TOKENS,
                    state_tracker.get("was_trimmed", False)
                )
                live.update(Group(curr_dash, curr_cols))
                await asyncio.sleep(0.08)
                
        refresh_task = asyncio.create_task(refresh_loop())
        
        # Invoke LangGraph statefully with thread checkpointer
        config_dict = {"configurable": {"thread_id": thread_id}}
        initial_input = {
            "query": query,
            "messages": [HumanMessage(content=query)]
        }
        
        try:
            result = await ragobot_graph.ainvoke(initial_input, config=config_dict)
            stop_refresh = True
            await refresh_task
            
            # Update state tracker
            state_tracker["token_count"] = result.get("token_count", 0)
            state_tracker["was_trimmed"] = result.get("was_trimmed", False)
            
            # Final update inside live context
            final_dash = render_token_dashboard(
                result.get("token_count", 0),
                config.MAX_CONTEXT_TOKENS,
                result.get("was_trimmed", False)
            )
            final_cols = render_dual_columns(
                result.get("ollama_output", ""),
                result.get("groq_output", ""),
                ollama_status="Settled",
                groq_status="Settled"
            )
            live.update(Group(final_dash, final_cols))
        except Exception as e:
            stop_refresh = True
            await refresh_task
            console.print(f"[bold red][!] Graph Execution Error:[/] {e}")
            return
            
    # Render Consensus Arbiter Output or Abstention Protocol
    final_text = result.get("final_response", "")
    is_abstained = result.get("is_abstained", False)
    console.print()
    if is_abstained:
        console.print(render_abstention_panel(final_text, result.get("abstention_reason", "")))
    else:
        console.print(render_consensus_panel(final_text, is_speaking=config.TTS_ENABLED))
    
    # Render Sources Boundary Table
    retrieved_docs = result.get("retrieved_docs", [])
    if retrieved_docs:
        console.print(render_sources_table(retrieved_docs))
    console.print()

def print_help() -> None:
    """Displays command matrix."""
    help_panel = Panel(
        """[bold cyan]/help[/]     - Display command matrix
[bold cyan]/tokens[/]   - Show detailed context window token diagnostic breakdown
[bold cyan]/voice[/]    - Toggle Windows Native Voice (TTS) on / off
[bold cyan]/stop[/]     - Stop currently speaking voice audio
[bold cyan]/reload[/]   - Re-scan and re-index Markdown documentation in data/
[bold cyan]/clear[/]    - Reset current conversation session
[bold cyan]/exit[/]     - Disconnect from RagObot terminal""",
        title="[bold yellow]⚡ RAGOBOT COMMAND MATRIX[/]",
        border_style="yellow"
    )
    console.print(help_panel)

async def interactive_shell() -> None:
    """Main cyber-console prompt loop."""
    console.clear()
    
    # 1. Ingest Markdown documentation
    with console.status("[bold cyan]Scanning and indexing Markdown source boundaries...[/]"):
        doc_count = rag_engine.ingest_documents()
        
    print_banner(doc_count)
    console.print("[dim italic]Type your query or enter [cyan]/help[/cyan] for system commands.[/dim italic]\n")
    
    thread_id = str(uuid.uuid4())
    state_tracker = {
        "token_count": 0,
        "was_trimmed": False,
        "turn_count": 0
    }
    
    while True:
        try:
            query = console.input("[bold cyan]RagObot[/][bold white] ❯ [/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Disengaging RagObot matrix. Goodbye.[/]")
            break
            
        if not query:
            continue
            
        # Command handling
        if query.lower() in ("/exit", "exit", "quit"):
            console.print("[bold yellow]Disengaging RagObot matrix. Goodbye.[/]")
            break
        elif query.lower() == "/help":
            print_help()
            continue
        elif query.lower() == "/voice":
            new_state = tts_engine.toggle()
            status_str = "[bold green]ENABLED[/]" if new_state else "[bold red]MUTED[/]"
            console.print(f"[bold magenta]🔊 Windows Native Voice Dispatch:[/] {status_str}")
            continue
        elif query.lower() == "/stop":
            tts_engine.stop()
            console.print("[dim]Audio stopped.[/dim]")
            continue
        elif query.lower() == "/clear":
            thread_id = str(uuid.uuid4())
            state_tracker = {"token_count": 0, "was_trimmed": False, "turn_count": 0}
            console.print("[bold green]✓ Session state and memory reset. Thread renewed.[/]")
            continue
        elif query.lower() == "/reload":
            with console.status("[bold cyan]Re-indexing Markdown documentation...[/]"):
                new_count = rag_engine.ingest_documents()
            console.print(f"[bold green]✓ Reload complete. {new_count} chunks indexed.[/]")
            continue
        elif query.lower() == "/tokens":
            console.print(render_token_dashboard(
                state_tracker["token_count"],
                config.MAX_CONTEXT_TOKENS,
                state_tracker["was_trimmed"]
            ))
            continue
            
        # Run Multi-Agent RAG Pipeline
        await run_query(query, thread_id, state_tracker)

def main():
    try:
        asyncio.run(interactive_shell())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Shutdown complete.[/]")

if __name__ == "__main__":
    main()
