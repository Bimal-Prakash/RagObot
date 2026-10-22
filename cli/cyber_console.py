from typing import Any
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.progress_bar import ProgressBar
import config

console = Console()

CYBER_ASCII_BANNER = r"""[bold cyan]
  ██████╗  █████╗  ██████╗  ██████╗ ██████╗  ██████╗ ████████╗
  ██╔══██╗██╔══██╗██╔════╝ ██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝
  ██████╔╝███████║██║  ███╗██║   ██║██████╔╝██║   ██║   ██║   
  ██╔══██╗██╔══██║██║   ██║██║   ██║██╔══██╗██║   ██║   ██║   
  ██║  ██║██║  ██║╚██████╔╝╚██████╔╝██████╔╝╚██████╔╝   ██║   
  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝   
[/][bold magenta]  ═══ [HYPER-CONCURRENT DUAL-ENGINE MULTI-AGENT RAG MATRIX] ═══[/]"""

def print_banner(doc_count: int = 0) -> None:
    """Renders the top cyber-console title and system status cards."""
    console.print(CYBER_ASCII_BANNER)
    
    status_table = Table.grid(expand=True)
    status_table.add_column(ratio=1)
    status_table.add_column(ratio=1)
    status_table.add_column(ratio=1)
    status_table.add_column(ratio=1)
    
    groq_status = "[bold green]ONLINE[/]" if config.GROQ_API_KEY else "[bold yellow]KEY REQ[/]"
    tts_status = "[bold green]ENABLED[/]" if config.TTS_ENABLED else "[bold red]MUTED[/]"
    
    card_ollama = Panel(
        f"[cyan]Engine:[/] Local Matrix\n[cyan]Model:[/] {config.OLLAMA_MODEL}\n[cyan]Status:[/] [bold green]ONLINE[/]",
        title="[bold cyan]1. OLLAMA BRAIN[/]",
        border_style="cyan",
        padding=(0, 1)
    )
    card_groq = Panel(
        f"[yellow]Engine:[/] Ultra-Fast LPU\n[yellow]Model:[/] {config.GROQ_MODEL}\n[yellow]Status:[/] {groq_status}",
        title="[bold yellow]2. GROQ CLOUD[/]",
        border_style="yellow",
        padding=(0, 1)
    )
    card_tts = Panel(
        f"[magenta]Engine:[/] WinSDK SAPI\n[magenta]Pipeline:[/] Headless Audio\n[magenta]Voice:[/] {tts_status}",
        title="[bold magenta]3. WIN TTS VOICE[/]",
        border_style="magenta",
        padding=(0, 1)
    )
    card_rag = Panel(
        f"[green]Source:[/] Markdown (.md)\n[green]Splitter:[/] Header Splitter\n[green]Indexed:[/] {doc_count} Chunks",
        title="[bold green]4. RAG INGESTION[/]",
        border_style="green",
        padding=(0, 1)
    )
    
    status_table.add_row(card_ollama, card_groq, card_tts, card_rag)
    console.print(status_table)
    console.print()

def render_token_dashboard(tokens: int, max_tokens: int = config.MAX_CONTEXT_TOKENS, was_trimmed: bool = False) -> Panel:
    """Builds the rolling context window and token budget diagnostic widget."""
    pct = min(100.0, (tokens / max_tokens) * 100) if max_tokens > 0 else 0
    bar = ProgressBar(total=max_tokens, completed=tokens, width=32)
    
    status_text = Text()
    status_text.append(f" Window Usage: ", style="bold white")
    status_text.append(f"{tokens:,} / {max_tokens:,} tokens ({pct:.1f}%)  ", style="bold cyan")
    
    if was_trimmed:
        alert_badge = Text(" ✂️ ALERT: CONTEXT WINDOW PRUNED (ROLLING RECOVERY ACTIVE) ", style="bold white on red")
    else:
        alert_badge = Text(" ✓ ROLLING MEMORY BALANCED ", style="bold black on green")
        
    dashboard_table = Table.grid(expand=True)
    dashboard_table.add_column(ratio=2)
    dashboard_table.add_column(ratio=2)
    dashboard_table.add_row(Group(status_text, bar), alert_badge)
    
    return Panel(
        dashboard_table,
        title="[bold cyan]📊 TOKEN BUDGET GUARD & ROLLING CONTEXT WINDOW[/]",
        border_style="cyan" if not was_trimmed else "red",
        padding=(0, 1)
    )

def render_dual_columns(
    ollama_text: str,
    groq_text: str,
    ollama_status: str = "Active",
    groq_status: str = "Active"
) -> Table:
    """Builds side-by-side thought stream layout panels."""
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    
    left_content = ollama_text.strip() or "[italic dim]Computing localized verification stream...[/]"
    right_content = groq_text.strip() or "[italic dim]Synthesizing structural reasoning stream...[/]"
    
    panel_left = Panel(
        left_content,
        title=f"[bold cyan]🤖 OLLAMA LOCAL MATRIX[/] [dim]({ollama_status})[/]",
        border_style="cyan",
        height=14
    )
    panel_right = Panel(
        right_content,
        title=f"[bold yellow]⚡ GROQ CLOUD LPU ENGINE[/] [dim]({groq_status})[/]",
        border_style="yellow",
        height=14
    )
    
    grid.add_row(panel_left, panel_right)
    return grid

def render_consensus_panel(synthesis_text: str, is_speaking: bool = False) -> Panel:
    """Builds the final consensus arbiter synthesis panel."""
    tts_badge = " [bold magenta]🔊 [WIN AUDIO DISPATCH ACTIVE][/]" if is_speaking else ""
    return Panel(
        Markdown(synthesis_text),
        title=f"[bold green]⚖️ CONSENSUS ARBITER SYNTHESIS & RESOLUTION[/]{tts_badge}",
        border_style="green",
        padding=(1, 2)
    )

def render_abstention_panel(abstention_text: str, reason: str = "") -> Panel:
    """Builds the abstention panel when evidence is unavailable or answer is not grounded."""
    content = abstention_text
    if reason:
        content += f"\n\n**Reason:** *{reason}*"
    return Panel(
        Markdown(content),
        title="[bold red]🛑 WORKFLOW ABSTENTION PROTOCOL ACTIVATED[/]",
        border_style="red",
        padding=(1, 2)
    )

def render_sources_table(retrieved_docs: list[dict[str, Any]]) -> Panel:
    """Renders table of retrieved RAG Markdown chunks with semantic breadcrumbs."""
    table = Table(box=None, expand=True, show_header=True, header_style="bold magenta")
    table.add_column("#", width=3, style="dim")
    table.add_column("Semantic Breadcrumb", style="cyan", ratio=2)
    table.add_column("Score", width=8, style="yellow")
    table.add_column("Snippet Preview", style="white", ratio=3)
    
    for i, doc in enumerate(retrieved_docs, start=1):
        snippet = doc.get("content", "").replace("\n", " ").strip()[:90] + "..."
        score_val = str(doc.get("score", "0.0"))
        breadcrumb = doc.get("breadcrumb", "Unknown")
        table.add_row(str(i), breadcrumb, score_val, snippet)
        
    return Panel(
        table,
        title="[bold magenta]📄 RETRIEVED MARKDOWN SOURCE BOUNDARIES[/]",
        border_style="magenta",
        padding=(0, 1)
    )
