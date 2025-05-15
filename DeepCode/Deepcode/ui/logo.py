from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box
import time

def print_logo():
    console = Console()
    console.clear()
    version = "v1.0.0"
    copyright_text = "© 2024 DeepCode AI Research"
    # 使用 Text.from_markup 正确渲染富文本
    logo = Text.from_markup('''
[bold cyan]██████╗ ███████╗███████╗██████╗      ██████╗ ██████╗ ██████╗ ███████╗[/bold cyan]
[bold blue]██╔══██╗██╔════╝██╔════╝██╔══██╗    ██╔════╝██╔═══██╗██╔══██╗██╔════╝[/bold blue]
[bold cyan]██║  ██║█████╗  █████╗  ██║  ██║    ██║     ██║   ██║██║  ██║█████╗  [/bold cyan]
[bold blue]██║  ██║██╔══╝  ██╔══╝  ██║  ██║    ██║     ██║   ██║██║  ██║██╔══╝  [/bold blue]
[bold cyan]██████╔╝███████╗███████╗██████╔╝    ╚██████╗╚██████╔╝██████╔╝███████╗[/bold cyan]
[bold blue]╚═════╝ ╚══════╝╚══════╝╚═════╝      ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝[/bold blue]
''', justify="center")
    tagline = Text.from_markup("[bold white]Intelligent Code Analysis & Generation Platform[/bold white]", justify="center")
    features = [
        "[cyan]•[/cyan] Advanced Code Understanding",
        "[cyan]•[/cyan] Intelligent Code Generation",
        "[cyan]•[/cyan] Automated Debugging",
        "[cyan]•[/cyan] Performance Optimization",
        "[cyan]•[/cyan] Smart Refactoring"
    ]
    features_panel = Panel(
        "\n".join(features),
        box=box.ROUNDED,
        border_style="cyan",
        padding=(1, 2),
        title="[bold white]Key Features[/bold white]"
    )
    logo_panel = Panel(logo, box=box.ROUNDED, border_style="cyan", padding=(1, 2))
    tagline_panel = Panel(tagline, box=box.ROUNDED, border_style="blue", padding=(0, 2))
    console.print("\n")
    console.print(logo_panel)
    console.print("\n")
    console.print(tagline_panel)
    console.print("\n")
    console.print(features_panel)
    console.print("\n")
    console.print(f"[dim]{version} | {copyright_text}[/dim]", justify="center")
    console.print("\n")
    with console.status("[bold green]Initializing DeepCode AI...[/bold green]", spinner="dots"):
        time.sleep(1.5) 