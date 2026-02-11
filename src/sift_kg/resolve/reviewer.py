"""Interactive terminal review for merge proposals and flagged relations.

Presents DRAFT items one-by-one with Rich panels. User approves, rejects,
or skips each item. Updated files are written on completion.
"""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sift_kg.resolve.models import MergeFile, RelationReviewFile, StatusType

console = Console()


def _read_key(prompt: str, valid: str = "arsq") -> str:
    """Read a single valid key from stdin.

    Args:
        prompt: Prompt text to display
        valid: String of valid key characters

    Returns:
        The key pressed (lowercase)
    """
    console.print(prompt, end="")
    while True:
        try:
            line = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "q"
        if line and line[0] in valid:
            return line[0]
        console.print(f"  [dim]Press one of: {', '.join(valid)}[/dim] ", end="")


def review_merges(merge_file: MergeFile) -> dict[str, int]:
    """Interactively review DRAFT merge proposals.

    Modifies merge_file in place, setting status to CONFIRMED or REJECTED.

    Args:
        merge_file: MergeFile with proposals to review

    Returns:
        Stats dict with counts of approved, rejected, skipped
    """
    drafts = merge_file.draft
    if not drafts:
        console.print("[dim]No merge proposals to review.[/dim]")
        return {"approved": 0, "rejected": 0, "skipped": 0}

    total = len(drafts)
    stats = {"approved": 0, "rejected": 0, "skipped": 0}

    console.print()
    console.print(f"[bold cyan]Entity Merge Review[/bold cyan]  —  {total} proposals to review")
    console.print("[dim]For each proposal, decide whether these entities are the same.[/dim]")
    console.print()

    for i, proposal in enumerate(drafts):
        # Build the panel content
        content = Text()
        content.append(f"Canonical: ", style="bold")
        content.append(f"{proposal.canonical_name}", style="green")
        content.append(f"  ({proposal.canonical_id})\n", style="dim")
        content.append(f"Type: ", style="bold")
        content.append(f"{proposal.entity_type}\n\n")

        # Member table
        member_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        member_table.add_column("Member", style="yellow")
        member_table.add_column("ID", style="dim")
        member_table.add_column("Confidence", justify="right")

        for member in proposal.members:
            conf_style = "green" if member.confidence >= 0.8 else "yellow" if member.confidence >= 0.5 else "red"
            member_table.add_row(
                member.name,
                member.id,
                f"[{conf_style}]{member.confidence:.0%}[/{conf_style}]",
            )

        panel = Panel(
            member_table,
            title=f"[bold]Merge {i + 1}/{total}[/bold]  —  {proposal.canonical_name} ({proposal.entity_type})",
            subtitle=f"[dim]{proposal.reason}[/dim]" if proposal.reason else None,
            border_style="cyan",
            padding=(1, 2),
        )
        console.print(panel)

        # Get user decision
        choice = _read_key("  [a]pprove  [r]eject  [s]kip  [q]uit → ")
        console.print()

        if choice == "a":
            proposal.status = "CONFIRMED"
            stats["approved"] += 1
            console.print(f"  [green]✓ Approved[/green]")
        elif choice == "r":
            proposal.status = "REJECTED"
            stats["rejected"] += 1
            console.print(f"  [red]✗ Rejected[/red]")
        elif choice == "s":
            stats["skipped"] += 1
            console.print(f"  [dim]⏭ Skipped[/dim]")
        elif choice == "q":
            stats["skipped"] += total - i
            console.print(f"  [dim]Quit — skipping remaining {total - i} proposals[/dim]")
            break

        console.print()

    # Summary
    console.print(
        f"[bold]Merge review complete:[/bold]  "
        f"[green]{stats['approved']} approved[/green]  "
        f"[red]{stats['rejected']} rejected[/red]  "
        f"[dim]{stats['skipped']} skipped[/dim]"
    )
    return stats


def review_relations(review_file: RelationReviewFile) -> dict[str, int]:
    """Interactively review DRAFT flagged relations.

    Modifies review_file in place, setting status to CONFIRMED or REJECTED.

    Args:
        review_file: RelationReviewFile with entries to review

    Returns:
        Stats dict with counts of approved, rejected, skipped
    """
    drafts = review_file.draft
    if not drafts:
        console.print("[dim]No flagged relations to review.[/dim]")
        return {"approved": 0, "rejected": 0, "skipped": 0}

    total = len(drafts)
    stats = {"approved": 0, "rejected": 0, "skipped": 0}

    console.print()
    console.print(f"[bold cyan]Relation Review[/bold cyan]  —  {total} flagged relations to review")
    console.print("[dim]For each relation, decide whether it's valid.[/dim]")
    console.print()

    for i, entry in enumerate(drafts):
        # Build relation display
        content = Text()
        content.append(f"{entry.source_name}", style="green")
        content.append(f"  —[{entry.relation_type}]→  ", style="bold cyan")
        content.append(f"{entry.target_name}", style="green")

        conf_style = "green" if entry.confidence >= 0.8 else "yellow" if entry.confidence >= 0.5 else "red"
        subtitle_parts = []
        if entry.flag_reason:
            subtitle_parts.append(entry.flag_reason)
        subtitle_parts.append(f"confidence: {entry.confidence:.0%}")
        if entry.source_document:
            subtitle_parts.append(f"from: {entry.source_document}")

        panel = Panel(
            content,
            title=f"[bold]Relation {i + 1}/{total}[/bold]",
            subtitle=f"[dim]{' │ '.join(subtitle_parts)}[/dim]",
            border_style=conf_style,
            padding=(1, 2),
        )
        console.print(panel)

        if entry.evidence:
            console.print(f"  [dim]Evidence: {entry.evidence}[/dim]")

        # Get user decision
        choice = _read_key("  [a]pprove  [r]eject  [s]kip  [q]uit → ")
        console.print()

        if choice == "a":
            entry.status = "CONFIRMED"
            stats["approved"] += 1
            console.print(f"  [green]✓ Approved[/green]")
        elif choice == "r":
            entry.status = "REJECTED"
            stats["rejected"] += 1
            console.print(f"  [red]✗ Rejected[/red]")
        elif choice == "s":
            stats["skipped"] += 1
            console.print(f"  [dim]⏭ Skipped[/dim]")
        elif choice == "q":
            stats["skipped"] += total - i
            console.print(f"  [dim]Quit — skipping remaining {total - i} relations[/dim]")
            break

        console.print()

    # Summary
    console.print(
        f"[bold]Relation review complete:[/bold]  "
        f"[green]{stats['approved']} approved[/green]  "
        f"[red]{stats['rejected']} rejected[/red]  "
        f"[dim]{stats['skipped']} skipped[/dim]"
    )
    return stats
