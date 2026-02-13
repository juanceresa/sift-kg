"""Interactive terminal review for merge proposals and flagged relations.

Presents DRAFT items one-by-one with Rich panels. User approves, rejects,
or skips each item. Updated files are written on completion.
"""


from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sift_kg.resolve.models import MergeFile, RelationReviewFile

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


def review_merges(merge_file: MergeFile, auto_approve_threshold: float = 0.85) -> dict[str, int]:
    """Interactively review DRAFT merge proposals.

    Modifies merge_file in place, setting status to CONFIRMED or REJECTED.
    Proposals where ALL members have confidence >= auto_approve_threshold
    are automatically confirmed without interactive review.

    Args:
        merge_file: MergeFile with proposals to review
        auto_approve_threshold: Auto-confirm proposals where all members
            meet this confidence. Set to 1.0 to disable auto-approve.

    Returns:
        Stats dict with counts of auto_approved, approved, rejected, skipped
    """
    drafts = merge_file.draft
    if not drafts:
        console.print("[dim]No merge proposals to review.[/dim]")
        return {"auto_approved": 0, "approved": 0, "rejected": 0, "skipped": 0}

    # Auto-approve high-confidence proposals
    auto_approved = []
    manual_review = []
    for proposal in drafts:
        min_conf = min(m.confidence for m in proposal.members)
        if auto_approve_threshold < 1.0 and min_conf >= auto_approve_threshold:
            proposal.status = "CONFIRMED"
            auto_approved.append(proposal)
        else:
            manual_review.append(proposal)

    stats = {"auto_approved": len(auto_approved), "approved": 0, "rejected": 0, "skipped": 0}

    console.print()
    if auto_approved:
        console.print(
            f"[bold green]Auto-approved {len(auto_approved)} proposals[/bold green] "
            f"(all members ≥ {auto_approve_threshold:.0%} confidence)"
        )
        console.print()

    if not manual_review:
        console.print("[dim]No remaining proposals need manual review.[/dim]")
        return stats

    total = len(manual_review)
    console.print(f"[bold cyan]Entity Merge Review[/bold cyan]  —  {total} proposals to review")
    console.print("[dim]For each proposal, decide whether these entities are the same.[/dim]")
    console.print()

    for i, proposal in enumerate(manual_review):
        # Build the panel content
        header = Text()
        header.append("Merge into: ", style="bold")
        header.append(f"{proposal.canonical_name}", style="green")
        header.append(f"  ({proposal.canonical_id})", style="dim")
        header.append("\nType: ", style="bold")
        header.append(f"{proposal.entity_type}")

        # Member table
        member_table = Table(
            show_header=True, header_style="bold", box=None, padding=(0, 2),
            title="Members to merge", title_style="bold yellow",
        )
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

        # Build panel body: header + table + reason
        from rich.console import Group
        parts = [header, Text(""), member_table]
        if proposal.reason:
            parts.append(Text(""))
            reason_text = Text()
            reason_text.append("Reason: ", style="bold")
            reason_text.append(proposal.reason, style="dim")
            parts.append(reason_text)

        panel = Panel(
            Group(*parts),
            title=f"[bold]Merge {i + 1}/{total}[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
        console.print(panel)

        # Get user decision
        choice = _read_key(r"  \[a]pprove  \[r]eject  \[s]kip  \[q]uit → ")
        console.print()

        if choice == "a":
            proposal.status = "CONFIRMED"
            stats["approved"] += 1
            console.print("  [green]✓ Approved[/green]")
        elif choice == "r":
            proposal.status = "REJECTED"
            stats["rejected"] += 1
            console.print("  [red]✗ Rejected[/red]")
        elif choice == "s":
            stats["skipped"] += 1
            console.print("  [dim]⏭ Skipped[/dim]")
        elif choice == "q":
            stats["skipped"] += total - i
            console.print(f"  [dim]Quit — skipping remaining {total - i} proposals[/dim]")
            break

        console.print()

    # Summary
    console.print(
        f"[bold]Merge review complete:[/bold]  "
        f"[green]{stats['auto_approved']} auto-approved[/green]  "
        f"[green]{stats['approved']} approved[/green]  "
        f"[red]{stats['rejected']} rejected[/red]  "
        f"[dim]{stats['skipped']} skipped[/dim]"
    )
    return stats


def review_relations(
    review_file: RelationReviewFile,
    auto_approve_threshold: float = 0.85,
    auto_reject_threshold: float = 0.0,
) -> dict[str, int]:
    """Interactively review DRAFT flagged relations.

    Modifies review_file in place, setting status to CONFIRMED or REJECTED.

    Args:
        review_file: RelationReviewFile with entries to review
        auto_approve_threshold: Auto-confirm relations at or above this confidence.
            Set to 1.0 to disable.
        auto_reject_threshold: Auto-reject relations below this confidence.
            Set to 0.0 to disable.

    Returns:
        Stats dict with counts of approved, rejected, skipped
    """
    drafts = review_file.draft
    if not drafts:
        console.print("[dim]No flagged relations to review.[/dim]")
        return {"approved": 0, "rejected": 0, "skipped": 0}

    # Auto-approve high-confidence relations
    auto_approved = 0
    if auto_approve_threshold < 1.0:
        for entry in drafts:
            if entry.confidence >= auto_approve_threshold:
                entry.status = "CONFIRMED"
                auto_approved += 1
        if auto_approved:
            console.print(
                f"[green]Auto-approved {auto_approved} relations "
                f"(confidence ≥ {auto_approve_threshold:.0%})[/green]"
            )

    # Auto-reject low-confidence relations
    auto_rejected = 0
    if auto_reject_threshold > 0.0:
        for entry in review_file.draft:
            if entry.confidence <= auto_reject_threshold:
                entry.status = "REJECTED"
                auto_rejected += 1
        if auto_rejected:
            console.print(
                f"[red]Auto-rejected {auto_rejected} relations "
                f"(confidence < {auto_reject_threshold:.0%})[/red]"
            )

    # Re-check drafts after auto-approve/reject
    drafts = review_file.draft
    total = len(drafts)
    stats = {"approved": auto_approved, "rejected": auto_rejected, "skipped": 0}

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
        choice = _read_key(r"  \[a]pprove  \[r]eject  \[s]kip  \[q]uit → ")
        console.print()

        if choice == "a":
            entry.status = "CONFIRMED"
            stats["approved"] += 1
            console.print("  [green]✓ Approved[/green]")
        elif choice == "r":
            entry.status = "REJECTED"
            stats["rejected"] += 1
            console.print("  [red]✗ Rejected[/red]")
        elif choice == "s":
            stats["skipped"] += 1
            console.print("  [dim]⏭ Skipped[/dim]")
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
