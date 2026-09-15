"""
Demo the recommendation engine with a real Codeforces user.

Usage:
    python -m scripts.demo_recommend --handle tourist
    python -m scripts.demo_recommend --handle tourist --platforms codeforces leetcode
"""
import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetchers.codeforces import fetch_user_submissions, fetch_user_info
from models.recommender import RecommenderEngine

console = Console()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def display_profile(profile):
    """Display user profile summary."""
    console.print(Panel(
        f"[bold]{profile.handle}[/bold] (CF rating: {profile.rating or 'N/A'})\n"
        f"Solved: {profile.total_solved} / {profile.total_attempted} "
        f"({profile.overall_solve_rate:.1%})\n"
        f"Difficulty level: {profile.difficulty_level:.3f}",
        title="User Profile",
    ))

    # Topic mastery table
    table = Table(title="Topic Mastery (top 15)")
    table.add_column("Topic", style="cyan")
    table.add_column("Solved/Attempted", justify="right")
    table.add_column("Solve Rate", justify="right")
    table.add_column("Avg Difficulty", justify="right")

    mastery_items = []
    for topic, m in profile.topic_mastery.items():
        if hasattr(m, "problems_attempted"):
            mastery_items.append((topic, m))
        else:
            from models.recommender import TopicMastery
            mastery_items.append((topic, TopicMastery(**m)))

    mastery_items.sort(key=lambda x: x[1].problems_attempted, reverse=True)

    for topic, m in mastery_items[:15]:
        rate_color = "green" if m.solve_rate > 0.7 else "yellow" if m.solve_rate > 0.4 else "red"
        table.add_row(
            topic,
            f"{m.problems_solved}/{m.problems_attempted}",
            f"[{rate_color}]{m.solve_rate:.1%}[/{rate_color}]",
            f"{m.avg_difficulty:.3f}",
        )
    console.print(table)


def display_recommendations(recs, title="Recommendations"):
    """Display recommendations table."""
    table = Table(title=title)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Platform", style="cyan")
    table.add_column("Problem", style="bold")
    table.add_column("Difficulty", justify="right")
    table.add_column("Tags", style="green")
    table.add_column("Score", justify="right")
    table.add_column("Why", style="dim")

    for i, rec in enumerate(recs, 1):
        platform_colors = {
            "codeforces": "blue",
            "atcoder": "cyan",
            "leetcode": "yellow",
        }
        color = platform_colors.get(rec.platform, "white")
        diff_str = f"{rec.difficulty_normalized:.2f}" if rec.difficulty_normalized else "N/A"
        tags_str = ", ".join(rec.tags[:3])
        reason_str = rec.reasons[0] if rec.reasons else ""

        table.add_row(
            str(i),
            f"[{color}]{rec.platform}[/{color}]",
            rec.name,
            diff_str,
            tags_str,
            f"{rec.score:.3f}",
            reason_str,
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Demo CPRS recommender")
    parser.add_argument("--handle", required=True, help="Codeforces username")
    parser.add_argument("--n", type=int, default=20, help="Number of recommendations")
    parser.add_argument("--platforms", nargs="*", default=None, help="Filter platforms")
    args = parser.parse_args()

    # Use tagged dataset if available, otherwise base
    dataset_path = DATA_DIR / "cprs_unified_tagged.json"
    if not dataset_path.exists():
        dataset_path = DATA_DIR / "cprs_unified.json"

    engine = RecommenderEngine(str(dataset_path))

    # Fetch user data
    console.print(f"\n[bold]Fetching data for [cyan]{args.handle}[/cyan]...[/bold]\n")
    try:
        user_info = fetch_user_info(args.handle)
        submissions = fetch_user_submissions(args.handle)
    except Exception as e:
        console.print(f"[red]Error fetching user data: {e}[/red]")
        return

    # Build profile
    profile = engine.build_user_profile(
        submissions, args.handle, rating=user_info.get("rating")
    )
    display_profile(profile)

    # Get recommendations
    console.print()
    recs = engine.recommend(profile, n=args.n, platforms=args.platforms)
    display_recommendations(recs, title=f"Top {len(recs)} Recommendations for {args.handle}")

    # Show cross-platform breakdown
    from collections import Counter
    platform_counts = Counter(r.platform for r in recs)
    console.print(f"\nPlatform mix: {dict(platform_counts)}")

    # Also show recommendations filtered per platform
    if not args.platforms:
        for platform in ["leetcode", "atcoder"]:
            cross_recs = engine.recommend(profile, n=5, platforms=[platform])
            if cross_recs:
                display_recommendations(
                    cross_recs,
                    title=f"Top {len(cross_recs)} from {platform.title()}"
                )


if __name__ == "__main__":
    main()
