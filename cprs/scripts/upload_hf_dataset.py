"""Push the CPRS catalogue release to the Hugging Face Hub.

Rebuilds ``data/hf_release/`` (so the card always matches the data and names the
right repo), checks that nothing unpublishable slipped in, then uploads.

Creates a **private** repo by default. Pass ``--public`` when you are ready for
it to be world-readable; visibility can also be flipped later in the Hub UI.

    # authenticate once
    huggingface-cli login

    python scripts/upload_hf_dataset.py --repo-id <username>/cprs-competitive-programming
    python scripts/upload_hf_dataset.py --repo-id <username>/... --public

    # see exactly what would be sent, without touching the network
    python scripts/upload_hf_dataset.py --repo-id <username>/... --dry-run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.export_hf_dataset import OUT, build_release

# The release directory must contain these and nothing else. Problem statements
# are the platforms' copyright and the interaction logs carry real Codeforces
# handles; neither is ours to publish, so the upload refuses to proceed if
# anything unexpected appears here.
ALLOWED = {"cprs_problems.parquet", "README.md", "release_stats.json"}


def check_payload():
    """Fail loudly rather than upload something that should not leave the machine."""
    found = {p.name for p in OUT.iterdir()}
    unexpected = found - ALLOWED
    if unexpected:
        sys.exit(
            f"refusing to upload: unexpected files in {OUT}: "
            f"{', '.join(sorted(unexpected))}\n"
            "Only the catalogue, its card and its statistics may be published."
        )
    missing = ALLOWED - found
    if missing:
        sys.exit(f"release is incomplete, missing: {', '.join(sorted(missing))}")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", required=True, help="e.g. yourname/cprs-competitive-programming")
    ap.add_argument("--public", action="store_true", help="make the dataset world-readable")
    ap.add_argument("--dry-run", action="store_true", help="build and check, but do not upload")
    args = ap.parse_args()

    df, s = build_release(args.repo_id)
    files = check_payload()

    print(f"release ready in {OUT}")
    for name in sorted(files):
        print(f"  {name:24} {(OUT / name).stat().st_size / 1e6:>6.2f} MB")
    print(f"\n{s['n_problems']:,} problems · {s['n_canonical_topics']} topics · "
          f"{s['predicted_tags_total']:,} rows with model-predicted tags")
    print(f"target: {args.repo_id} ({'public' if args.public else 'private'})")

    if args.dry_run:
        print("\ndry run — nothing uploaded.")
        return

    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    try:
        api.whoami()
    except Exception:
        sys.exit("not logged in — run `huggingface-cli login` first.")

    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=not args.public,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(OUT),
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"CPRS catalogue: {s['n_problems']:,} problems across three platforms",
        )
    except HfHubHTTPError as e:
        sys.exit(f"upload failed: {e}")

    print(f"\nhttps://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
