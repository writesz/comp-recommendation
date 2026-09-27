"""Invariants of the public Hugging Face release.

This release leaves the machine and carries the project's name, so the claims
that matter are the ones about what is *not* in it: no verbatim problem
statements (the platforms' copyright) and no user interaction data (real
Codeforces handles). The rest guards the provenance columns a downstream user
would otherwise have no way to check.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.export_hf_dataset import DEFAULT_REPO_ID, build_table, card, stats
from scripts.report_stats import TAXONOMY
from scripts.upload_hf_dataset import ALLOWED, check_payload

DATA = Path(__file__).resolve().parents[1] / "data"

needs_data = pytest.mark.skipif(
    not (DATA / "cprs_unified_tagged.csv").exists(),
    reason="catalogue not built; run scripts/build_dataset.py",
)

pytestmark = needs_data


@pytest.fixture(scope="module")
def table():
    return build_table()


# --- what must not be published ---------------------------------------------

def test_no_statement_text_column(table):
    """Problem statements are the platforms' copyright and are never released."""
    assert not any("statement" in c.lower() or "body" in c.lower() for c in table.columns)


def test_no_free_text_long_enough_to_be_a_statement(table):
    """Guard against a statement sneaking in through some other column."""
    longest = max(table[c].astype(str).str.len().max() for c in table.columns)
    assert longest < 500, "a column holds text long enough to be a problem statement"


def test_no_user_or_handle_column(table):
    """The interaction cohort is used in aggregate only; no handle is published."""
    forbidden = ("handle", "user", "submission", "solve_time")
    assert not any(f in c.lower() for c in table.columns for f in forbidden)


def test_upload_payload_is_allowlisted(tmp_path, monkeypatch):
    """An extra file in the release directory aborts the upload."""
    import scripts.upload_hf_dataset as up

    for name in ALLOWED:
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(up, "OUT", tmp_path)
    assert check_payload() == ALLOWED

    (tmp_path / "cf_submissions.jsonl").write_text("{}")
    with pytest.raises(SystemExit, match="refusing to upload"):
        check_payload()


def test_upload_payload_rejects_incomplete_release(tmp_path, monkeypatch):
    import scripts.upload_hf_dataset as up

    (tmp_path / "README.md").write_text("x")
    monkeypatch.setattr(up, "OUT", tmp_path)
    with pytest.raises(SystemExit, match="incomplete"):
        check_payload()


# --- provenance the card promises -------------------------------------------

def test_predicted_tags_are_atcoder_only(table):
    """Only AtCoder lacked tags, so only AtCoder rows may carry model output."""
    assert set(table.loc[table["tags_predicted"], "platform"]) == {"atcoder"}


@pytest.mark.skipif(
    not (DATA / "report_stats.json").exists(),
    reason="derived stats not built; run scripts/report_stats.py",
)
def test_predicted_flag_matches_tagger_coverage(table):
    """The flag must agree with the coverage the report states independently."""
    with open(DATA / "report_stats.json") as f:
        expected = json.load(f)["tagger_coverage"]["atcoder"]["topic_tagged_after"]
    assert int(table["tags_predicted"].sum()) == expected


def test_predicted_rows_actually_have_tags(table):
    assert table.loc[table["tags_predicted"], "tags_unified"].map(len).gt(0).all()


def test_difficulty_source_present_exactly_when_difficulty_is(table):
    """A scale label without a value, or a value on no known scale, is a bug."""
    assert (table["difficulty_raw"].notna() == table["difficulty_source"].notna()).all()


def test_normalized_difficulty_in_unit_interval(table):
    d = table["difficulty_normalized"].dropna()
    assert d.between(0.0, 1.0).all()


def test_canonical_topic_flag_ignores_other_prefixed_labels(table):
    """``other:abc`` is a contest name, not a topic — it must not count."""
    only_other = table[
        table["tags_unified"].map(lambda ts: bool(ts) and not any(t in TAXONOMY for t in ts))
    ]
    assert not only_other.empty, "fixture no longer exercises the other: case"
    assert not only_other["has_canonical_topic"].any()


def test_ids_unique_and_well_formed(table):
    assert table["cprs_id"].is_unique
    assert (table["cprs_id"].str.split(":").str[0].isin(
        {"cf", "ac", "lc"}
    )).all()


# --- the card is generated, not transcribed ---------------------------------

def test_card_quotes_the_real_row_count(table):
    s = stats(table)
    text = card(s, {"strings": 0.74}, 0.44, DEFAULT_REPO_ID)
    assert f"{len(table):,}" in text
    assert str(s["n_canonical_topics"]) in text


def test_card_names_the_target_repo(table):
    text = card(stats(table), {"strings": 0.74}, 0.44, "someone/cprs")
    assert 'load_dataset("someone/cprs"' in text
