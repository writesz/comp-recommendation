"""Tests for the CodeChef platform integration.

Covers the three places CodeChef differs from the platforms already
supported, because those are where a regression would be silent:

* its difficulty field is a string carrying sentinel values for unrated
  problems, rather than an absent key;
* its tag fields mix genuine topics with setter usernames, so the taxonomy
  has to act as a whitelist;
* its submission rows carry the problem code in a cell attribute and only
  sometimes in a link, which is what made the first parser drop most rows.
"""
import pytest

from fetchers import codechef as CC
from models import contests as C
from models.unified_schema import (
    Platform,
    normalize_cc_difficulty,
    unify_tags,
    unify_tags_strict,
)


# --- difficulty normalisation ----------------------------------------------

def test_normalize_cc_difficulty_maps_into_unit_range():
    """Bounds are [200, 4000], set empirically from the observed catalogue."""
    assert normalize_cc_difficulty(200) == 0.0
    assert normalize_cc_difficulty(4000) == 1.0
    mid = normalize_cc_difficulty(2100)          # midpoint of the range
    assert 0.49 < mid < 0.51


def test_normalize_cc_difficulty_spans_the_observed_catalogue():
    """The real catalogue runs 19-4076; both ends must land in range."""
    assert normalize_cc_difficulty(19) == 0.0
    assert normalize_cc_difficulty(4076) == 1.0
    median = normalize_cc_difficulty(2071)       # catalogue median
    assert 0.45 < median < 0.55


def test_normalize_cc_difficulty_accepts_strings():
    """The list endpoint returns every numeric field as a string."""
    assert normalize_cc_difficulty("1994") == normalize_cc_difficulty(1994)


@pytest.mark.parametrize("sentinel", [-1, 0, 9999, "-1", "9999"])
def test_normalize_cc_difficulty_rejects_sentinels(sentinel):
    """Unrated problems are marked, not omitted — they must not become 0.0."""
    assert normalize_cc_difficulty(sentinel) is None


@pytest.mark.parametrize("bad", [None, "", "n/a", object()])
def test_normalize_cc_difficulty_rejects_junk(bad):
    assert normalize_cc_difficulty(bad) is None


def test_normalize_cc_difficulty_clamps_outliers():
    assert normalize_cc_difficulty(50) == 0.0
    assert normalize_cc_difficulty(8000) == 1.0


# --- tag unification --------------------------------------------------------

def test_strict_mapper_drops_setter_usernames():
    """CodeChef `user_tags` include the setter's handle; those are not topics."""
    tags = ["Mathematics", "nishank_adm", "u_admin_codechef_pw", "drupesh97"]
    assert unify_tags_strict(tags) == ["math"]


def test_strict_mapper_drops_generic_and_difficulty_tags():
    assert unify_tags_strict(["Algorithms", "cakewalk", "Special"]) == []


def test_strict_mapper_maps_codechef_vocabulary():
    out = unify_tags_strict([
        "Computational Geometry", "DP Approach", "Basic Programming Concepts",
        "Modular Arithmetic", "Graph Theory",
    ])
    assert out == sorted([
        "geometry", "dynamic_programming", "implementation",
        "number_theory", "graphs",
    ])


def test_loose_mapper_would_have_kept_usernames():
    """Contrast: why CodeChef needs the strict mapper at all."""
    assert "other:nishank_adm" in unify_tags(["nishank_adm"])


def test_strict_mapper_is_case_insensitive():
    assert unify_tags_strict(["MATHEMATICS", "  greedy  "]) == ["greedy", "math"]


# --- submission parsing -----------------------------------------------------

def _row(code, verdict, linked=False, time_="08:36 PM 10/06/26", lang="C++"):
    problem_cell = (
        f"<td title='{code}'><a href='/problems/{code}'>{code}</a></td>"
        if linked else f"<td title='{code}'>{code}</td>"
    )
    return (
        f"<tr><td title='{time_}'>{time_}</td>{problem_cell}"
        f"<td title=''><span title='{verdict}'></span></td>"
        f"<td title='{lang}'>{lang}</td><td title='View'>View</td></tr>"
    )


def test_parses_rows_without_a_problem_link():
    """
    Most rows carry the code only in the cell's title attribute; the link
    appears just for problems that reached the practice section. Reading the
    link alone silently dropped roughly nine rows in ten.
    """
    html = f"<table>{_row('EQMNG', 'accepted')}{_row('FINELE', 'wrong answer')}</table>"
    rows = CC._parse_submission_rows(html)
    assert [r["problem_code"] for r in rows] == ["EQMNG", "FINELE"]
    assert [r["accepted"] for r in rows] == [True, False]


def test_parses_linked_and_unlinked_rows_alike():
    html = f"<table>{_row('AAA', 'accepted', linked=True)}{_row('BBB', 'accepted')}</table>"
    assert [r["problem_code"] for r in CC._parse_submission_rows(html)] == ["AAA", "BBB"]


def test_skips_header_and_footer_rows():
    html = (
        "<table><tr><th>Time</th><th>Problem</th></tr>"
        + _row("AAA", "accepted")
        + "<tr><td colspan='3'>no more</td></tr></table>"
    )
    rows = CC._parse_submission_rows(html)
    assert len(rows) == 1


def test_partial_score_cell_does_not_mask_the_verdict():
    """An accepted partial-scoring row titles the cell '(100)', not the verdict."""
    html = (
        "<table><tr><td title='08:36 PM 10/06/26'>t</td><td title='EQMNG'>EQMNG</td>"
        "<td title='(100)'><span title='accepted'></span></td>"
        "<td title='C++'>C++</td><td title='View'>View</td></tr></table>"
    )
    rows = CC._parse_submission_rows(html)
    assert rows[0]["accepted"] is True
    assert rows[0]["verdict"] == "accepted"


def test_solved_problem_codes_dedupes_and_needs_acceptance():
    subs = [
        {"problem_code": "A", "accepted": False},
        {"problem_code": "A", "accepted": True},
        {"problem_code": "A", "accepted": True},
        {"problem_code": "B", "accepted": False},
    ]
    assert CC.solved_problem_codes(subs) == {"A"}


def test_parser_tolerates_empty_content():
    assert CC._parse_submission_rows("") == []


# --- statement extraction ---------------------------------------------------

def test_extract_statement_prefers_the_structured_component():
    detail = {
        "problemComponents": {"statement": "<p>Chef has <b>N</b> cakes.</p>"},
        "body": "<p>should not be used</p>",
    }
    assert CC.extract_statement(detail) == "Chef has N cakes."


def test_extract_statement_falls_back_to_body():
    assert CC.extract_statement({"body": "<p>fallback text</p>"}) == "fallback text"


def test_extract_statement_is_empty_when_there_is_nothing():
    assert CC.extract_statement({}) == ""
    assert CC.extract_statement({"problemComponents": {}, "body": ""}) == ""


def test_extract_statement_drops_inline_maths():
    detail = {"body": "<p>Given $N \\leq 10^5$ find the answer.</p>"}
    out = CC.extract_statement(detail)
    assert "10^5" not in out
    assert out.startswith("Given") and out.endswith("find the answer.")


def test_strip_markup_survives_malformed_attributes():
    """
    Regression: author-written statements can carry attributes that make lxml
    raise from inside BeautifulSoup rather than recover, which aborted a
    multi-hour harvest partway through. Extraction must degrade, not raise.
    """
    nasty = '<p xmlns:broken="x" {notanattr}=1>hello <b>world</b></p>'
    assert CC._strip_markup(nasty) == "hello world"


def test_strip_markup_last_resort_is_a_regex_strip(monkeypatch):
    """With every parser failing, text still comes back rather than an exception."""
    import fetchers.codechef as mod

    def explode(*a, **k):
        raise RuntimeError("parser unavailable")

    monkeypatch.setattr(mod, "BeautifulSoup", explode)
    assert mod._strip_markup("<p>plain <i>text</i></p>").split() == ["plain", "text"]


# --- contest history --------------------------------------------------------

def test_from_codechef_reconstructs_deltas():
    """CodeChef reports only the rating after each contest."""
    out = C.from_codechef([
        {"name": "Starters 1", "rating": 1500, "rank": 900,
         "end_date": "2026-01-10 22:00:00"},
        {"name": "Starters 2", "rating": 1580, "rank": 400,
         "end_date": "2026-02-10 22:00:00"},
    ])
    assert [r.new_rating for r in out] == [1500, 1580]
    assert out[0].delta is None          # nothing to subtract from
    assert out[1].delta == 80
    assert out[1].old_rating == 1500
    assert all(r.platform == "codechef" for r in out)


def test_from_codechef_orders_by_date_not_input_order():
    out = C.from_codechef([
        {"name": "Later", "rating": 1600, "end_date": "2026-02-10 22:00:00"},
        {"name": "Earlier", "rating": 1500, "end_date": "2026-01-10 22:00:00"},
    ])
    assert [r.name for r in out] == ["Earlier", "Later"]
    assert out[1].delta == 100


def test_from_codechef_survives_a_bad_date():
    out = C.from_codechef([{"name": "X", "rating": 1500, "end_date": "not a date"}])
    assert out[0].timestamp == 0


def test_codechef_calibration_uses_the_problem_rating_scale():
    """
    CodeChef rates users and problems on one scale, so a 2000-rated user
    calibrates to the same level a 2000-rated problem normalises to.
    """
    analysis = C.analyse(C.from_codechef([
        {"name": f"Starters {i}", "rating": 2000, "rank": 100,
         "end_date": f"2026-0{i}-10 22:00:00"}
        for i in range(1, 9)
    ]))
    cal = C.difficulty_calibration(analysis, "codechef")
    assert cal is not None
    assert cal["level"] == pytest.approx(normalize_cc_difficulty(2000), abs=0.02)


# --- schema -----------------------------------------------------------------

def test_codechef_is_a_known_platform():
    assert Platform.CODECHEF.value == "codechef"
