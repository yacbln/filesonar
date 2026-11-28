import os
import time

import pytest
from conftest import write
from samples import docx, photo, text_pdf

from filesonar.search import MAX_LIMIT, Searcher, fts_query


@pytest.fixture
def corpus(home, indexer):
    text_pdf(
        home / "insurance/auto-policy.pdf",
        ["Car insurance policy. Vehicle: 2019 Toyota Corolla. Premium 640 EUR per year."],
    )
    docx(
        home / "housing/lease.docx",
        [
            "Residential lease agreement for the apartment.",
            "Monthly rent is 950 and the deposit is two months.",
        ],
    )
    write(home / "recipes/couscous.md", "Couscous with lamb and seven vegetables, serves six.")
    write(home / "work/notes.txt", "Meeting notes: migrate the billing service to Postgres.")
    photo(home / "photos/IMG_0042.jpg")  # captioned as a red bicycle by the fake captioner
    indexer.run()
    return home


@pytest.fixture
def searcher(db, embedder):
    return Searcher(db, embedder)


def top(hits):
    return os.path.basename(hits[0].path)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("toyota insurance", "auto-policy.pdf"),
        ("apartment rent deposit", "lease.docx"),
        ("lamb couscous recipe", "couscous.md"),
        ("billing postgres migration", "notes.txt"),
        ("red bicycle", "IMG_0042.jpg"),
    ],
)
def test_finds_files_by_content(corpus, searcher, query, expected):
    hits = searcher.search(query)
    assert top(hits) == expected
    assert hits[0].score == 1.0
    assert all(0 < h.score <= 1 for h in hits)


def test_result_shape(corpus, searcher):
    hit = searcher.search("toyota")[0]
    data = hit.to_dict()
    assert set(data) == {
        "path",
        "name",
        "kind",
        "score",
        "modified",
        "size_bytes",
        "page",
        "matched_in",
        "snippet",
        "other_copies",
        "source",
        "details",
    }
    assert hit.kind == "pdf" and hit.page == 1 and hit.matched_in == "text"
    assert "Toyota" in hit.snippet
    assert len(hit.snippet) <= 300


def test_image_match_reports_caption(corpus, searcher):
    hit = searcher.search("bicycle lake", kind="image")[0]
    assert hit.matched_in == "caption" and hit.kind == "image"


def test_kind_and_folder_filters(corpus, searcher):
    assert {h.kind for h in searcher.search("lease rent car", kind="pdf")} == {"pdf"}
    hits = searcher.search("notes rent car lamb", folder=str(corpus / "recipes"))
    assert [top(hits)] == ["couscous.md"] and len(hits) == 1
    # a folder that merely shares a prefix is not included
    assert searcher.search("lamb", folder=str(corpus / "recip")) == []


def test_date_filters(corpus, searcher, indexer):
    old = corpus / "recipes/couscous.md"
    os.utime(old, (time.time(), time.mktime((2020, 3, 1, 12, 0, 0, 0, 0, -1))))
    indexer.run()
    assert searcher.search("couscous", modified_before="2020-03-01")
    assert not searcher.search("couscous", modified_after="2021-01-01")
    assert searcher.search("couscous", modified_after="2020-02-28", modified_before="2020-03-02")


def test_one_result_per_file(corpus, searcher):
    hits = searcher.search("the and of for")
    assert len({h.path for h in hits}) == len(hits)


def test_limit_is_clamped(corpus, searcher):
    assert len(searcher.search("a the and", limit=1)) == 1
    assert len(searcher.search("a the and", limit=10_000)) <= MAX_LIMIT
    assert len(searcher.search("a the and", limit=-5)) == 1


@pytest.mark.parametrize(
    "query",
    [
        '"',
        'toyota"; DROP TABLE files; --',
        "NEAR(toyota insurance)",
        "name:lease",
        "AND OR NOT",
        "*",
        "toyota*",
        "{text}: x",
        "ünïcödé ✓ 日本",
        "'); --",
    ],
)
def test_hostile_queries_never_error(corpus, searcher, db, query):
    searcher.search(query)  # must not raise
    assert db.stats()["files"] == 5


def test_fts_query_quotes_everything():
    assert (
        fts_query('NEAR(car home) "x" name:lease')
        == '"near" OR "car" OR "home" OR "name" OR "lease"'
    )
    assert fts_query("   ") is None
    assert fts_query("x") is None
    assert fts_query("7") == '"7"'


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"query": ""}, "empty"),
        ({"query": "x" * 1001}, "too long"),
        ({"query": "a", "kind": "hologram"}, "kind"),
        ({"query": "a", "modified_after": "yesterday"}, "date"),
    ],
)
def test_bad_arguments(searcher, kwargs, message):
    with pytest.raises(ValueError, match=message):
        searcher.search(**kwargs)


def test_empty_index(searcher):
    assert searcher.search("anything") == []


def test_falls_back_to_keywords_when_model_is_down(corpus, searcher, embedder):
    embedder.fail = True
    hits = searcher.search("toyota")
    assert top(hits) == "auto-policy.pdf"
    assert "keyword only" in searcher.last_warning
    embedder.fail = False
    searcher.search("toyota")
    assert searcher.last_warning is None


def test_unrelated_query_returns_nothing(corpus, searcher):
    assert searcher.search("quantum chromodynamics") == []


def test_filters_do_not_surface_unrelated_files(corpus, searcher):
    assert searcher.search("couscous", kind="image") == []


def test_embedding_model_mismatch_falls_back_to_keywords(corpus, db, embedder):
    from conftest import FakeEmbedder

    other = FakeEmbedder()
    other.name = "another-model"
    s = Searcher(db, other)
    assert top(s.search("toyota")) == "auto-policy.pdf"
    assert "rebuild" in s.last_warning


def test_search_by_file_name(corpus, searcher):
    assert top(searcher.search("IMG_0042")) == "IMG_0042.jpg"


def test_exact_file_name_ranks_first(home, indexer, searcher):
    for stamp in ("09-03-33", "09-06-54", "10-11-12", "21-45-00"):
        write(home / f"shots/Screenshot from 2025-06-21 {stamp}.txt", "a screenshot of a sticker")
    indexer.run()
    hits = searcher.search("Screenshot from 2025-06-21 09-03-33")
    assert hits[0].name == "Screenshot from 2025-06-21 09-03-33.txt"
    assert searcher.search("21-45-00")[0].name == "Screenshot from 2025-06-21 21-45-00.txt"


def test_identical_copies_are_one_result(home, indexer, searcher):
    for where in ("Downloads/book.txt", "Downloads/book (1).txt", "Documents/book-copy.txt"):
        write(home / where, "Solutions architect study guide: VPC, subnets and route tables.")
    write(home / "other.txt", "A guide to route planning for cyclists.")
    indexer.run()
    hits = searcher.search("study guide subnets")
    assert len([h for h in hits if "book" in h.name]) == 1
    assert len(hits[0].other_copies) == 2
    assert hits[0].path not in hits[0].other_copies


def test_clear_keyword_winner_beats_middling_matches():
    """Measured on a real index: with the textbook constant (60), a file ranked first by
    keywords lost to files that were only around 10th in both keyword and meaning
    results. The constant in use must let the clear winner through."""
    from filesonar.search import fuse

    middling = list(range(100, 110))
    keyword = [1, *middling]  # 1 is the clear keyword winner
    meaning = [200, *middling]  # 1 is not close in meaning
    scores = fuse([(keyword, 1.0), (meaning, 1.0)])
    # Strong in both lists still wins (two signals beat one) ...
    assert scores[100] > scores[1]
    # ... but files only 6th-10th in both no longer beat the clear keyword winner.
    assert all(scores[1] > scores[m] for m in middling[5:])
    old = fuse([(keyword, 1.0), (meaning, 1.0)], k=60)
    assert all(old[1] < old[m] for m in middling[5:])  # the behaviour this guards against


@pytest.mark.parametrize(
    ("query", "kinds"),
    [
        ("photos of our dog", {"image"}),
        ("screenshot of the rent payment", {"image"}),
        ("video from the concert", {"video"}),
        ("voice memo with the mechanic's advice", {"audio"}),
        ("recording of rain", {"audio"}),
        ("car insurance policy", set()),
        ("the voice of reason", set()),
    ],
)
def test_wanted_kinds(query, kinds):
    from filesonar.search import wanted_kinds

    assert wanted_kinds(query) == kinds


@pytest.mark.parametrize(
    ("query", "date"),
    [
        ("bank statement for May 2023", ("2023", 5)),
        ("electric bill for Dec 2024", ("2024", 12)),
        ("2021 tax return", ("2021", None)),
        ("taxes 2021 vs 2023", None),  # two years: no single date wanted
        ("bank statement for May", None),  # no year
        ("order number 512-8830114", None),
    ],
)
def test_wanted_date(query, date):
    from filesonar.search import wanted_date

    assert wanted_date(query) == date


@pytest.mark.parametrize(
    ("query", "codes"),
    [
        ("booking reference K7Q2ZP", ["k7q2zp"]),
        ("order number 512-8830114", ["512-8830114"]),
        ("policy NW-58213-04", ["nw-58213-04"]),
        ("tax return 2023", []),
        ("my W-2 form", []),
        ("the H2O bill", []),
    ],
)
def test_wanted_codes(query, codes):
    from filesonar.search import wanted_codes

    assert wanted_codes(query) == codes


def test_named_kind_and_date_break_ties(home, indexer, searcher, monkeypatch):
    """With keyword and meaning scores equal, the kind of file and the date the query
    names decide: the photo beats a note about the same thing, and the statement for
    the month asked about beats the one a year later."""
    import filesonar.search as fsearch

    write(home / "Notes" / "bike.txt", "sell the red bicycle in the garage")
    photo(home / "Pictures" / "IMG_0042.jpg")  # the fake captioner sees a red bicycle
    for year in (2023, 2024):
        text_pdf(home / "Bank" / f"statement-{year}.pdf",
                 [f"Checking statement. Statement period: May 1 - May 31, {year}"])  # fmt: skip
    indexer.run()
    real = fsearch.fuse
    monkeypatch.setattr(fsearch, "fuse", lambda lists, k=None: dict.fromkeys(real(lists, k), 1.0))
    assert searcher.search("photo of the red bicycle")[0].kind == "image"
    assert searcher.search("picture of the bicycle in the garage")[0].kind == "image"
    assert searcher.search("checking statement May 2023")[0].name == "statement-2023.pdf"
    assert searcher.search("checking statement May 2024")[0].name == "statement-2024.pdf"
    write(home / "Travel" / "a.txt", "Flight booking. Booking reference: K7Q2ZP")
    write(home / "Travel" / "b.txt", "Flight booking. Booking reference: R58VJ5")
    indexer.run()
    assert searcher.search("booking reference R58VJ5")[0].name == "b.txt"


def test_confidence_reflects_how_well_the_best_result_matches(corpus, searcher, monkeypatch):
    # With five files any match stands out; this checks the shared-words part alone.
    monkeypatch.setattr("filesonar.search._STANDOUT_GAP", 9.0)
    searcher.search("car insurance policy")
    assert searcher.last_confidence == "high"
    # Only the common word matches; the word that says what is wanted is in no file.
    searcher.search("giraffe policy")
    assert searcher.last_confidence == "low"
    assert searcher.search("car insurance", kind="video") == []  # no videos here
    assert searcher.last_confidence is None  # nothing returned, nothing to judge


@pytest.mark.parametrize(
    ("query", "terms"),
    [
        ("photos of our dog", ["dog"]),
        ("where's my car insurance policy", ["car", "insurance", "policy"]),
        ("voice memo about the mechanic", ["mechanic"]),
        ("the W-2 from 2023", ["2023"]),
    ],
)
def test_query_terms(query, terms):
    from filesonar.search import query_terms

    assert query_terms(query) == terms


def test_each_embedding_model_has_its_own_cutoffs():
    from filesonar.search import calibration

    assert calibration("embeddinggemma", None) == (0.60, 0.55)
    assert calibration("embeddinggemma:latest", None) == (0.60, 0.55)
    assert calibration("nomic-embed-text", None) == (0.44, 0.35)
    # 0.44 in an older config was nomic's default, not a choice for another model.
    assert calibration("embeddinggemma", 0.44)[0] == 0.60
    assert calibration("nomic-embed-text", 0.5)[0] == 0.5  # a real choice is kept
    assert calibration("embeddinggemma", 0.5)[0] == 0.5
    assert calibration("some-new-model", None) == (0.44, 0.35)


def test_a_match_far_closer_than_every_other_file_is_confident(corpus, searcher, monkeypatch):
    """No shared words and not close by the usual measure, but nothing else is anywhere
    near: "renting a flat" for the only lease is a confident answer."""
    import filesonar.search as fsearch

    monkeypatch.setattr(fsearch, "query_terms", lambda q: ["nomatch"])  # no shared words
    searcher._confident_distance = 0.0  # every match counts as "not close"
    searcher.search("lease agreement apartment rent")
    assert searcher.last_signals[0] == 0.0
    assert searcher.last_confidence == "high"  # it stands out from every other file
    monkeypatch.setattr(fsearch, "_STANDOUT_GAP", 9.0)  # as if other files were as close
    searcher.search("lease agreement apartment rent")
    assert searcher.last_confidence == "low"
