"""Tests for the NER cleaning pipeline."""

from trocr_handwritten.ner.cleaning.export import (
    _is_full_iso,
    _fix_declaration_month_order,
)


def _make_rows(specs):
    """Build minimal row dicts from (act_id, declaration_date, birth_date) tuples."""
    rows = []
    for act_id, decl, birth in specs:
        commune, year_str = act_id.split("_")[:2]
        rows.append(
            {
                "act_id": act_id,
                "commune": commune,
                "year_registry": year_str,
                "declaration_date": decl,
                "birth_date": birth,
            }
        )
    return rows


def _months(rows, col="declaration_date"):
    return [int(r[col][5:7]) for r in rows if _is_full_iso(r.get(col))]


class TestDeclarationMonthOrder:
    """declaration_date months must be non-decreasing within commune × year, sorted by act_id."""

    def test_already_ordered(self):
        rows = _make_rows(
            [
                ("abymes_1841_p001_order1", "1841-01-10", "1841-01-09"),
                ("abymes_1841_p002_order1", "1841-03-05", "1841-03-04"),
                ("abymes_1841_p003_order1", "1841-07-20", "1841-07-19"),
            ]
        )
        result = _fix_declaration_month_order(rows)
        assert _months(result) == [1, 3, 7]

    def test_single_violation_corrected(self):
        rows = _make_rows(
            [
                ("abymes_1841_p001_order1", "1841-03-10", "1841-03-09"),
                (
                    "abymes_1841_p002_order1",
                    "1841-02-05",
                    "1841-02-04",
                ),  # wrong: Feb after Mar
                ("abymes_1841_p003_order1", "1841-04-20", "1841-04-19"),
                ("abymes_1841_p004_order1", "1841-04-25", "1841-04-24"),
                ("abymes_1841_p005_order1", "1841-05-01", "1841-04-30"),
            ]
        )
        result = _fix_declaration_month_order(rows)
        months = _months(result)
        for i in range(1, len(months)):
            assert (
                months[i] >= months[i - 1]
            ), f"Month ordering violated at position {i}: {months}"

    def test_multiple_communes_independent(self):
        rows = _make_rows(
            [
                ("abymes_1841_p001_order1", "1841-01-10", "1841-01-09"),
                ("abymes_1841_p002_order1", "1841-06-05", "1841-06-04"),
                ("gosier_1841_p001_order1", "1841-08-01", "1841-07-31"),
                (
                    "gosier_1841_p002_order1",
                    "1841-02-01",
                    "1841-01-31",
                ),  # wrong within gosier
                ("gosier_1841_p003_order1", "1841-09-10", "1841-09-09"),
            ]
        )
        result = _fix_declaration_month_order(rows)
        abymes = sorted(
            [r for r in result if r["commune"] == "abymes"], key=lambda r: r["act_id"]
        )
        gosier = sorted(
            [r for r in result if r["commune"] == "gosier"], key=lambda r: r["act_id"]
        )
        for group in (abymes, gosier):
            months = _months(group)
            for i in range(1, len(months)):
                assert months[i] >= months[i - 1]

    def test_missing_dates_skipped(self):
        rows = _make_rows(
            [
                ("abymes_1841_p001_order1", "1841-01-10", "1841-01-09"),
                ("abymes_1841_p002_order1", None, None),
                ("abymes_1841_p003_order1", "1841-06-20", "1841-06-19"),
            ]
        )
        result = _fix_declaration_month_order(rows)
        months = _months(result)
        assert months == [1, 6]

    def test_neighbor_vote_picks_ascending_month(self):
        rows = _make_rows(
            [
                ("abymes_1841_p001_order1", "1841-05-01", "1841-05-01"),
                ("abymes_1841_p002_order1", "1841-05-10", "1841-05-10"),
                (
                    "abymes_1841_p003_order1",
                    "1841-02-15",
                    "1841-02-14",
                ),  # wrong: Feb after May
                ("abymes_1841_p004_order1", "1841-06-01", "1841-06-01"),
                ("abymes_1841_p005_order1", "1841-06-10", "1841-06-10"),
            ]
        )
        result = _fix_declaration_month_order(rows)
        months = _months(result)
        for i in range(1, len(months)):
            assert months[i] >= months[i - 1]
        # corrected month must be >= 5 (prev_month)
        assert months[2] >= 5
