"""Smoke tests. These must pass with no API key and no network."""

from hdsim.core import DecisionTask, DomainConfig, Household, Member, parse_value
from hdsim.core import replay


def _config():
    return DomainConfig(
        name="test",
        task=DecisionTask(name="trip_count", domain="household travel", context="tomorrow",
                          target_description="the total number of trips", unit="trips"),
        fact_columns=["AGE", "WORKER"],
        translations={"AGE": lambda v: f"I am {v} years old.",
                      "WORKER": {"1": "I am employed.", "2": "I am not employed."}},
        banned_patterns=[r"\btrips?\b"],
    )


def test_facts_render_from_translations():
    facts = _config().facts_for({"AGE": 34, "WORKER": "1"})
    assert facts == ["I am 34 years old.", "I am employed."]


def test_unknown_column_is_skipped():
    assert _config().facts_for({"MISSING": 1}) == []


def test_leak_guard_catches_the_outcome():
    assert _config().leaks_outcome("I make four trips a day") is not None
    assert _config().leaks_outcome("I commute to work") is None


def test_parse_value_prefers_the_label():
    cfg = _config()
    assert parse_value("I think 3 or 4. FINAL_VALUE: 8", cfg) == 8
    assert parse_value("no label here, just 5", cfg) == 5
    assert parse_value("nothing numeric", cfg) is None


def test_parse_value_clamps_to_range():
    assert parse_value("FINAL_VALUE: 999", _config()) == 30


def test_roster_names_every_other_member():
    hh = Household("h1", [Member(1), Member(2), Member(3)])
    hh.build_roster(describe=lambda m: f"person {m.person_id}",
                    relate=lambda a, b: "housemate")
    assert hh.member(1).roster.count("housemate") == 2
    assert "person 1" not in hh.member(1).roster


def test_single_member_roster_says_lives_alone():
    hh = Household("h1", [Member(1)])
    hh.build_roster(describe=lambda m: "", relate=lambda a, b: "")
    assert "live alone" in hh.member(1).roster


def test_replay_runs_without_a_key():
    record = replay.get()
    assert record["consensus_value"] is not None
    text = replay.render(record)
    assert "Agreed:" in text
    assert len(replay.available()) >= 1


def test_transcript_parser_handles_markdown_speaker_labels():
    """Models emit speaker labels in several shapes. All must parse, with no stray emphasis."""
    from hdsim.core.negotiate import parse_transcript

    text = (
        "**Round 1**\n"
        "- **Member 1:** I think 3 trips.\n"
        "**Member 2:** ** I disagree, 4 trips.\n"
        "- Member 3: Agreed, 4 trips.\n"
        "Round 2\n"
        "**Wife**: Fine, 4 it is.\n"
        "FINAL_VALUE: 4\n"
    )
    turns = parse_transcript(text)
    assert [t["speaker"] for t in turns] == ["Member 1", "Member 2", "Member 3", "Wife"]
    assert [t["round"] for t in turns] == [1, 1, 1, 2]
    assert not any(t["text"].startswith("*") for t in turns), "emphasis leaked into turn text"
    assert not any(t["text"].endswith("*") for t in turns)
    assert turns[0]["text"] == "I think 3 trips."


def test_capsule_length_floor_scales_with_fact_count():
    """A short fact list must not be rejected for being short. Regression from the first live run."""
    from hdsim.core.persona import check_capsule

    cfg = _config()
    facts = ["I am 44 years old.", "I work full time."]
    good = "I am 44 years old. I work full time."
    assert check_capsule(good, facts, cfg) is None


def test_capsule_rejects_dropped_numeric_facts():
    from hdsim.core.persona import check_capsule

    cfg = _config()
    facts = ["I am 44 years old.", "My household owns 2 vehicles."]
    dropped = "I am 44 years old and I live with my family in a house near town."
    assert "dropped facts" in (check_capsule(dropped, facts, cfg) or "")
