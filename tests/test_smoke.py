"""Smoke tests. These must pass with no API key and no network."""

import pytest

from hdsim.core import (
    DecisionTask,
    DomainConfig,
    Household,
    Member,
    cli,
    evaluate,
    parse_value,
    replay,
)
from hdsim.core.evaluate import paired_bootstrap
from hdsim.core.negotiate import _read_turn


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


def test_stage2_parses_the_consensus_format():
    """The stage-two completion format, as the ported parser reads it."""
    from hdsim.core import stage2

    completion = (
        "[Member 1] ACKNOWLEDGE: roster MY_VALUE: 4 PREFERRED_TOTAL: 6\n"
        "[Member 2] ACKNOWLEDGE: Member 1 MY_VALUE: 2 PREFERRED_TOTAL: 6\n"
        "FINAL_CONSENSUS: 6\n"
    )
    result = stage2.parse_completion_to_result(completion, n_members=2)
    assert result["final_value"] == 6
    assert result["n_rounds"] == 1
    assert len(result["transcript"]) == 3


def test_stage2_final_label_is_the_household_sum_not_an_average():
    """The distinction the three labels exist to preserve."""
    from hdsim.core import stage2

    assert "sum" in stage2.CONSENSUS_SYSTEM_TEMPLATE.lower()
    assert stage2.MEMBER_LABEL != stage2.TOTAL_LABEL != stage2.FINAL_LABEL


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


def test_number_check_ignores_how_a_number_is_written():
    """A faithful rendering is not a dropped fact.

    Regression: a real NHTS persona was rejected because the model wrote "17,000" and the check
    read that as the two numbers 17 and 000, then reported the original 17000 as missing.
    """
    from hdsim.core.persona import check_capsule

    cfg = _config()
    facts = ["My area's density is 17000.", "My household owns 2 vehicles."]
    assert check_capsule("My area's density is 17,000. My household owns two vehicles.",
                         facts, cfg) is None
    # a genuinely dropped number is still caught
    assert "dropped facts" in (check_capsule("My area is dense and I own vehicles.",
                                             facts, cfg) or "")


def test_bool_proposal_parses_an_integer_answer():
    """A yes or no domain gets "FINAL_VALUE: 1" back, not the word "yes"."""
    task = DecisionTask(name="moved", domain="mobility", context="two years",
                        target_description="whether the household moves", unit="a yes or no",
                        final_label="FINAL_VALUE", value_type=bool, value_range=(0, 1))
    config = DomainConfig(name="m", task=task)
    assert parse_value("reasons\n\nFINAL_VALUE: 1", config) is True
    assert parse_value("reasons\n\nFINAL_VALUE: 0", config) is False
    assert parse_value("reasons\n\nFINAL_VALUE: yes", config) is True
    assert parse_value("no label anywhere", config) is None


def test_turn_keeps_dialogue_and_drops_the_reward_fields():
    names = {1: "Head", 2: "Spouse"}
    turn = _read_turn(
        "[Member 1] ACKNOWLEDGE: I drive to work and back. MY_VALUE: 2 PREFERRED_TOTAL: 6", names)
    assert turn["speaker"] == "Head"
    assert turn["text"] == "I drive to work and back."
    # a bare cross-reference is not dialogue and must not become a turn
    assert _read_turn("[Member 2] ACKNOWLEDGE: [Member 1] MY_VALUE: 4 PREFERRED_TOTAL: 4",
                      names)["text"] == ""
    assert _read_turn("FINAL_CONSENSUS: 6", names) is None


def test_a_yes_or_no_result_reads_as_yes_or_no():
    record = {"household_id": "x", "unit": "a yes or no", "consensus_value": False,
              "ground_truth": None,
              "members": [{"person_id": "1", "role": "Head", "proposal_value": True,
                           "capsule": ""}],
              "transcript": [{"round": 1, "speaker": "Head", "text": "I would rather stay."}]}
    out = replay.render(record)
    assert "Agreed: no" in out
    assert "a yes or no" not in out
    assert "Head                   yes" in out


# --- serialisation: a saved run must come back the way it went in -----------------------------

def test_household_survives_a_save_and_load():
    """Every field that render() reads has to survive the round trip, not just most of them."""
    h = Household(household_id="x", members=[Member(person_id=1, label="Head")],
                  consensus_value=4, rounds=2, unit="trips",
                  transcript=[{"round": 1, "speaker": "Head", "text": "hi"}])
    back = Household.from_dict(h.to_dict())
    for field in ("household_id", "consensus_value", "rounds", "unit", "transcript"):
        assert getattr(back, field) == getattr(h, field), field
    assert back.members[0].label == "Head"


def test_a_recorded_run_replays():
    h = Household(household_id="x", members=[Member(person_id=1, label="Head")],
                  consensus_value=6, unit="trips",
                  transcript=[{"round": 1, "speaker": "Head", "text": "Two each way."}])
    h.members[0].proposal_value = 6
    out = replay.render(h.to_record())
    assert "Head" in out and "Agreed: 6 trips" in out and "Two each way." in out


# --- metrics: a yes or no decision must not be scored as if it were a count -------------------

def test_counts_are_scored_as_counts():
    hs = [Household(household_id=str(i), consensus_value=v, ground_truth=v + 1)
          for i, v in enumerate([2, 4, 6, 8])]
    s = evaluate.score(hs)
    assert s.n == 4 and s.mae == 1.0 and s.bias == -1.0


def test_a_yes_or_no_set_is_refused_by_the_count_metrics():
    """within_1 is 1.0 for every possible 0/1 prediction, so MAE-style scoring cannot be offered."""
    hs = [Household(household_id=str(i), consensus_value=p, ground_truth=t)
          for i, (p, t) in enumerate([(True, True), (False, False), (False, True), (True, False)])]
    assert evaluate.is_binary(hs)
    with pytest.raises(TypeError, match="score_binary"):
        evaluate.score(hs)


def test_a_yes_or_no_set_scores_as_a_classifier():
    hs = [Household(household_id=str(i), consensus_value=p, ground_truth=t)
          for i, (p, t) in enumerate([(True, True), (False, False), (False, True), (True, False)])]
    s = evaluate.score_binary(hs)
    assert (s.n, s.accuracy, s.precision, s.recall, s.f1) == (4, 0.5, 0.5, 0.5, 0.5)
    perfect = [Household(household_id=str(i), consensus_value=t, ground_truth=t)
               for i, t in enumerate([True, False, True, False])]
    assert evaluate.score_binary(perfect).f1 == 1.0


def test_bootstrap_picks_the_right_metric_for_the_domain():
    yes = [Household(household_id=str(i), consensus_value=t, ground_truth=t)
           for i, t in enumerate([True, False, True, False])]
    assert paired_bootstrap(yes, yes, n_boot=50)["metric"] == "f1"
    counts = [Household(household_id=str(i), consensus_value=v, ground_truth=v)
              for i, v in enumerate([2, 4])]
    assert paired_bootstrap(counts, counts, n_boot=50)["metric"] == "mae"


# --- the CLI is a user-facing surface too -----------------------------------------------------

def test_an_unknown_recording_is_reported_without_repr_quotes(capsys):
    assert cli.main(["demo", "nope"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("No recording"), err


def test_asking_for_personas_says_so_when_there_are_none():
    record = {"household_id": "x", "unit": "trips", "consensus_value": 4, "ground_truth": None,
              "members": [{"person_id": "1", "role": "Head", "proposal_value": 4, "capsule": ""}],
              "transcript": [{"round": 1, "speaker": "Head", "text": "hi"}]}
    assert "does not carry personas" in replay.render(record, show_personas=True)
