"""The Red Alert reflexes (examples/llm/redalert_autopilot.py) and how
RedAlertPlayer merges them with the model's orders."""

from __future__ import annotations

import threading

from altruagent import WAIT, DecisionContext, GameState, WithReasoning
from examples.llm import redalert
from examples.llm.redalert_autopilot import Autopilot, drop_model_duplicates

CONTEXT = DecisionContext(session_id="r", tournament_id=None, game_type="red_alert", agent_id="a", seat_position=0)


def _raw(*, buildings=(), units=(), production=(), available=(), legal=None, power=(100, 0), events=(), tick=1000):
    return {
        "phase": "playing",
        "pacing": {"mode": "realtime"},
        "observation": {
            "tick": tick,
            "economy": {"cash": 3000, "power_provided": power[0], "power_drained": power[1], "harvesters": 0},
            "buildings": [{"id": 10 + i, "type": t, "x": 5, "y": 5} for i, t in enumerate(buildings)],
            "units": list(units),
            "production": list(production),
            "available_production": list(available),
            "events": list(events),
            "base_center": [5, 5],
        },
        "legal_actions": legal or {},
    }


def test_deploys_the_mcv_when_there_is_no_construction_yard():
    raw = _raw(units=[{"id": 7, "type": "mcv"}], legal={"deploy": [7]})

    assert Autopilot().orders(raw, 0.0) == [{"cmd": "deploy", "units": [7]}]


def test_places_every_finished_building():
    raw = _raw(buildings=["fact"], legal={"place": [{"item": "powr"}, "tent"]})

    orders = Autopilot().orders(raw, 0.0)

    assert {"cmd": "place", "item": "powr"} in orders and {"cmd": "place", "item": "tent"} in orders


def test_follows_the_economic_core_in_order():
    pilot = Autopilot()
    available = ["powr", "tent", "proc", "weap"]

    assert {"cmd": "build", "item": "powr"} in pilot.orders(_raw(buildings=["fact"], available=available), 0.0)
    assert {"cmd": "build", "item": "tent"} in pilot.orders(_raw(buildings=["fact", "powr"], available=available), 0.0)
    assert {"cmd": "build", "item": "proc"} in pilot.orders(_raw(buildings=["fact", "powr", "tent"], available=available), 0.0)
    assert {"cmd": "build", "item": "proc"} in pilot.orders(
        _raw(buildings=["fact", "powr", "tent", "proc"], available=available), 10.0
    )
    assert {"cmd": "build", "item": "weap"} in pilot.orders(
        _raw(buildings=["fact", "powr", "tent", "proc", "proc"], available=available), 20.0
    )


def test_builds_power_when_spare_power_runs_low_and_waits_for_a_busy_queue():
    available = ["powr", "proc"]
    low = _raw(buildings=["fact", "powr", "barr"], available=available, power=(100, 80))
    busy = _raw(buildings=["fact", "powr", "barr"], available=available, power=(100, 80),
                production=[{"queue": "Building", "item": "proc", "ready": False}])

    assert {"cmd": "build", "item": "powr"} in Autopilot().orders(low, 0.0)
    assert not any(o["cmd"] == "build" for o in Autopilot().orders(busy, 0.0))


def test_trains_harvesters_up_to_the_target():
    two = [{"id": 1, "type": "harv"}, {"id": 2, "type": "harv"}]
    three = two + [{"id": 3, "type": "harv"}]
    base = ["fact", "powr", "barr", "proc", "weap"]

    assert {"cmd": "train", "item": "harv", "count": 1} in Autopilot().orders(_raw(buildings=base, units=two, available=["harv"]), 0.0)
    assert Autopilot().orders(_raw(buildings=base, units=three, available=["harv"]), 0.0) == []


def test_sends_idle_combat_units_to_a_recent_attack_only():
    recent = _raw(buildings=["fact"], legal={"combat_units": [21, 22]}, tick=1000,
                  events=[{"kind": "under_attack", "tick": 990, "actor": 10}])
    old = _raw(buildings=["fact"], legal={"combat_units": [21, 22]}, tick=1000,
               events=[{"kind": "under_attack", "tick": 100, "actor": 10}])

    assert Autopilot().orders(recent, 0.0) == [{"cmd": "attack_move", "units": [21, 22], "to": [5, 5]}]
    assert Autopilot().orders(old, 0.0) == []


def test_does_not_resend_the_same_order_within_a_few_seconds():
    pilot = Autopilot()
    raw = _raw(buildings=["fact"], legal={"place": ["powr"]})

    assert pilot.orders(raw, 0.0) and pilot.orders(raw, 1.0) == []
    assert pilot.orders(raw, 10.0) == [{"cmd": "place", "item": "powr"}]


def test_model_duplicates_of_autopilot_jobs_are_dropped():
    raw = _raw(production=[{"item": "powr"}])
    model = [{"cmd": "place", "item": "tent"}, {"cmd": "build", "item": "powr"}, {"cmd": "train", "item": "e1", "count": 5}]

    assert drop_model_duplicates(model, raw) == [{"cmd": "train", "item": "e1", "count": 5}]


class SlowProvider:
    """Answers only when released, like a model that takes seconds."""

    model = "fake"

    def __init__(self, answer):
        self.answer = answer
        self.release = threading.Event()

    def complete_structured(self, messages, schema_name, schema):
        self.release.wait(5)
        return self.answer


def test_background_player_sends_reflexes_while_the_model_is_thinking():
    answer = {"orders": [{"cmd": "train", "item": "e1", "count": 3, "units": None, "to": None}], "reasoning_summary": "Rifles."}
    provider = SlowProvider(answer)
    player = redalert.RedAlertPlayer(provider, log=lambda line: None, autopilot=True, background=True)
    raw = _raw(buildings=["fact"], legal={"place": ["powr"]})

    first = player.choose_action(GameState.from_mcp_state(raw), CONTEXT)
    assert isinstance(first, WithReasoning) and first.action["orders"] == [{"cmd": "place", "item": "powr"}]

    provider.release.set()
    player._future.result(5)
    second = player.choose_action(GameState.from_mcp_state(_raw(buildings=["fact"])), CONTEXT)
    assert second.action["orders"] == [{"cmd": "train", "item": "e1", "count": 3}]
    assert second.reasoning_summary == "Rifles."


def test_player_without_autopilot_behaves_as_before():
    provider = SlowProvider({"orders": [], "reasoning_summary": "Nothing."})
    provider.release.set()
    player = redalert.RedAlertPlayer(provider, log=lambda line: None)

    assert player.choose_action(GameState.from_mcp_state(_raw(legal={"place": ["powr"]})), CONTEXT) is WAIT


def test_a_unit_attacked_far_from_home_does_not_pull_the_army():
    scout = {"id": 30, "type": "e1", "x": 80, "y": 80}
    raw = _raw(buildings=["fact"], units=[scout], legal={"combat_units": [21, 22]}, tick=1000,
               events=[{"kind": "under_attack", "tick": 995, "actor": 30}])

    assert Autopilot().orders(raw, 0.0) == []


def test_model_spending_is_gated_until_the_economy_is_up():
    from examples.llm.redalert_autopilot import gate_model_spending

    early = _raw(buildings=["fact", "powr", "tent", "proc"], units=[{"id": 1, "type": "e1"}])
    model = [{"cmd": "build", "item": "pbox"}, {"cmd": "train", "item": "e1", "count": 10},
             {"cmd": "train", "item": "1tnk", "count": 2}, {"cmd": "attack_move", "units": [1], "to": [9, 9]}]

    assert gate_model_spending(model, early) == [
        {"cmd": "train", "item": "e1", "count": 2}, {"cmd": "attack_move", "units": [1], "to": [9, 9]}
    ]

    ready = _raw(buildings=["fact", "powr", "tent", "proc", "proc", "weap"],
                 units=[{"id": i, "type": "harv"} for i in range(3)])
    assert gate_model_spending(model, ready) == model


def test_a_harvester_attacked_far_from_home_is_defended():
    harvester = {"id": 40, "type": "harv", "x": 80, "y": 80}
    raw = _raw(buildings=["fact"], units=[harvester], legal={"combat_units": [21]}, tick=1000,
               events=[{"kind": "under_attack", "tick": 995, "actor": 40}])

    assert Autopilot().orders(raw, 0.0) == [{"cmd": "attack_move", "units": [21], "to": [80, 80]}]


def test_queued_infantry_counts_toward_the_early_cap():
    from examples.llm.redalert_autopilot import gate_model_spending

    raw = _raw(buildings=["fact", "powr", "tent", "proc"], production=[{"item": "e1"}, {"item": "e1"}, {"item": "e1"}])

    assert gate_model_spending([{"cmd": "train", "item": "e1", "count": 3}], raw) == []


def test_light_tanks_are_upgraded_once_medium_tanks_can_be_built():
    from examples.llm.redalert_autopilot import upgrade_tanks

    order = [{"cmd": "train", "item": "1tnk", "count": 2}]

    assert upgrade_tanks(order, _raw(available=["1tnk"])) == order
    assert upgrade_tanks(order, _raw(available=["1tnk", "2tnk"])) == [{"cmd": "train", "item": "2tnk", "count": 2}]


def test_status_line_shows_ore_and_the_enemy_army():
    raw = _raw()
    raw["observation"]["economy"]["ore"] = 450
    raw["observation"]["visible_enemies"] = [{"type": "3tnk"}, {"type": "3tnk"}, {"type": "e3"}, {"type": "x", "wreck": True}]

    line = redalert.status_line(raw)

    assert "cash 3000+ore 450" in line and "(3tnkx2 e3x1)" in line
