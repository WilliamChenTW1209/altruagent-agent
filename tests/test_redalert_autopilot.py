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
                 units=[{"id": i, "type": "harv"} for i in range(3)] + [{"id": 50 + i, "type": "e1"} for i in range(12)])
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


READY = ["fact", "powr", "tent", "proc", "proc", "weap"]
HARVESTERS = [{"id": 100 + i, "type": "harv"} for i in range(3)]


def test_no_counter_before_the_economy_is_up_or_against_a_few_enemies():
    early = _raw(buildings=["fact", "powr", "tent"], available=["e1", "e3"])
    early["observation"]["visible_enemies"] = [{"id": i, "type": "3tnk"} for i in range(5)]
    few = _raw(buildings=READY, units=HARVESTERS, available=["e1", "e3"])
    few["observation"]["visible_enemies"] = [{"id": 1, "type": "3tnk"}]

    assert not any(o["cmd"] == "train" and o["item"] == "e3" for o in Autopilot().orders(early, 0.0))
    assert {"cmd": "train", "item": "e3", "count": 5} not in Autopilot().orders(few, 0.0)  # no counter batch


def test_a_building_already_queued_or_built_once_is_not_queued_again():
    raw = _raw(buildings=["fact", "fix"], production=[{"item": "pbox"}])
    model = [{"cmd": "build", "item": "pbox"}, {"cmd": "build", "item": "fix"},
             {"cmd": "build", "item": "gun"}, {"cmd": "build", "item": "gun"}]

    assert drop_model_duplicates(model, raw) == [{"cmd": "build", "item": "gun"}]


def test_idle_infantry_are_assigned_to_guard_harvesters_and_kept_from_the_model():
    pilot = Autopilot()
    units = HARVESTERS + [{"id": 300 + i, "type": "e1"} for i in range(8)]
    raw = _raw(buildings=READY, units=units, legal={"combat_units": [300 + i for i in range(8)]})

    guards = [o for o in pilot.orders(raw, 0.0) if o["cmd"] == "guard"]

    assert len(guards) == 6 and {o["target"] for o in guards} == {100, 101, 102}
    attack = [{"cmd": "attack_move", "units": [300, 301, 302, 303, 304, 305, 306, 307], "to": [9, 9]}]
    kept = pilot.strip_guards(attack)[0]["units"]
    assert len(kept) == 2 and not set(kept) & pilot.guards


def test_a_defense_is_built_per_refinery_and_placed_beside_it():
    raw = _raw(buildings=READY, units=HARVESTERS, available=["ftur"])
    raw["observation"]["buildings"][3].update(x=40, y=20)  # first refinery

    assert {"cmd": "build", "item": "ftur"} in Autopilot().orders(raw, 0.0)

    placing = _raw(buildings=READY, units=HARVESTERS, legal={"place": ["ftur"]})
    placing["observation"]["buildings"][3].update(x=40, y=20)
    placing["observation"]["buildings"][4].update(x=60, y=60)
    assert {"cmd": "place", "item": "ftur", "at": [43, 21]} in Autopilot().orders(placing, 0.0)


def test_a_queued_tank_is_cancelled_when_a_core_building_has_no_money():
    stuck = _raw(buildings=["fact", "powr", "tent", "proc", "weap"],
                 production=[{"queue": "Building", "item": "proc", "ready": False}, {"queue": "Vehicle", "item": "2tnk"}])
    stuck["observation"]["economy"]["cash"] = 0
    funded = _raw(buildings=["fact", "powr", "tent", "proc", "weap"],
                  production=[{"queue": "Building", "item": "proc", "ready": False}, {"queue": "Vehicle", "item": "2tnk"}])

    assert {"cmd": "cancel", "item": "2tnk"} in Autopilot().orders(stuck, 0.0)
    assert not any(o["cmd"] == "cancel" for o in Autopilot().orders(funded, 0.0))


def test_a_queued_tank_is_cancelled_for_a_starved_harvester_too():
    raw = _raw(buildings=["fact", "powr", "tent", "proc", "weap"],
               production=[{"queue": "Vehicle", "item": "2tnk"}, {"queue": "Vehicle", "item": "harv", "ready": False}])
    raw["observation"]["economy"]["cash"] = 0

    assert {"cmd": "cancel", "item": "2tnk"} in Autopilot().orders(raw, 0.0)


def test_small_attacks_gather_at_the_staging_point_and_big_ones_go():
    from examples.llm.redalert_autopilot import group_attacks

    raw = _raw(legal={"enemy_base": [85, 5]})  # base_center is [5, 5]
    raw["observation"]["visible_enemies"] = [{"id": 900, "type": "harv", "x": 80, "y": 10},
                                             {"id": 901, "type": "2tnk", "x": 80, "y": 10}]
    small = {"cmd": "attack_move", "units": [1, 2, 3], "to": [80, 10]}
    big = {"cmd": "attack_move", "units": list(range(1, 9)), "to": [80, 10]}
    raid = {"cmd": "attack", "units": [1, 2], "target": 900}
    poke = {"cmd": "attack", "units": [1, 2], "target": 901}
    scout = {"cmd": "move", "units": [7], "to": [80, 10]}
    home = {"cmd": "attack_move", "units": [4], "to": [8, 8]}

    assert group_attacks([small, big, raid, poke, scout, home], raw) == [
        {"cmd": "move", "units": [1, 2, 3], "to": [15, 5]}, big, raid,
        {"cmd": "move", "units": [1, 2], "to": [15, 5]}, scout, home,
    ]


def test_rocket_soldiers_are_kept_home_guarding_the_yard_and_from_the_model():
    pilot = Autopilot()
    units = HARVESTERS + [{"id": 400 + i, "type": "e3"} for i in range(8)]
    raw = _raw(buildings=READY, units=units, legal={"combat_units": [400 + i for i in range(8)]})

    home = [o for o in pilot.orders(raw, 0.0) if o["cmd"] == "guard" and o["target"] == 10]  # 10: the yard

    assert len(home) == 6
    assert pilot.strip_guards([{"cmd": "attack_move", "units": [400 + i for i in range(8)], "to": [9, 9]}]) == [
        {"cmd": "attack_move", "units": [406, 407], "to": [9, 9]}
    ]


def test_holding_a_won_game_recalls_the_army_and_drops_attacks():
    from examples.llm.redalert_autopilot import hold_lead

    raw = _raw(buildings=["fact"], units=[{"id": 50, "type": "2tnk", "x": 80, "y": 80}, {"id": 51, "type": "harv", "x": 80, "y": 80}])
    raw["observation"]["military"] = {"kills_cost": 20000, "deaths_cost": 5000}
    raw["time"] = {"remaining_seconds": 240}
    attack = {"cmd": "attack_move", "units": [50], "to": [80, 80]}
    defend = {"cmd": "attack_move", "units": [50], "to": [6, 6]}

    assert {"cmd": "move", "units": [50], "to": [5, 5]} in Autopilot().orders(raw, 0.0)
    assert hold_lead([attack, defend, {"cmd": "train", "item": "e3"}], raw) == [defend, {"cmd": "train", "item": "e3"}]

    raw["time"] = {"remaining_seconds": 900}
    assert hold_lead([attack], raw) == [attack]


def test_numbers_first_and_no_transports_or_anti_air():
    from examples.llm.redalert_autopilot import gate_model_spending

    few = _raw(buildings=READY, units=HARVESTERS + [{"id": 60, "type": "e1"}])
    many = _raw(buildings=READY, units=HARVESTERS + [{"id": 60 + i, "type": "e1"} for i in range(12)])
    model = [{"cmd": "train", "item": "3tnk"}, {"cmd": "train", "item": "apc", "count": 3},
             {"cmd": "build", "item": "fix"}, {"cmd": "train", "item": "e3", "count": 3}]

    assert gate_model_spending(model, few) == [{"cmd": "train", "item": "e3", "count": 3}]
    assert gate_model_spending(model, many) == [model[0], model[2], model[3]]


def test_steady_infantry_starts_with_the_first_refinery():
    raw = _raw(buildings=["fact", "powr", "barr", "proc"], available=["e1"])

    assert {"cmd": "train", "item": "e1", "count": 3} in Autopilot().orders(raw, 0.0)


def test_crushing_them_near_the_end_pushes_instead_of_holding():
    from examples.llm.redalert_autopilot import hold_lead

    army = [{"id": 500 + i, "type": "2tnk", "x": 5, "y": 5} for i in range(32)]
    attack = {"cmd": "attack_move", "units": [500], "to": [80, 80]}
    raw = _raw(buildings=READY, units=HARVESTERS + army,
               legal={"combat_units": [500 + i for i in range(32)], "attack_now": attack})
    raw["observation"]["military"] = {"kills_cost": 20000, "deaths_cost": 5000}
    raw["time"] = {"remaining_seconds": 200}
    raw["observation"]["visible_enemies"] = [{"id": 1, "type": "e3"}, {"id": 2, "type": "3tnk"}]

    assert attack in Autopilot().orders(raw, 0.0)
    assert hold_lead([attack], raw) == [attack]

    raw["observation"]["visible_enemies"] = [{"id": i, "type": "3tnk"} for i in range(10)]  # a real army
    assert attack not in Autopilot().orders(raw, 0.0)
    assert hold_lead([attack], raw) == []


def test_service_depot_is_built_once_the_infantry_screen_is_up():
    few = _raw(buildings=READY, units=HARVESTERS + [{"id": 70, "type": "e1"}], available=["fix"])
    many = _raw(buildings=READY, units=HARVESTERS + [{"id": 70 + i, "type": "e1"} for i in range(12)], available=["fix"])
    built = _raw(buildings=READY + ["fix"], units=HARVESTERS + [{"id": 70 + i, "type": "e1"} for i in range(12)],
                 available=["fix"])

    assert {"cmd": "build", "item": "fix"} not in Autopilot().orders(few, 0.0)
    assert {"cmd": "build", "item": "fix"} in Autopilot().orders(many, 0.0)
    assert {"cmd": "build", "item": "fix"} not in Autopilot().orders(built, 0.0)



# -- production planner -------------------------------------------------------------------


def _seen(raw, *types):
    raw["observation"]["visible_enemies"] = [{"id": 900 + i, "type": t} for i, t in enumerate(types)]
    return raw


def test_planner_answers_what_the_enemy_fields():
    rockets = _seen(_raw(buildings=READY, units=HARVESTERS, available=["e1", "e3", "3tnk"]), *["e3"] * 8)
    tanks = _seen(_raw(buildings=READY, units=HARVESTERS, available=["e1", "e3", "3tnk"]), *["3tnk"] * 6)

    vs_rockets = Autopilot().orders(rockets, 0.0)
    vs_tanks = Autopilot().orders(tanks, 0.0)

    assert {"cmd": "train", "item": "e1", "count": 3} in vs_rockets
    assert {"cmd": "train", "item": "e3", "count": 3} in vs_tanks
    assert {"cmd": "train", "item": "3tnk", "count": 1} in vs_tanks


def test_planner_never_builds_anti_air_or_transports_and_needs_money():
    raw = _raw(buildings=READY, units=HARVESTERS, available=["ftrk", "apc", "e3"])
    assert not any(o.get("item") in ("ftrk", "apc") for o in Autopilot().orders(raw, 0.0))

    raw["observation"]["economy"].update(cash=0, ore=0)
    from examples.llm.redalert_planner import Planner
    assert Planner().produce(raw["observation"], {"e3", "3tnk"}, 0.0) == [{"cmd": "train", "item": "3tnk", "count": 1}]


def test_planner_spreads_the_army_instead_of_one_type():
    from examples.llm.redalert_planner import Planner

    planner, mix = Planner(), {"infantry": 1.0, "tank": 0.0, "light": 0.0}
    assert planner.score("e1", mix, {"e1": 40, "e3": 2}) < planner.score("e1", mix, {"e1": 5, "e3": 5})


def test_model_unit_choices_are_checked_against_the_enemy():
    pilot = Autopilot()
    infantry = _seen(_raw(available=["e1", "e3"]), *["e3"] * 8)
    pilot.orders(infantry, 0.0)
    assert pilot.fit_to_enemy([{"cmd": "train", "item": "e3", "count": 3}], infantry) == [
        {"cmd": "train", "item": "e1", "count": 3}]

    mixed = Autopilot()
    tanky = _seen(_raw(available=["e1", "e3"]), *(["e3"] * 6 + ["3tnk"] * 4))
    mixed.orders(tanky, 0.0)
    order = [{"cmd": "train", "item": "e3", "count": 3}]
    assert mixed.fit_to_enemy(order, tanky) == order


def test_while_holding_the_planner_still_trains_and_status_shows_time_left():
    raw = _raw(buildings=READY, units=HARVESTERS, available=["2tnk", "e3"],
               legal={"combat_units": list(range(200, 215)), "attack_now": {"cmd": "attack_move", "units": [200], "to": [50, 50]}})
    raw["observation"]["military"] = {"kills_cost": 20000, "deaths_cost": 5000}
    raw["time"] = {"remaining_seconds": 240.7}

    orders = Autopilot().orders(raw, 0.0)
    assert any(o["cmd"] == "train" for o in orders)
    assert raw["legal_actions"]["attack_now"] not in orders
    assert "(left 4:00)" in redalert.status_line(raw)


def test_a_silent_model_still_gets_the_backup_attack():
    legal = {"combat_units": list(range(200, 212)), "attack_now": {"cmd": "attack_move", "units": [200], "to": [50, 50]}}
    raw = _raw(buildings=READY, units=HARVESTERS, available=["3tnk"], legal=legal)

    assert legal["attack_now"] in Autopilot().orders(raw, 0.0, model_silent=True)
    assert legal["attack_now"] not in Autopilot().orders(raw, 0.0)


def test_enemy_mix_is_weighted_by_value_so_tanks_count_for_their_cost():
    from examples.llm.redalert_planner import Planner

    planner = Planner()
    planner.observe({"visible_enemies": [{"id": i, "type": "e3"} for i in range(7)]
                     + [{"id": 50 + i, "type": "3tnk"} for i in range(5)]}, 0.0)
    mix = planner.enemy_mix(0.0)
    assert mix["tank"] > 0.7

    army = {"units": [{"id": i, "type": "e1"} for i in range(40)] + [{"id": 100 + i, "type": "e3"} for i in range(10)]}
    assert planner.best("infantry", {"e1", "e3"}, army, 0.0) == "e3"


def test_units_shoot_what_they_counter_not_what_is_in_front():
    raw = _raw(units=[{"id": 1, "type": "e1", "x": 10, "y": 10}, {"id": 2, "type": "e1", "x": 11, "y": 10},
                      {"id": 3, "type": "e3", "x": 10, "y": 11}])
    raw["observation"]["visible_enemies"] = [
        {"id": 90, "type": "3tnk", "x": 12, "y": 10},  # in front
        {"id": 91, "type": "e3", "x": 15, "y": 10},    # hiding behind it
        {"id": 92, "type": "e1", "x": 40, "y": 40},    # too far to matter
    ]

    orders = [o for o in Autopilot().orders(raw, 0.0) if o["cmd"] == "attack"]

    assert {"cmd": "attack", "units": [1, 2], "target": 91} in orders  # riflemen go for the rockets
    assert {"cmd": "attack", "units": [3], "target": 90} in orders     # rockets take the tank


def test_units_with_nothing_they_counter_nearby_are_left_alone():
    raw = _raw(units=[{"id": 3, "type": "e3", "x": 10, "y": 10}])
    raw["observation"]["visible_enemies"] = [{"id": 91, "type": "e1", "x": 12, "y": 10}]

    assert not any(o["cmd"] == "attack" for o in Autopilot().orders(raw, 0.0))


def test_flamethrowers_are_answered_with_tanks_and_hunted_by_them():
    flamers = _seen(_raw(buildings=READY, units=HARVESTERS, available=["e1", "e2", "e3", "3tnk"]), *["e4"] * 20)

    trains = [o for o in Autopilot().orders(flamers, 0.0) if o["cmd"] == "train"]
    assert {"cmd": "train", "item": "3tnk", "count": 1} in trains
    assert not any(o["item"] in ("e1", "e2") for o in trains)  # no riflemen fed to them (e3: home guard)

    raw = _raw(units=[{"id": 5, "type": "3tnk", "x": 10, "y": 10}])
    raw["observation"]["visible_enemies"] = [{"id": 80, "type": "2tnk", "x": 11, "y": 10},
                                             {"id": 81, "type": "e4", "x": 13, "y": 10}]
    assert {"cmd": "attack", "units": [5], "target": 81} in Autopilot().orders(raw, 0.0)


def test_tanks_are_queued_without_saving_up_and_infantry_waits_for_spare_money():
    from examples.llm.redalert_planner import Planner

    obs = {"economy": {"cash": 0, "ore": 200}, "units": [], "production": []}
    orders = Planner().produce(obs, {"e2", "3tnk"}, 0.0)
    assert {"cmd": "train", "item": "3tnk", "count": 1} in orders
    assert not any(o["item"] == "e2" for o in orders)

    obs["economy"]["ore"] = 2000
    assert any(o["item"] == "e2" for o in Planner().produce(obs, {"e2", "3tnk"}, 0.0))
