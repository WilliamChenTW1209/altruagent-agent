"""Red Alert reflexes: the orders that need no judgement, sent on every view.

The model is slow (seconds per answer) and the world isn't, so the jobs a
fixed rule does well are done here, in code, about 5 times a second:

- deploy the MCV;
- place every finished building at once (a waiting building blocks its queue);
- keep power ahead of demand;
- the economic core: power plant, barracks, refinery, war factory, and
  harvesters up to ``TARGET_HARVESTERS``;
- send idle combat units to whatever of ours is under attack.

Everything else (what army to build, when and where to attack) is the
model's. ``AUTOPILOT_NOTE`` tells it so, and ``drop_model_duplicates`` keeps
it from queueing what the autopilot already queued.
"""

from __future__ import annotations

from typing import Any

POWER_PLANT = "powr"
BARRACKS = ("tent", "barr")
REFINERY = "proc"
WAR_FACTORY = "weap"
HARVESTER = "harv"
MCV = "mcv"
CONSTRUCTION_YARD = "fact"

TARGET_HARVESTERS = 3
POWER_MARGIN = 40  # build a power plant when spare power falls below this
REPEAT_SECONDS = 3.0  # don't resend the same autopilot order sooner (verdicts lag)
UNDER_ATTACK_TICKS = 25 * 6  # react to under_attack events from the last 6 s
DEFENDERS_PER_ALERT = 50
HOME_RADIUS = 20  # cells from base_center that count as home for defending

AUTOPILOT_NOTE = (
    "AUTOPILOT: code already handles, on every view and without you: deploying the MCV, "
    "placing every finished building, power plants, the economic core (power plant, barracks, "
    "refinery, war factory) and harvesters up to 3, and sending idle combat units to defend "
    "what is under attack. Do NOT order those yourself. Your job is the army and the plan: "
    "which units and defenses to build, extra production buildings and tech, scouting, and "
    "when and where to attack or retreat."
)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _types(actors: list[dict[str, Any]]) -> list[str]:
    return [str(a.get("type", "")).lower() for a in actors if not a.get("wreck")]


class Autopilot:
    """One match's reflexes. ``orders(state, now)`` returns the batch to send."""

    def __init__(self) -> None:
        self._sent: dict[str, float] = {}  # order key -> when last sent

    def orders(self, state: dict[str, Any], now: float) -> list[dict[str, Any]]:
        obs = _dict(state.get("observation"))
        legal = _dict(state.get("legal_actions"))
        available = {str(i).lower() for i in obs.get("available_production") or []}
        queued = {str(p.get("item", "")).lower() for p in _list(obs.get("production"))}
        buildings = _types(_list(obs.get("buildings")))
        units = _list(obs.get("units"))
        out: list[dict[str, Any]] = []

        # Deploy the MCV (only while there is no construction yard).
        if CONSTRUCTION_YARD not in buildings and legal.get("deploy"):
            out.append({"cmd": "deploy", "units": [int(u) for u in legal["deploy"][:1]]})

        # Place every finished building.
        for item in legal.get("place") or []:
            name = item.get("item") if isinstance(item, dict) else item
            if name:
                out.append({"cmd": "place", "item": str(name).lower()})

        # Power first, then the economic core in order, one building at a time.
        eco = _dict(obs.get("economy"))
        provided, drained = eco.get("power_provided") or 0, eco.get("power_drained") or 0
        building_busy = any(str(p.get("queue")) == "Building" and not p.get("ready") for p in _list(obs.get("production")))
        want = None
        if POWER_PLANT not in buildings or provided - drained < POWER_MARGIN:
            want = POWER_PLANT
        elif not any(b in buildings for b in BARRACKS):
            want = next((b for b in BARRACKS if b in available), None)
        elif REFINERY not in buildings:
            want = REFINERY
        elif WAR_FACTORY not in buildings:
            want = WAR_FACTORY
        if want and want in available and want not in queued and not building_busy:
            out.append({"cmd": "build", "item": want})

        # Harvesters (a refinery brings one; the war factory trains more).
        harvesters = sum(1 for t in _types(units) if t == HARVESTER)
        if REFINERY in buildings and harvesters < TARGET_HARVESTERS and HARVESTER in available and HARVESTER not in queued:
            out.append({"cmd": "train", "item": HARVESTER, "count": 1})

        out.extend(self._defend(obs, legal))
        return [o for o in out if self._fresh(o, now)]

    def _defend(self, obs: dict[str, Any], legal: dict[str, Any]) -> list[dict[str, Any]]:
        """Idle combat units to the newest thing of ours under attack."""
        defenders = [int(u) for u in legal.get("combat_units") or []][:DEFENDERS_PER_ALERT]
        if not defenders:
            return []
        tick = obs.get("tick") or 0
        alerts = [e for e in _list(obs.get("events")) if e.get("kind") == "under_attack"
                  and tick - (e.get("tick") or 0) <= UNDER_ATTACK_TICKS]
        if not alerts:
            return []
        # Only what is attacked near home: a scout hit at the enemy base must not
        # pull the whole army after it.
        base = obs.get("base_center")
        for alert in reversed(alerts):
            cell = _cell_of(alert, obs)
            if cell and _near(cell, base):
                return [{"cmd": "attack_move", "units": defenders, "to": list(cell)}]
        return []

    def _fresh(self, order: dict[str, Any], now: float) -> bool:
        key = repr(sorted(order.items()))
        if now - self._sent.get(key, -1e9) < REPEAT_SECONDS:
            return False
        self._sent[key] = now
        return True


def _near(cell: tuple[int, int], base: Any) -> bool:
    """Within HOME_RADIUS cells of the base centre (no known base: never)."""
    if not (isinstance(base, (list, tuple)) and len(base) == 2):
        return False
    return max(abs(cell[0] - base[0]), abs(cell[1] - base[1])) <= HOME_RADIUS


def _cell_of(event: dict[str, Any], obs: dict[str, Any]) -> tuple[int, int] | None:
    """Where an event happened: its own x/y, else its actor's position."""
    if isinstance(event.get("x"), int) and isinstance(event.get("y"), int):
        return event["x"], event["y"]
    actor_id = next((event[k] for k in ("actor", "actor_id", "id", "building", "unit") if isinstance(event.get(k), int)), None)
    for actor in _list(obs.get("buildings")) + _list(obs.get("units")):
        if actor.get("id") == actor_id and isinstance(actor.get("x"), int) and isinstance(actor.get("y"), int):
            return actor["x"], actor["y"]
    return None


def drop_model_duplicates(model_orders: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    """The model's orders minus the autopilot's jobs (deploy, place, the core
    buildings and harvesters, when those are already handled)."""
    obs = _dict(state.get("observation"))
    queued = {str(p.get("item", "")).lower() for p in _list(obs.get("production"))}
    core = {POWER_PLANT, REFINERY, WAR_FACTORY, HARVESTER, *BARRACKS}
    kept = []
    for order in model_orders:
        cmd, item = order.get("cmd"), str(order.get("item", "")).lower()
        if cmd in ("deploy", "place"):
            continue
        if cmd in ("build", "train") and item in core and item in queued:
            continue
        kept.append(order)
    return kept
