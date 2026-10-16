"""Red Alert production planner: build what beats what the enemy fields.

Every view it updates what it has seen of the enemy army (the last
ENEMY_WINDOW_SECONDS, by category), scores each unit we can build now against
that mix, per credit, minus a penalty for what we already have too much of,
and keeps the infantry and vehicle queues busy with the best choice. It also
replaces a model order whose unit scores far below the best one.

The numbers are judgement, not engine data: how well one of our units fights
each enemy category, with infantry cheap, rockets for armour, flamers and
riflemen for infantry, and heavier tanks better against everything.
"""

from __future__ import annotations

from typing import Any

INFANTRY_CAT, TANK_CAT, LIGHT_CAT, FLAME_CAT = "infantry", "tank", "light", "flame"
CATEGORIES = (INFANTRY_CAT, TANK_CAT, LIGHT_CAT, FLAME_CAT)
# Flamethrowers are their own threat: they roast massed infantry up close but
# barely scratch armour, and tanks crush them. Twenty arrived at once.
CATEGORY = {
    **{t: INFANTRY_CAT for t in ("e1", "e2", "e3", "e6", "dog", "medi", "spy", "thf", "shok", "e7")},
    "e4": FLAME_CAT,
    **{t: TANK_CAT for t in ("2tnk", "3tnk", "4tnk", "ttnk", "ctnk", "stnk", "qtnk")},
    **{t: LIGHT_CAT for t in ("1tnk", "jeep", "apc", "ftrk", "arty", "v2rl", "mgg", "mrj", "truk")},
}

# How well one of our units does against each enemy category (1.0 = good).
MATCHUP = {
    "e1": {INFANTRY_CAT: 1.0, TANK_CAT: 0.15, LIGHT_CAT: 0.4, FLAME_CAT: 0.5},
    "e2": {INFANTRY_CAT: 0.9, TANK_CAT: 0.4, LIGHT_CAT: 0.6, FLAME_CAT: 0.6},
    "e3": {INFANTRY_CAT: 0.25, TANK_CAT: 1.0, LIGHT_CAT: 0.9, FLAME_CAT: 0.2},
    "e4": {INFANTRY_CAT: 1.3, TANK_CAT: 0.15, LIGHT_CAT: 0.3, FLAME_CAT: 0.6},
    "shok": {INFANTRY_CAT: 0.8, TANK_CAT: 0.8, LIGHT_CAT: 0.8, FLAME_CAT: 0.7},
    "1tnk": {INFANTRY_CAT: 0.6, TANK_CAT: 0.45, LIGHT_CAT: 0.8, FLAME_CAT: 0.9},
    "2tnk": {INFANTRY_CAT: 0.75, TANK_CAT: 0.9, LIGHT_CAT: 1.0, FLAME_CAT: 1.2},
    "3tnk": {INFANTRY_CAT: 0.85, TANK_CAT: 1.1, LIGHT_CAT: 1.1, FLAME_CAT: 1.3},
    "4tnk": {INFANTRY_CAT: 1.1, TANK_CAT: 1.4, LIGHT_CAT: 1.3, FLAME_CAT: 1.5},
    "ttnk": {INFANTRY_CAT: 1.0, TANK_CAT: 0.9, LIGHT_CAT: 1.0, FLAME_CAT: 1.1},
    "arty": {INFANTRY_CAT: 0.9, TANK_CAT: 0.35, LIGHT_CAT: 0.4, FLAME_CAT: 0.6},
    "v2rl": {INFANTRY_CAT: 0.9, TANK_CAT: 0.45, LIGHT_CAT: 0.4, FLAME_CAT: 0.6},
}
COST = {"e1": 100, "e2": 160, "e3": 300, "e4": 300, "shok": 400, "1tnk": 700, "2tnk": 850,
        "3tnk": 1150, "4tnk": 1700, "ttnk": 1350, "arty": 600, "v2rl": 700}
DEFAULT_VALUE = {INFANTRY_CAT: 200, TANK_CAT: 1100, LIGHT_CAT: 650, FLAME_CAT: 300}  # enemy types missing from COST
INFANTRY_UNITS = ("e1", "e2", "e3", "e4", "shok")
VEHICLE_UNITS = ("1tnk", "2tnk", "3tnk", "4tnk", "ttnk", "arty", "v2rl")

# Before anything is seen: what this opponent has fielded (rocket floods, then tanks).
DEFAULT_MIX = {INFANTRY_CAT: 0.6, TANK_CAT: 0.3, LIGHT_CAT: 0.1, FLAME_CAT: 0.0}
ENEMY_WINDOW_SECONDS = 120.0
REPLACE_BELOW = 0.85
# When the best infantry is this much worse than the best vehicle (e.g. against
# flamethrowers), the barracks stops feeding infantry into it.
INFANTRY_SKIP_BELOW = 0.8
INFANTRY_RESERVE = 300  # infantry waits until this much beyond its price is spare (tanks first)  # a model pick scoring under this fraction of the best is replaced
BATCH = {"infantry": 3, "vehicle": 1}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


class Planner:
    def __init__(self) -> None:
        self._seen: dict[int, tuple[str, float, float]] = {}  # enemy id -> (category, value, last seen)
        self.last_plan: dict[str, str | None] = {"infantry": None, "vehicle": None}

    def observe(self, obs: dict[str, Any], now: float) -> None:
        for e in _list(obs.get("visible_enemies")):
            kind = str(e.get("type", "")).lower()
            cat = CATEGORY.get(kind)
            if cat and not e.get("wreck") and isinstance(e.get("id"), int):
                self._seen[e["id"]] = (cat, COST.get(kind, DEFAULT_VALUE[cat]), now)

    def enemy_mix(self, now: float) -> dict[str, float]:
        """The enemy army seen recently, by share of its value: seven rocket
        soldiers and five heavy tanks are mostly tanks, not mostly infantry."""
        value = {c: 0.0 for c in CATEGORIES}
        for cat, worth, seen in self._seen.values():
            if now - seen <= ENEMY_WINDOW_SECONDS:
                value[cat] += worth
        total = sum(value.values())
        return {c: v / total for c, v in value.items()} if total else dict(DEFAULT_MIX)

    def score(self, unit: str, mix: dict[str, float], ours: dict[str, int]) -> float:
        """How much this unit adds: its matchup against each enemy category,
        weighted by how much of that category our army does not yet cover.
        Cost counts only weakly (fourth root), so cheap riflemen don't win by
        price alone: 51 of them died to four heavy tanks and flamers."""
        matchup = MATCHUP.get(unit)
        if not matchup:
            return 0.0
        cover = _coverage(ours)
        need = {c: max(0.0, mix.get(c, 0.0) - cover[c]) + 0.25 * mix.get(c, 0.0) for c in mix}
        value = sum(matchup[c] * need[c] for c in need)
        return value * (1000 / COST.get(unit, 1000)) ** 0.25

    def best(self, kind: str, available: set[str], obs: dict[str, Any], now: float) -> str | None:
        options = [u for u in (INFANTRY_UNITS if kind == "infantry" else VEHICLE_UNITS) if u in available]
        if not options:
            return None
        mix, ours = self.enemy_mix(now), _our_army(obs)
        return max(options, key=lambda u: self.score(u, mix, ours))

    def produce(self, obs: dict[str, Any], available: set[str], now: float) -> list[dict[str, Any]]:
        """Keep each idle queue (infantry, vehicle) on the best unit we can afford."""
        production = _list(obs.get("production"))
        eco = _dict(obs.get("economy"))
        money = (eco.get("cash") or 0) + (eco.get("ore") or 0)
        picks = {kind: self.best(kind, available, obs, now) for kind in ("infantry", "vehicle")}
        self.last_plan.update(picks)
        mix, ours = self.enemy_mix(now), _our_army(obs)
        if picks["infantry"] and picks["vehicle"] and self.score(picks["infantry"], mix, ours) < (
            INFANTRY_SKIP_BELOW * self.score(picks["vehicle"], mix, ours)
        ):
            picks["infantry"] = None  # tanks only: infantry would just feed the flamers
        # Production is paid as it goes, so a tank is queued whenever the factory
        # is idle, whatever the cash. Infantry only takes money that is spare:
        # waiting to afford a 1150 tank while 160 grenadiers were always
        # affordable meant no tank was ever built.
        out = []
        vehicle_busy = any(str(p.get("queue")) == "Vehicle" for p in production)
        if picks["vehicle"] and not vehicle_busy:
            out.append({"cmd": "train", "item": picks["vehicle"], "count": BATCH["vehicle"]})
        tanks_wanted = picks["vehicle"] is not None
        infantry_busy = any(str(p.get("queue")) == "Infantry" for p in production)
        pick = picks["infantry"]
        if pick and not infantry_busy and (not tanks_wanted or money >= COST.get(pick, 0) + INFANTRY_RESERVE):
            out.append({"cmd": "train", "item": pick, "count": BATCH["infantry"]})
        return out

    def refine(self, model_orders: list[dict[str, Any]], obs: dict[str, Any], available: set[str],
               now: float) -> list[dict[str, Any]]:
        """A model train order for a unit far worse than the best in its queue
        becomes the best one."""
        mix, ours = self.enemy_mix(now), _our_army(obs)
        out = []
        for order in model_orders:
            item = str(order.get("item", "")).lower()
            kind = "infantry" if item in INFANTRY_UNITS else "vehicle" if item in VEHICLE_UNITS else None
            if order.get("cmd") == "train" and kind:
                best = self.best(kind, available, obs, now)
                if best and self.score(item, mix, ours) < REPLACE_BELOW * self.score(best, mix, ours):
                    order = {**order, "item": best}
            out.append(order)
        return out


def _coverage(ours: dict[str, int]) -> dict[str, float]:
    """Share of our army's fighting power (matchup x credits) against each category."""
    power = {c: sum(n * MATCHUP[u][c] * COST.get(u, 0) for u, n in ours.items()) for c in CATEGORIES}
    total = sum(power.values())
    return {c: (v / total if total else 0.0) for c, v in power.items()}


def _our_army(obs: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for u in _list(obs.get("units")):
        t = str(u.get("type", "")).lower()
        if t in MATCHUP and not u.get("wreck"):
            counts[t] = counts.get(t, 0) + 1
    return counts
