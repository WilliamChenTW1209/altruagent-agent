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

INFANTRY_CAT, TANK_CAT, LIGHT_CAT = "infantry", "tank", "light"
CATEGORY = {
    **{t: INFANTRY_CAT for t in ("e1", "e2", "e3", "e4", "e6", "dog", "medi", "spy", "thf", "shok", "e7")},
    **{t: TANK_CAT for t in ("2tnk", "3tnk", "4tnk", "ttnk", "ctnk", "stnk", "qtnk")},
    **{t: LIGHT_CAT for t in ("1tnk", "jeep", "apc", "ftrk", "arty", "v2rl", "mgg", "mrj", "truk")},
}

# How well one of our units does against each enemy category (1.0 = good).
MATCHUP = {
    "e1": {INFANTRY_CAT: 1.0, TANK_CAT: 0.15, LIGHT_CAT: 0.4},
    "e2": {INFANTRY_CAT: 0.9, TANK_CAT: 0.4, LIGHT_CAT: 0.6},
    "e3": {INFANTRY_CAT: 0.25, TANK_CAT: 1.0, LIGHT_CAT: 0.9},
    "e4": {INFANTRY_CAT: 1.3, TANK_CAT: 0.15, LIGHT_CAT: 0.3},
    "shok": {INFANTRY_CAT: 0.8, TANK_CAT: 0.8, LIGHT_CAT: 0.8},
    "1tnk": {INFANTRY_CAT: 0.6, TANK_CAT: 0.45, LIGHT_CAT: 0.8},
    "2tnk": {INFANTRY_CAT: 0.75, TANK_CAT: 0.9, LIGHT_CAT: 1.0},
    "3tnk": {INFANTRY_CAT: 0.85, TANK_CAT: 1.1, LIGHT_CAT: 1.1},
    "4tnk": {INFANTRY_CAT: 1.1, TANK_CAT: 1.4, LIGHT_CAT: 1.3},
    "ttnk": {INFANTRY_CAT: 1.0, TANK_CAT: 0.9, LIGHT_CAT: 1.0},
    "arty": {INFANTRY_CAT: 0.9, TANK_CAT: 0.35, LIGHT_CAT: 0.4},
    "v2rl": {INFANTRY_CAT: 0.9, TANK_CAT: 0.45, LIGHT_CAT: 0.4},
}
COST = {"e1": 100, "e2": 160, "e3": 300, "e4": 300, "shok": 400, "1tnk": 700, "2tnk": 850,
        "3tnk": 1150, "4tnk": 1700, "ttnk": 1350, "arty": 600, "v2rl": 700}
INFANTRY_UNITS = ("e1", "e2", "e3", "e4", "shok")
VEHICLE_UNITS = ("1tnk", "2tnk", "3tnk", "4tnk", "ttnk", "arty", "v2rl")

# Before anything is seen: what this opponent has fielded (rocket floods, then tanks).
DEFAULT_MIX = {INFANTRY_CAT: 0.6, TANK_CAT: 0.3, LIGHT_CAT: 0.1}
ENEMY_WINDOW_SECONDS = 120.0
REPLACE_BELOW = 0.6  # a model pick scoring under this fraction of the best is replaced
BATCH = {"infantry": 3, "vehicle": 1}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


class Planner:
    def __init__(self) -> None:
        self._seen: dict[int, tuple[str, float]] = {}  # enemy id -> (category, last seen)
        self.last_plan: dict[str, str | None] = {"infantry": None, "vehicle": None}

    def observe(self, obs: dict[str, Any], now: float) -> None:
        for e in _list(obs.get("visible_enemies")):
            cat = CATEGORY.get(str(e.get("type", "")).lower())
            if cat and not e.get("wreck") and isinstance(e.get("id"), int):
                self._seen[e["id"]] = (cat, now)

    def enemy_mix(self, now: float) -> dict[str, float]:
        recent = [c for c, t in self._seen.values() if now - t <= ENEMY_WINDOW_SECONDS]
        if not recent:
            return dict(DEFAULT_MIX)
        return {c: recent.count(c) / len(recent) for c in (INFANTRY_CAT, TANK_CAT, LIGHT_CAT)}

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
        out = []
        for kind, queue in (("infantry", "Infantry"), ("vehicle", "Vehicle")):
            pick = self.best(kind, available, obs, now)
            self.last_plan[kind] = pick
            busy = any(str(p.get("queue")) == queue for p in production)
            if pick and not busy and money >= COST.get(pick, 0):
                out.append({"cmd": "train", "item": pick, "count": BATCH[kind]})
                money -= COST.get(pick, 0)
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
    power = {c: sum(n * MATCHUP[u][c] * COST.get(u, 0) for u, n in ours.items()) for c in (INFANTRY_CAT, TANK_CAT, LIGHT_CAT)}
    total = sum(power.values())
    return {c: (v / total if total else 0.0) for c, v in power.items()}


def _our_army(obs: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for u in _list(obs.get("units")):
        t = str(u.get("type", "")).lower()
        if t in MATCHUP and not u.get("wreck"):
            counts[t] = counts.get(t, 0) + 1
    return counts
