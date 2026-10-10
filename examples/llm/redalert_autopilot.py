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
TARGET_REFINERIES = 2  # each comes with a free harvester, and needs no war factory
EARLY_INFANTRY = 3  # what the model may train before the economy is up
INFANTRY = ("e1", "e2", "e3", "e4", "e6", "dog", "medi", "spy", "thf")
TANKS = ("1tnk", "2tnk", "3tnk", "4tnk", "apc", "ftrk", "jeep", "arty", "v2rl", "ttnk", "ctnk", "stnk", "mgg", "mrj")
ANTI_INFANTRY = ("e1", "e2", "e4", "dog")  # what beats rocket infantry
ROCKETS = ("e3",)
ONE_IS_ENOUGH = ("fix", "dome", "atek", "stek", "gap", "iron", "pdox", "mslo")
ENEMY_MEMORY_SECONDS = 60.0  # enemies seen this recently still count toward their army
COUNTER_AT = 3  # react to an enemy type once we have seen this many
COUNTER_BATCH = 5
POWER_MARGIN = 40  # build a power plant when spare power falls below this
REPEAT_SECONDS = 3.0  # don't resend the same autopilot order sooner (verdicts lag)
UNDER_ATTACK_TICKS = 25 * 6  # react to under_attack events from the last 6 s
DEFENDERS_PER_ALERT = 50
HOME_RADIUS = 20  # cells from base_center that count as home for defending

AUTOPILOT_NOTE = (
    "AUTOPILOT: code already handles, on every view and without you: deploying the MCV, "
    "placing every finished building, power plants, the economic core (power plant, barracks, "
    "two refineries, war factory) and harvesters up to 3, and sending idle combat units to defend "
    "what is under attack near home. Do NOT order those yourself. Until that core is up, your "
    "build orders and all training beyond 3 infantry are dropped, so the cash goes to the economy. "
    "Once 2tnk can be built, any 1tnk you order is trained as 2tnk. A building already queued "
    "is never queued again. Code also trains anti-infantry when it sees many enemy rocket "
    "soldiers, and rocket soldiers (e3) when it sees many enemy tanks. "
    "Your job is the army and the plan: "
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
        self._enemies: dict[int, tuple[str, float]] = {}  # enemy id -> (type, last seen)

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
        elif buildings.count(REFINERY) < TARGET_REFINERIES:
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
        out.extend(self._counter(state, obs, available, queued, now))
        return [o for o in out if self._fresh(o, now)]

    def _counter(self, state: dict[str, Any], obs: dict[str, Any], available: set[str], queued: set[str],
                 now: float) -> list[dict[str, Any]]:
        """Answer the enemy's army with what beats it: many rocket soldiers ->
        anti-infantry; many tanks -> rocket soldiers. The model, told so in the
        strategy, kept building tanks into rocket soldiers anyway."""
        for e in _list(obs.get("visible_enemies")):
            if not e.get("wreck") and isinstance(e.get("id"), int):
                self._enemies[e["id"]] = (str(e.get("type", "")).lower(), now)
        recent = [t for t, seen in self._enemies.values() if now - seen <= ENEMY_MEMORY_SECONDS]
        if not economy_ready(state):
            return []
        ours = _types(_list(obs.get("units")))
        enemy_rockets = sum(1 for t in recent if t in ROCKETS)
        enemy_tanks = sum(1 for t in recent if t in TANKS)
        out = []
        anti = next((i for i in ANTI_INFANTRY if i in available), None)
        if anti and enemy_rockets >= COUNTER_AT and sum(1 for t in ours if t in ANTI_INFANTRY) < enemy_rockets                 and anti not in queued:
            out.append({"cmd": "train", "item": anti, "count": COUNTER_BATCH})
        if "e3" in available and enemy_tanks >= COUNTER_AT and sum(1 for t in ours if t in ROCKETS) < enemy_tanks                 and "e3" not in queued:
            out.append({"cmd": "train", "item": "e3", "count": COUNTER_BATCH})
        return out

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
        # Harvesters are the exception: they are the economy, defend them anywhere.
        base = obs.get("base_center")
        harvesters = {u.get("id") for u in _list(obs.get("units")) if str(u.get("type", "")).lower() == HARVESTER}
        for alert in reversed(alerts):
            cell = _cell_of(alert, obs)
            if cell and (_near(cell, base) or _actor_id(alert) in harvesters):
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


def _actor_id(event: dict[str, Any]) -> int | None:
    return next((event[k] for k in ("actor", "actor_id", "id", "building", "unit") if isinstance(event.get(k), int)), None)


def _cell_of(event: dict[str, Any], obs: dict[str, Any]) -> tuple[int, int] | None:
    """Where an event happened: its own x/y, else its actor's position."""
    if isinstance(event.get("x"), int) and isinstance(event.get("y"), int):
        return event["x"], event["y"]
    actor_id = _actor_id(event)
    for actor in _list(obs.get("buildings")) + _list(obs.get("units")):
        if actor.get("id") == actor_id and isinstance(actor.get("x"), int) and isinstance(actor.get("y"), int):
            return actor["x"], actor["y"]
    return None


def economy_ready(state: dict[str, Any]) -> bool:
    """The core is up: a war factory, TARGET_REFINERIES refineries and
    TARGET_HARVESTERS harvesters. Until then the cash is the autopilot's."""
    obs = _dict(state.get("observation"))
    buildings = _types(_list(obs.get("buildings")))
    harvesters = sum(1 for t in _types(_list(obs.get("units"))) if t == HARVESTER)
    return (
        WAR_FACTORY in buildings
        and buildings.count(REFINERY) >= TARGET_REFINERIES
        and harvesters >= TARGET_HARVESTERS
    )


def gate_model_spending(model_orders: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    """Before the economy is up, the model may not spend: no ``build``, and
    ``train`` only infantry while it has fewer than EARLY_INFANTRY. Twice the
    model spent the opening cash on riflemen and the economy never started."""
    if economy_ready(state):
        return model_orders
    obs = _dict(state.get("observation"))
    infantry = sum(1 for t in _types(_list(obs.get("units"))) if t in INFANTRY) + sum(
        1 for p in _list(obs.get("production")) if str(p.get("item", "")).lower() in INFANTRY
    )
    kept = []
    for order in model_orders:
        if order.get("cmd") == "build":
            continue
        if order.get("cmd") == "train":
            room = EARLY_INFANTRY - infantry
            if str(order.get("item", "")).lower() not in INFANTRY or room <= 0:
                continue
            order = {**order, "count": max(1, min(int(order.get("count") or 1), room))}
            infantry += order["count"]
        kept.append(order)
    return kept


TANK_UPGRADES = {"1tnk": ("2tnk",)}  # light tank -> what replaces it once available


def upgrade_tanks(model_orders: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    """A ``train 1tnk`` becomes ``2tnk`` once that can be built: light tanks
    lost head-on to heavy ones, and the model kept ordering them."""
    available = {str(i).lower() for i in _dict(state.get("observation")).get("available_production") or []}
    out = []
    for order in model_orders:
        item = str(order.get("item", "")).lower()
        better = next((b for b in TANK_UPGRADES.get(item, ()) if b in available), None)
        out.append({**order, "item": better} if order.get("cmd") == "train" and better else order)
    return out


def drop_model_duplicates(model_orders: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    """The model's orders minus the autopilot's jobs (deploy, place, the core
    buildings and harvesters, when those are already handled)."""
    obs = _dict(state.get("observation"))
    queued = {str(p.get("item", "")).lower() for p in _list(obs.get("production"))}
    built = set(_types(_list(obs.get("buildings"))))
    core = {POWER_PLANT, REFINERY, WAR_FACTORY, HARVESTER, *BARRACKS}
    kept, building = [], set()
    for order in model_orders:
        cmd, item = order.get("cmd"), str(order.get("item", "")).lower()
        if cmd in ("deploy", "place"):
            continue
        if cmd in ("build", "train") and item in core and item in queued:
            continue
        # A building already queued (or one you only ever need, already built)
        # is not queued again: twice a second service depot cost 1200.
        if cmd == "build" and (item in queued or item in building or (item in ONE_IS_ENOUGH and item in built)):
            continue
        if cmd == "build":
            building.add(item)
        kept.append(order)
    return kept
