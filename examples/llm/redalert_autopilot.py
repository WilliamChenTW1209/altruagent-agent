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
# Answers to an opponent who raids harvesters with rocket soldiers.
GUARDS_PER_HARVESTER = 2
MAX_GUARDS = 6
REFINERY_DEFENSES = ("ftur", "pbox", "gun", "tsla")  # anti-infantry first
STEADY_INFANTRY = 8  # anti-infantry kept on hand once the economy is up

# Backup army, only while the model is silent (API down, out of credit, slow).
MODEL_SILENT_SECONDS = 30.0
BACKUP_TANKS = ("4tnk", "3tnk", "2tnk", "1tnk", "ftrk")  # best first
BACKUP_ATTACK_AT = 12  # idle combat units before the backup sends them in
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
    "soldiers, and rocket soldiers (e3) when it sees many enemy tanks. Once the economy is up it "
    "also keeps 8 anti-infantry on hand, builds one anti-infantry defense beside each refinery, "
    "and keeps 2 infantry guarding each harvester (up to 6); orders you give those guards are dropped. "
    "Attacks away from home need at least 8 units in one order: a smaller group is sent to the "
    "staging point in front of the base instead, so gather the army there and attack all at once "
    "(a single scout, and raids that target a visible enemy harvester, are allowed). "
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
        self.guards: set[int] = set()  # our units assigned to guard harvesters

    def orders(self, state: dict[str, Any], now: float, *, model_silent: bool = False) -> list[dict[str, Any]]:
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

        # Place every finished building (defenses next to a refinery).
        for item in legal.get("place") or []:
            name = item.get("item") if isinstance(item, dict) else item
            if name:
                order: dict[str, Any] = {"cmd": "place", "item": str(name).lower()}
                spot = _defense_spot(obs) if order["item"] in REFINERY_DEFENSES else None
                if spot:
                    order["at"] = list(spot)
                out.append(order)

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

        out.extend(_free_cash_for_core(obs))
        out.extend(self._defend(obs, legal))
        out.extend(self._counter(state, obs, available, queued, now))
        if economy_ready(state):
            out.extend(self._guard_harvesters(obs, legal))
            out.extend(_refinery_defense(obs, available, queued, building_busy))
            out.extend(_steady_infantry(obs, available, queued))
        if model_silent:
            out.extend(_backup_army(state, legal, available, queued))
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

    def _guard_harvesters(self, obs: dict[str, Any], legal: dict[str, Any]) -> list[dict[str, Any]]:
        """Keep a few infantry guarding each harvester, out of the attacks."""
        units = {u.get("id"): str(u.get("type", "")).lower() for u in _list(obs.get("units")) if not u.get("wreck")}
        harvesters = [i for i, t in units.items() if t == HARVESTER]
        idle = {int(u) for u in legal.get("combat_units") or []}
        self.guards = {g for g in self.guards if g in units and g not in idle}  # dead or idle: reassign
        if not harvesters:
            return []
        want = min(MAX_GUARDS, GUARDS_PER_HARVESTER * len(harvesters)) - len(self.guards)
        pick = sorted((u for u in idle if units.get(u) in ANTI_INFANTRY), reverse=True)[: max(0, want)]
        out = []
        for n, unit in enumerate(pick):
            self.guards.add(unit)
            out.append({"cmd": "guard", "units": [unit], "target": harvesters[n % len(harvesters)]})
        return out

    def strip_guards(self, model_orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The model's unit orders without our harvester guards in them."""
        out = []
        for order in model_orders:
            units = order.get("units")
            if isinstance(units, list) and self.guards:
                units = [u for u in units if u not in self.guards]
                if not units:
                    continue
                order = {**order, "units": units}
            out.append(order)
        return out

    def _fresh(self, order: dict[str, Any], now: float) -> bool:
        key = repr(sorted(order.items()))
        if now - self._sent.get(key, -1e9) < REPEAT_SECONDS:
            return False
        self._sent[key] = now
        return True


MIN_ATTACK_GROUP = 8  # units that must go out together; fewer wait at the staging point
STAGING_DISTANCE = 10  # cells from the base centre toward the enemy


def _staging(obs: dict[str, Any], legal: dict[str, Any]) -> list[int] | None:
    base, enemy = obs.get("base_center"), legal.get("enemy_base")
    if not (isinstance(base, (list, tuple)) and isinstance(enemy, (list, tuple))):
        return None
    dx, dy = enemy[0] - base[0], enemy[1] - base[1]
    far = max(abs(dx), abs(dy)) or 1
    step = min(1.0, STAGING_DISTANCE / far)
    return [int(base[0] + dx * step), int(base[1] + dy * step)]


def group_attacks(model_orders: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    """No trickling into the enemy: an order sending fewer than
    MIN_ATTACK_GROUP units away from home becomes a move to the staging point,
    so the army gathers and goes in together. Kept as they are: a single
    scout's move, raids on a visible enemy harvester, and anything near home."""
    obs, legal = _dict(state.get("observation")), _dict(state.get("legal_actions"))
    base, staging = obs.get("base_center"), _staging(obs, legal)
    enemies = {e.get("id"): e for e in _list(obs.get("visible_enemies"))}
    out = []
    for order in model_orders:
        cmd, units = order.get("cmd"), order.get("units") or []
        if cmd not in ("move", "attack_move", "attack") or not staging or len(units) >= MIN_ATTACK_GROUP:
            out.append(order)
            continue
        target = enemies.get(order.get("target"))
        if target and str(target.get("type", "")).lower() == HARVESTER:
            out.append(order)  # a raid on their economy
            continue
        if cmd == "move" and len(units) == 1:
            out.append(order)  # a scout
            continue
        to = order.get("to") or ([target.get("x"), target.get("y")] if target else None)
        if isinstance(to, (list, tuple)) and len(to) == 2 and all(isinstance(c, int) for c in to) and _near(tuple(to), base):
            out.append(order)  # defending at home
            continue
        out.append({"cmd": "move", "units": units, "to": staging})
    return out


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


def _free_cash_for_core(obs: dict[str, Any]) -> list[dict[str, Any]]:
    """A core item (power, refinery, war factory, harvester) stuck in the queue with no
    money: cancel one queued tank so the cash goes to the economy first. A lost
    refinery once waited 90 s behind three queued tanks."""
    production = _list(obs.get("production"))
    stuck = any(
        str(p.get("item", "")).lower() in (POWER_PLANT, REFINERY, WAR_FACTORY, HARVESTER) and not p.get("ready")
        for p in production
    )
    eco = _dict(obs.get("economy"))
    broke = (eco.get("cash") or 0) + (eco.get("ore") or 0) < 100
    tank = next((str(p.get("item")).lower() for p in production if str(p.get("item", "")).lower() in TANKS), None)
    return [{"cmd": "cancel", "item": tank}] if stuck and broke and tank else []


def _refineries(obs: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in _list(obs.get("buildings")) if str(b.get("type", "")).lower() == REFINERY]


def _defenses(obs: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in _list(obs.get("buildings")) if str(b.get("type", "")).lower() in REFINERY_DEFENSES]


def _defense_spot(obs: dict[str, Any]) -> tuple[int, int] | None:
    """Beside the refinery with the fewest defenses within 6 cells."""
    refineries = [r for r in _refineries(obs) if isinstance(r.get("x"), int) and isinstance(r.get("y"), int)]
    if not refineries:
        return None
    defenses = _defenses(obs)

    def guarded(r: dict[str, Any]) -> int:
        return sum(1 for d in defenses if abs((d.get("x") or 0) - r["x"]) <= 6 and abs((d.get("y") or 0) - r["y"]) <= 6)

    best = min(refineries, key=guarded)
    return best["x"] + 3, best["y"] + 1


def _refinery_defense(obs: dict[str, Any], available: set[str], queued: set[str], building_busy: bool) -> list[dict[str, Any]]:
    """One anti-infantry defense per refinery."""
    if building_busy or len(_defenses(obs)) >= len(_refineries(obs)):
        return []
    item = next((d for d in REFINERY_DEFENSES if d in available), None)
    if not item or any(q in REFINERY_DEFENSES for q in queued):
        return []
    return [{"cmd": "build", "item": item}]


def _steady_infantry(obs: dict[str, Any], available: set[str], queued: set[str]) -> list[dict[str, Any]]:
    """Keep STEADY_INFANTRY anti-infantry around without waiting to see rockets."""
    anti = next((i for i in ANTI_INFANTRY if i in available), None)
    have = sum(1 for t in _types(_list(obs.get("units"))) if t in ANTI_INFANTRY)
    if not anti or have >= STEADY_INFANTRY or any(q in ANTI_INFANTRY for q in queued):
        return []
    return [{"cmd": "train", "item": anti, "count": min(3, STEADY_INFANTRY - have)}]


def _backup_army(state: dict[str, Any], legal: dict[str, Any], available: set[str], queued: set[str]) -> list[dict[str, Any]]:
    """With no model, keep an army coming and use it: the best tank and some
    rocket soldiers, and an attack once enough units are idle."""
    if not economy_ready(state):
        return []
    out = []
    tank = next((t for t in BACKUP_TANKS if t in available), None)
    if tank and tank not in queued:
        out.append({"cmd": "train", "item": tank, "count": 2})
    if "e3" in available and "e3" not in queued:
        out.append({"cmd": "train", "item": "e3", "count": 2})
    attack = legal.get("attack_now")
    if isinstance(attack, dict) and len(legal.get("combat_units") or []) >= BACKUP_ATTACK_AT:
        out.append(attack)
    return out


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
