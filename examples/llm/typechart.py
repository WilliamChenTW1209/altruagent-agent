"""Gen 9 type effectiveness, so the model reads matchups instead of computing them.

Only type matchups: abilities (Levitate, Water Absorb, …), items and
field effects aren't counted, and the prompt says so.
"""

from __future__ import annotations

# attacking type -> {defending type: multiplier}; anything missing is 1x.
_CHART: dict[str, dict[str, float]] = {
    "NORMAL": {"ROCK": 0.5, "GHOST": 0, "STEEL": 0.5},
    "FIRE": {"FIRE": 0.5, "WATER": 0.5, "GRASS": 2, "ICE": 2, "BUG": 2, "ROCK": 0.5, "DRAGON": 0.5, "STEEL": 2},
    "WATER": {"FIRE": 2, "WATER": 0.5, "GRASS": 0.5, "GROUND": 2, "ROCK": 2, "DRAGON": 0.5},
    "ELECTRIC": {"WATER": 2, "ELECTRIC": 0.5, "GRASS": 0.5, "GROUND": 0, "FLYING": 2, "DRAGON": 0.5},
    "GRASS": {"FIRE": 0.5, "WATER": 2, "GRASS": 0.5, "POISON": 0.5, "GROUND": 2, "FLYING": 0.5, "BUG": 0.5,
              "ROCK": 2, "DRAGON": 0.5, "STEEL": 0.5},
    "ICE": {"FIRE": 0.5, "WATER": 0.5, "GRASS": 2, "ICE": 0.5, "GROUND": 2, "FLYING": 2, "DRAGON": 2, "STEEL": 0.5},
    "FIGHTING": {"NORMAL": 2, "ICE": 2, "POISON": 0.5, "FLYING": 0.5, "PSYCHIC": 0.5, "BUG": 0.5, "ROCK": 2,
                 "GHOST": 0, "DARK": 2, "STEEL": 2, "FAIRY": 0.5},
    "POISON": {"GRASS": 2, "POISON": 0.5, "GROUND": 0.5, "ROCK": 0.5, "GHOST": 0.5, "STEEL": 0, "FAIRY": 2},
    "GROUND": {"FIRE": 2, "ELECTRIC": 2, "GRASS": 0.5, "POISON": 2, "FLYING": 0, "BUG": 0.5, "ROCK": 2, "STEEL": 2},
    "FLYING": {"ELECTRIC": 0.5, "GRASS": 2, "FIGHTING": 2, "BUG": 2, "ROCK": 0.5, "STEEL": 0.5},
    "PSYCHIC": {"FIGHTING": 2, "POISON": 2, "PSYCHIC": 0.5, "DARK": 0, "STEEL": 0.5},
    "BUG": {"FIRE": 0.5, "GRASS": 2, "FIGHTING": 0.5, "POISON": 0.5, "FLYING": 0.5, "PSYCHIC": 2, "GHOST": 0.5,
            "DARK": 2, "STEEL": 0.5, "FAIRY": 0.5},
    "ROCK": {"FIRE": 2, "ICE": 2, "FIGHTING": 0.5, "GROUND": 0.5, "FLYING": 2, "BUG": 2, "STEEL": 0.5},
    "GHOST": {"NORMAL": 0, "PSYCHIC": 2, "GHOST": 2, "DARK": 0.5},
    "DRAGON": {"DRAGON": 2, "STEEL": 0.5, "FAIRY": 0},
    "DARK": {"FIGHTING": 0.5, "PSYCHIC": 2, "GHOST": 2, "DARK": 0.5, "FAIRY": 0.5},
    "STEEL": {"FIRE": 0.5, "WATER": 0.5, "ELECTRIC": 0.5, "ICE": 2, "ROCK": 2, "STEEL": 0.5, "FAIRY": 2},
    "FAIRY": {"FIRE": 0.5, "FIGHTING": 2, "POISON": 0.5, "DRAGON": 2, "DARK": 2, "STEEL": 0.5},
}


def effectiveness(move_type: str, defender_types: list[str]) -> float | None:
    """The type multiplier (0, 0.25 … 4), or None if a type is unknown."""
    attack = str(move_type).upper()
    defenders = [str(t).upper() for t in defender_types if t]
    if attack not in _CHART or not defenders or any(t not in _CHART for t in defenders):
        return None
    multiplier = 1.0
    for defender in defenders:
        multiplier *= _CHART[attack].get(defender, 1)
    return multiplier


def describe(multiplier: float) -> str:
    if multiplier == 0:
        return "0x (no effect)"
    text = f"{multiplier:g}x"
    if multiplier > 1:
        return f"{text} (super effective)"
    if multiplier < 1:
        return f"{text} (not very effective)"
    return text
