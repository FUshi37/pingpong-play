"""Raw mocap XYZ acceptance bounds, in meters; no receiver or timer work."""
from __future__ import annotations

import math


def normalize_observation_bounds(bounds):
    """Three inclusive axis pairs; None on an axis leaves it unrestricted."""
    if bounds is None:
        return (None, None, None)
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 3:
        raise ValueError('Mocap observation range requires X/Y/Z limits')
    result = []
    for axis, pair in zip('XYZ', bounds):
        if pair is None:
            result.append(None)
            continue
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f'Mocap observation {axis} limits require MIN,MAX')
        try:
            low, high = map(float, pair)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'Mocap observation {axis} limits must be finite and increasing') from exc
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError(f'Mocap observation {axis} limits must be finite and increasing')
        result.append((low, high))
    return tuple(result)


def parse_observation_bounds(values):
    """Absent new options retain legacy filters; explicit blanks remove them."""
    if all(value is None for value in values):
        return None
    pairs = [None if value is None or not str(value).strip()
             else str(value).split(',') for value in values]
    return normalize_observation_bounds(pairs)
