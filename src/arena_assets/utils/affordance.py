"""Seat heights and seat placement read off a triangle mesh in the built model frame."""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from arena_simulation_setup.utils.affordance import Footprint

SEAT_BAND = (0.20, 0.90)
SEAT_BIN = 0.02
SEAT_WINDOW = 0.05
SEAT_GRID = 0.02
SEAT_FALLBACK = 0.45
SEAT_SIZE = 0.45


def height_map(points: np.ndarray, triangles: np.ndarray, footprint: Footprint) -> tuple[np.ndarray, np.ndarray]:
    """Cell centers on a 2 cm grid over a ((x_min, x_max), (y_min, y_max)) footprint and the top surface height at each, NaN where empty."""
    (x_min, x_max), (y_min, y_max) = footprint
    xs = np.arange(x_min + SEAT_GRID / 2, x_max, SEAT_GRID)
    ys = np.arange(y_min + SEAT_GRID / 2, y_max, SEAT_GRID)
    cells = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)
    top = np.full((len(xs), len(ys)), np.nan)
    if len(triangles) == 0 or top.size == 0:
        return cells, top
    corners = np.asarray(points, np.float64)[np.asarray(triangles, np.int64)]
    lows, highs = corners.min(axis=1), corners.max(axis=1)
    near = (highs[:, 0] >= x_min) & (lows[:, 0] <= x_max) & (highs[:, 1] >= y_min) & (lows[:, 1] <= y_max)
    for a, b, c in corners[near]:
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-12:
            continue
        lo = np.minimum(np.minimum(a, b), c)
        hi = np.maximum(np.maximum(a, b), c)
        i0, i1 = np.searchsorted(xs, lo[0]), np.searchsorted(xs, hi[0], side="right")
        j0, j1 = np.searchsorted(ys, lo[1]), np.searchsorted(ys, hi[1], side="right")
        if i0 >= i1 or j0 >= j1:
            continue
        px, py = cells[i0:i1, j0:j1, 0], cells[i0:i1, j0:j1, 1]
        wa = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / det
        wb = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / det
        wc = 1.0 - wa - wb
        inside = (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9)
        z = np.where(inside, wa * a[2] + wb * b[2] + wc * c[2], np.nan)
        block = top[i0:i1, j0:j1]
        top[i0:i1, j0:j1] = np.fmax(block, z)
    return cells, top


def _seat(top: np.ndarray) -> tuple[float, np.ndarray] | None:
    """Seat height: the 2 cm bin of top heights in the seat band covering the most cells, the median of the cells within 5 cm of it."""
    in_band = (top >= SEAT_BAND[0]) & (top <= SEAT_BAND[1])
    if not in_band.any():
        return None
    bins = np.floor((top[in_band] - SEAT_BAND[0]) / SEAT_BIN).astype(np.int64)
    dominant = SEAT_BAND[0] + (int(np.argmax(np.bincount(bins))) + 0.5) * SEAT_BIN
    near = in_band & (np.abs(np.nan_to_num(top, nan=-1.0) - dominant) <= SEAT_WINDOW)
    return float(np.median(top[near])), near


def surface_z(points: np.ndarray, triangles: np.ndarray, footprint: Footprint) -> float | None:
    """Seat height over an object-frame footprint, None when no seat surface lies under it."""
    _, top = height_map(points, triangles, footprint)
    seat = _seat(top)
    return None if seat is None else round(seat[0], 3)


def largest_seat(points: np.ndarray, triangles: np.ndarray) -> tuple[float, float, float] | None:
    """Center cell and height of the seat surface covering the most of the top view, None without one."""
    if len(triangles) == 0:
        return None
    lo, hi = np.asarray(points).min(axis=0), np.asarray(points).max(axis=0)
    cells, top = height_map(points, triangles, ((float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))))
    seat = _seat(top)
    if seat is None:
        return None
    z, near = seat
    seats = cells[near]
    x, y = seats[np.argmin(np.linalg.norm(seats - seats.mean(axis=0), axis=1))]
    return (round(float(x), 3), round(float(y), 3), round(z, 3))
