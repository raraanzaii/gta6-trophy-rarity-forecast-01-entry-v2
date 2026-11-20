"""Forecast Exophase trophy earn rates on the logit scale.

Reads every ``*.in`` case file from the input directory, rebuilds each
trophy's history from the recorded page reads, and writes five quantiles
per trophy to the output directory.  Standard library only, fully
deterministic, and defensive: any surprise degrades to the house
baseline row for the affected trophy instead of crashing.
"""

import json
import math
import os
from datetime import datetime, timezone
from statistics import NormalDist

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
ZED = tuple(NormalDist().inv_cdf(t) for t in LEVELS)


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def clamp_rate(value):
    """Keep a percent inside the range where its logit is finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number < 0.01:
        return 0.01
    if number > 99.99:
        return 99.99
    return number


def to_logit(percent):
    p = clamp_rate(percent)
    if p is None:
        return None
    frac = p / 100.0
    return math.log(frac / (1.0 - frac))


def to_percent(logit_value):
    if logit_value >= 35.0:
        return 100.0
    if logit_value <= -35.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-logit_value))


def parse_stamp(text):
    """Parse an ISO-8601 UTC stamp into an aware datetime, or None."""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def hours_since_epoch(moment):
    return (moment - EPOCH).total_seconds() / 3600.0


def log_time(hours):
    """ln(1 + hours since the epoch); hours may be slightly negative."""
    arg = 1.0 + hours
    if arg < 0.02:
        arg = 0.02
    return math.log(arg)


def median_of(values):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    mid = count // 2
    if count % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def line_slope(xs, ys, ridge_share=0.0):
    """Least-squares slope of ys on xs, optionally ridge-shrunk.

    With ``ridge_share`` r the denominator grows by r * Sxx, so the slope
    is multiplied by 1 / (1 + r); r = 0.18 shrinks it to about 0.85.
    """
    count = len(xs)
    if count < 2:
        return None
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    sxx = 0.0
    sxy = 0.0
    for xv, yv in zip(xs, ys):
        dx = xv - mean_x
        sxx += dx * dx
        sxy += dx * (yv - mean_y)
    if sxx <= 1e-12:
        return None
    return sxy / (sxx * (1.0 + ridge_share) + 1e-12)


def pinball_on_logit(actual_logit, centre_logit, sigma):
    """Mean pinball loss of a five-level forecast on the logit scale."""
    total = 0.0
    for tau, zed in zip(LEVELS, ZED):
        diff = actual_logit - (centre_logit + zed * sigma)
        total += 2.0 * max(tau * diff, (tau - 1.0) * diff)
    return total / len(LEVELS)


# ---------------------------------------------------------------------------
# history handling
# ---------------------------------------------------------------------------

def compress_history(history):
    """Drop consecutive reads that repeat the previous kept read.

    The trophy page often stays frozen between Exophase refreshes, so a
    long run of identical reads carries no extra signal; keeping only
    the first read of each run preserves every distinct state.
    """
    kept = []
    last_key = None
    for read in history:
        if not isinstance(read, dict):
            continue
        moment = parse_stamp(read.get("slot") or read.get("at"))
        if moment is None:
            continue
        rates = read.get("rates")
        if not isinstance(rates, list):
            continue
        players = read.get("players_tracked")
        key = (players, tuple(rates))
        if key == last_key:
            continue
        last_key = key
        hours = hours_since_epoch(moment)
        kept.append({
            "hours": hours,
            "x": log_time(hours),
            "players": players,
            "rates": rates,
        })
    return kept


def trophy_series(reads, index):
    """(x, hours, logit, rate, players) points for one trophy."""
    points = []
    for read in reads:
        rates = read["rates"]
        if index >= len(rates):
            continue
        raw = rates[index]
        if raw is None:
            continue
        logit_value = to_logit(raw)
        if logit_value is None:
            continue
        players = read["players"]
        if not isinstance(players, (int, float)) or players <= 0:
            players = None
        points.append((read["x"], read["hours"], logit_value,
                       clamp_rate(raw), players))
    return points


def players_series(reads):
    points = []
    for read in reads:
        players = read["players"]
        if isinstance(players, (int, float)) and players > 0:
            points.append((read["x"], float(players)))
    return points


# ---------------------------------------------------------------------------
# candidate medians (all on the logit scale)
# ---------------------------------------------------------------------------

def cand_persistence(points, _players, _x_target):
    if not points:
        return None
    return points[-1][2]


def cand_trend(points, _players, x_target):
    """Ridge-shrunk logit trend on log-time over several windows."""
    if len(points) < 3:
        return None
    last_x, last_hours, last_logit = points[-1][0], points[-1][1], points[-1][2]
    slopes = []
    for window in (24.0, 48.0, None):
        if window is None:
            chosen = points
        else:
            chosen = [p for p in points if p[1] >= last_hours - window]
        if len(chosen) < 3:
            continue
        slope = line_slope([p[0] for p in chosen],
                           [p[2] for p in chosen], ridge_share=0.18)
        if slope is not None and math.isfinite(slope):
            slopes.append(slope)
    if not slopes:
        return None
    slope = sum(slopes) / len(slopes)
    shift = slope * (x_target - last_x)
    if shift > 2.5:
        shift = 2.5
    elif shift < -2.5:
        shift = -2.5
    return last_logit + shift


def cand_decomposition(points, players, x_target):
    """Split the rate into earners over players and forecast both parts.

    Both parts are fitted in log space against log-time: player growth
    is multiplicative yet saturates in clock time because the regressor
    is ln(1 + hours), and the earners slope is damped before it is
    extrapolated.  When the player base jumps, the implied rate falls
    even if earners keep growing.
    """
    if len(points) < 3 or len(players) < 3:
        return None
    last_x, _h, last_logit, _r, _p = points[-1]
    slope_p = line_slope([p[0] for p in players],
                         [math.log(p[1]) for p in players])
    if slope_p is None:
        return None
    last_players = players[-1][1]
    growth = math.exp(slope_p * (x_target - players[-1][0]))
    players_ahead = last_players * growth
    if players_ahead < last_players * 0.8:
        players_ahead = last_players * 0.8
    if players_ahead > last_players * 12.0:
        players_ahead = last_players * 12.0
    if players_ahead < 10.0:
        players_ahead = 10.0
    earners = [(p[0], p[3] / 100.0 * p[4]) for p in points if p[4]]
    if len(earners) < 3:
        return None
    slope_e = line_slope([e[0] for e in earners],
                         [math.log(e[1]) for e in earners])
    if slope_e is None:
        return None
    earners_ahead = earners[-1][1] * math.exp(
        0.6 * slope_e * (x_target - earners[-1][0]))
    if earners_ahead < 0.0:
        earners_ahead = 0.0
    if earners_ahead > players_ahead:
        earners_ahead = players_ahead
    implied = to_logit(100.0 * earners_ahead / players_ahead)
    if implied is None:
        return None
    shift = implied - last_logit
    if shift > 1.5:
        shift = 1.5
    elif shift < -1.5:
        shift = -1.5
    return last_logit + shift


def cand_drift(points, _players, x_target):
    """Half of the recent twelve-hour logit drift, capped."""
    if len(points) < 2:
        return None
    last_x, last_hours, last_logit = points[-1][0], points[-1][1], points[-1][2]
    recent = [p for p in points if p[1] >= last_hours - 12.0]
    if len(recent) < 2:
        recent = points[-3:]
    if len(recent) < 2:
        return None
    slope = line_slope([p[0] for p in recent], [p[2] for p in recent])
    if slope is None or not math.isfinite(slope):
        return None
    shift = 0.5 * slope * (x_target - last_x)
    if shift > 1.5:
        shift = 1.5
    elif shift < -1.5:
        shift = -1.5
    return last_logit + shift


CANDIDATES = (cand_persistence, cand_trend, cand_decomposition, cand_drift)


def candidate_medians(points, players, x_target):
    """Median logit from every candidate; None where one cannot speak."""
    answers = []
    for maker in CANDIDATES:
        try:
            value = maker(points, players, x_target)
        except Exception:
            value = None
        if value is not None and not math.isfinite(value):
            value = None
        answers.append(value)
    return answers


# ---------------------------------------------------------------------------
# rolling-origin backtest inside one case
# ---------------------------------------------------------------------------

def backtest_pairs(reads):
    """(origin_index, target_index) pairs spanning the case history."""
    count = len(reads)
    if count < 8:
        return []
    pairs = []
    seen = set()
    target_slots = sorted({count - 1, int(count * 0.8), int(count * 0.6)},
                          reverse=True)
    for j in target_slots:
        if j < 4:
            continue
        span = reads[j]["hours"] - reads[0]["hours"]
        if span <= 0:
            continue
        for share in (0.35, 0.18):
            origin_time = reads[j]["hours"] - share * span
            i = -1
            for k in range(j):
                if reads[k]["hours"] <= origin_time:
                    i = k
            if i < 2 or (i, j) in seen:
                continue
            seen.add((i, j))
            pairs.append((i, j))
    return pairs[:6]


def run_backtest(reads, trophy_count):
    """Score every candidate on the case's own past.

    Returns (global_scores, per_trophy_scores, strategy_scores,
    strategy_per_trophy, residuals) where the strategy is the global
    winner or the 0.65/0.35 blend of the two best candidates.
    """
    pairs = backtest_pairs(reads)
    if not pairs:
        return None
    n_cand = len(CANDIDATES)
    global_sum = [0.0] * n_cand
    global_n = [0] * n_cand
    per_sum = [[0.0] * n_cand for _ in range(trophy_count)]
    per_n = [[0] * n_cand for _ in range(trophy_count)]
    # Raw predictions are kept so the blended strategy can be scored too.
    stored = []  # (pair_pos, trophy, actual_logit, [medians], days_bt)
    for pos, (i, j) in enumerate(pairs):
        head = reads[:i + 1]
        players_head = players_series(head)
        days_bt = max((reads[j]["hours"] - reads[i]["hours"]) / 24.0, 1e-4)
        sigma_bt = 0.5 * math.sqrt(days_bt)
        x_target = reads[j]["x"]
        for t in range(trophy_count):
            actual_raw = reads[j]["rates"][t] if t < len(reads[j]["rates"]) else None
            actual = to_logit(actual_raw) if actual_raw is not None else None
            if actual is None:
                continue
            points = trophy_series(head, t)
            if not points:
                continue
            medians = candidate_medians(points, players_head, x_target)
            stored.append((pos, t, actual, medians, days_bt))
            for c in range(n_cand):
                if medians[c] is None:
                    continue
                loss = pinball_on_logit(actual, medians[c], sigma_bt)
                global_sum[c] += loss
                global_n[c] += 1
                per_sum[t][c] += loss
                per_n[t][c] += 1
    if not stored or global_n[0] == 0:
        return None
    global_scores = []
    for c in range(n_cand):
        if global_n[c] > 0:
            global_scores.append(global_sum[c] / global_n[c])
        else:
            global_scores.append(float("inf"))
    order = sorted(range(n_cand), key=lambda c: (global_scores[c], c))
    best, second = order[0], order[1]
    if global_scores[best] <= 0.90 * global_scores[second]:
        mode = ("single", best, 1.0, best)
    else:
        mode = ("blend", best, 0.65, second)

    def strategy_median(medians):
        kind, first, weight, other = mode
        if kind == "single":
            return medians[first]
        a_val, b_val = medians[first], medians[other]
        if a_val is None:
            return b_val
        if b_val is None:
            return a_val
        return weight * a_val + (1.0 - weight) * b_val

    per_scores = []
    for t in range(trophy_count):
        row = []
        for c in range(n_cand):
            if per_n[t][c] > 0:
                row.append(per_sum[t][c] / per_n[t][c])
            else:
                row.append(float("inf"))
        per_scores.append(row)
    strat_sum = 0.0
    strat_n = 0
    strat_per_sum = [0.0] * trophy_count
    strat_per_n = [0] * trophy_count
    residuals = []
    for pos, t, actual, medians, days_bt in stored:
        i, j = pairs[pos]
        sigma_bt = 0.5 * math.sqrt(days_bt)
        centre = strategy_median(medians)
        if centre is None:
            continue
        loss = pinball_on_logit(actual, centre, sigma_bt)
        strat_sum += loss
        strat_n += 1
        strat_per_sum[t] += loss
        strat_per_n[t] += 1
        residuals.append((actual - centre) / math.sqrt(days_bt))
    strategy_score = strat_sum / strat_n if strat_n else float("inf")
    strat_per = []
    for t in range(trophy_count):
        if strat_per_n[t] > 0:
            strat_per.append(strat_per_sum[t] / strat_per_n[t])
        else:
            strat_per.append(float("inf"))
    return {
        "mode": mode,
        "global": global_scores,
        "per_trophy": per_scores,
        "strategy": strategy_score,
        "strategy_per_trophy": strat_per,
        "residuals": residuals,
    }


# ---------------------------------------------------------------------------
# forecast assembly
# ---------------------------------------------------------------------------

def house_sigma(days):
    return 0.5 * math.sqrt(max(days, 0.0))


def quantile_row(trophy_id, centre_logit, sigma):
    values = []
    for zed in ZED:
        pct = to_percent(centre_logit + zed * sigma)
        values.append(round(min(max(pct, 0.0), 100.0), 4))
    for k in range(1, len(values)):
        if values[k] < values[k - 1]:
            values[k] = values[k - 1]
    return {"id": trophy_id, "q": values}


def last_known_logit(reads, index):
    for read in reversed(reads):
        rates = read["rates"]
        if index < len(rates) and rates[index] is not None:
            value = to_logit(rates[index])
            if value is not None:
                return value
    return None


def forecast_case(case):
    trophies = case.get("trophies") or []
    ids = []
    for pos, entry in enumerate(trophies):
        if isinstance(entry, dict) and isinstance(entry.get("id"), int):
            ids.append(entry["id"])
        else:
            ids.append(pos + 1)
    horizon_minutes = case.get("horizon_minutes")
    if not isinstance(horizon_minutes, (int, float)) or horizon_minutes <= 0:
        horizon_minutes = 90
    days = horizon_minutes / 1440.0
    sigma_floor = house_sigma(days)
    history = case.get("history") or []
    reads = compress_history(history)
    target_moment = parse_stamp(case.get("target"))
    if target_moment is not None:
        x_target = log_time(hours_since_epoch(target_moment))
    elif reads:
        x_target = reads[-1]["x"] + math.log1p(days * 24.0)
    else:
        x_target = log_time(days * 24.0)

    backtest = None
    try:
        if len(reads) >= 8:
            backtest = run_backtest(reads, len(ids))
    except Exception:
        backtest = None

    sigma = sigma_floor
    if backtest is not None and backtest["residuals"]:
        centred = median_of(backtest["residuals"])
        deviations = [abs(r - centred) for r in backtest["residuals"]]
        unit = 1.4826 * median_of(deviations)
        if math.isfinite(unit):
            if unit > 1.25:
                unit = 1.25
            sigma = unit * math.sqrt(days)
            if sigma < sigma_floor:
                sigma = sigma_floor
            if sigma > 3.0:
                sigma = 3.0

    players_all = players_series(reads)
    rows = []
    for t, trophy_id in enumerate(ids):
        try:
            base = last_known_logit(reads, t)
            if base is None:
                base = to_logit(1.0)
            centre = base
            used_sigma = sigma_floor
            if backtest is not None:
                points = trophy_series(reads, t)
                medians = candidate_medians(points, players_all, x_target)
                kind, first, weight, other = backtest["mode"]
                if kind == "single":
                    chosen = medians[first]
                else:
                    a_val, b_val = medians[first], medians[other]
                    if a_val is not None and b_val is not None:
                        chosen = weight * a_val + (1.0 - weight) * b_val
                    elif a_val is not None:
                        chosen = a_val
                    else:
                        chosen = b_val
                # Per-trophy override: a candidate that clearly wins on
                # this trophy alone replaces the global strategy.
                per_row = backtest["per_trophy"][t]
                strat_here = backtest["strategy_per_trophy"][t]
                best_here = min(range(len(CANDIDATES)),
                                key=lambda c: (per_row[c], c))
                if (medians[best_here] is not None
                        and math.isfinite(per_row[best_here])
                        and math.isfinite(strat_here)
                        and per_row[best_here] <= 0.85 * strat_here):
                    chosen = medians[best_here]
                if chosen is not None and math.isfinite(chosen):
                    centre = chosen
                    used_sigma = sigma
            rows.append(quantile_row(trophy_id, centre, used_sigma))
        except Exception:
            try:
                base = last_known_logit(reads, t)
                if base is None:
                    base = to_logit(1.0)
                rows.append(quantile_row(trophy_id, base, sigma_floor))
            except Exception:
                rows.append(quantile_row(trophy_id, to_logit(1.0),
                                         sigma_floor))
    return {"trophies": rows}


def fallback_case(case):
    """House rows for every trophy, used when anything above fails."""
    trophies = case.get("trophies") or []
    horizon_minutes = case.get("horizon_minutes")
    if not isinstance(horizon_minutes, (int, float)) or horizon_minutes <= 0:
        horizon_minutes = 90
    sigma = house_sigma(horizon_minutes / 1440.0)
    history = case.get("history") or []
    rows = []
    for pos, entry in enumerate(trophies):
        trophy_id = pos + 1
        if isinstance(entry, dict) and isinstance(entry.get("id"), int):
            trophy_id = entry["id"]
        rate = None
        for read in reversed(history):
            if not isinstance(read, dict):
                continue
            rates = read.get("rates")
            if isinstance(rates, list) and pos < len(rates) and rates[pos] is not None:
                rate = rates[pos]
                break
        centre = to_logit(rate) if rate is not None else None
        if centre is None:
            centre = to_logit(1.0)
        rows.append(quantile_row(trophy_id, centre, sigma))
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError:
        names = []
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        pass
    for name in names:
        case_id = name[:-3]
        try:
            with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
                case = json.load(fh)
            if isinstance(case, dict) and case.get("id"):
                case_id = str(case["id"])
        except Exception:
            case = None
        result = None
        if isinstance(case, dict):
            try:
                result = forecast_case(case)
            except Exception:
                result = None
            if result is None:
                try:
                    result = fallback_case(case)
                except Exception:
                    result = None
        if result is None:
            result = {"trophies": []}
        try:
            text = json.dumps(result, separators=(",", ":"))
            with open(os.path.join(out_dir, case_id + ".out"), "w",
                      encoding="utf-8") as fh:
                fh.write(text)
        except OSError:
            pass


if __name__ == "__main__":
    main()
