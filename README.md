# gta6-trophy-rarity-forecast-01-entry-v2

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/raraanzaii/gta6-trophy-rarity-forecast-01-entry-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/raraanzaii/gta6-trophy-rarity-forecast-01-entry-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A logit-scale forecaster for GTA VI trophy earn rates on Exophase. All
arithmetic happens on the logit of the clamped rate, because that is the
scale the pinball loss is charged on. The recorded history is first
compressed by dropping consecutive identical reads, since the page
stays frozen between Exophase refreshes and repeated copies of the same
state add nothing. For every trophy the program builds four candidate
medians: plain persistence of the last observed rate; a ridge-shrunk
linear trend of the logit against ln(1 + hours since 2026-11-19),
fitted over the last 24 hours, the last 48 hours and the full history
and clamped to a ±2.5 logit move; a decomposition that forecasts the
tracked-player count and the earner count separately in log space
(earners damped by 0.6) and divides them, clamped to ±1.5 logit, which
is what lets the forecast fall when a refresh floods the page with new
players; and half of the last twelve hours of logit drift, also
clamped to ±1.5 logit. The candidates are then judged by a
rolling-origin backtest run inside the case's own history: the winner
is used outright when it beats the runner-up by at least ten percent,
otherwise the two best are blended 0.65/0.35 in logit space, and a
candidate that wins a single trophy by fifteen percent or more
overrides the global choice for that trophy. The spread comes from the
median absolute deviation of the standardised backtest residuals,
scaled by the square root of the horizon in days, and it is never
allowed to be narrower than the house spread of 0.5 × √(days). Any
anomaly — too little data, a missing rate, an exception — falls back
per trophy to the house baseline row, so the program always writes a
complete, valid forecast file.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=kagami-tanaka&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
