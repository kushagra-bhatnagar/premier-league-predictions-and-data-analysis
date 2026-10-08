"""Shared feature engineering for the EPL outcome model."""
import numpy as np
import pandas as pd

CLASSES = ["A", "D", "H"]            # fixed order -> class index 0, 1, 2
STAT_COLS = ["gf", "ga", "sh", "sot", "opp_sh", "opp_sot", "corners", "fouls", "yellow", "red"]
WINDOWS = (5, 10)
COMPACT = ["elo_diff", "d_sot_r10", "d_opp_sot_r10", "d_gd_r10"]   # final model inputs
HOME_ADV = 60


class Elo:
    def __init__(self, k=20, home_adv=HOME_ADV, base=1500, season_regress=0.25):
        self.k, self.ha, self.base, self.reg = k, home_adv, base, season_regress
        self.r = {}

    def get(self, t):
        return self.r.get(t, self.base)

    def new_season(self):
        for t in self.r:
            self.r[t] = self.r[t] * (1 - self.reg) + self.base * self.reg

    def expected_home(self, h, a):
        return 1 / (1 + 10 ** (-(self.get(h) + self.ha - self.get(a)) / 400))

    def update(self, h, a, gh, ga):
        e = self.expected_home(h, a)
        s = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
        gd = abs(gh - ga)
        mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        d = self.k * mult * (s - e)
        self.r[h] = self.get(h) + d
        self.r[a] = self.get(a) - d


def add_team_features(df):
    """Per-team rolling features on the long (one row per team per match) table.

    Everything is shifted by one match, so a row only sees matches BEFORE it.
    Returns (long table, list of per-team feature columns).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "team"]).reset_index(drop=True)
    df["pts"] = df["result"].map({"W": 3, "D": 1, "L": 0})
    df["gd"] = df["gf"] - df["ga"]
    df["sot_ratio"] = df["sot"] / df["sh"].clip(lower=1)
    df["opp_sot_ratio"] = df["opp_sot"] / df["opp_sh"].clip(lower=1)
    base_stats = ["pts", "gd"] + STAT_COLS + ["sot_ratio", "opp_sot_ratio"]

    g = df.groupby("team", sort=False)
    for w in WINDOWS:
        for c in base_stats:
            df[f"{c}_r{w}"] = g[c].transform(lambda s: s.shift(1).rolling(w, min_periods=3).mean())
    gv = df.groupby(["team", "venue"], sort=False)
    for c in ["pts", "gd", "gf", "ga"]:
        df[f"{c}_venue5"] = gv[c].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    df["rest"] = g["date"].transform(lambda s: s.diff().dt.days).clip(upper=14)
    df["n_prev"] = g.cumcount().clip(upper=38)
    feat_cols = [c for c in df.columns if "_r5" in c or "_r10" in c or "_venue5" in c] + ["rest", "n_prev"]
    return df, feat_cols


def add_match_features(m):
    """Elo-derived and home-minus-away difference features (needs elo_h / elo_a and *_h / *_a columns)."""
    m["elo_diff"] = m.elo_h - m.elo_a
    m["elo_exp"] = 1 / (1 + 10 ** (-(m.elo_diff + HOME_ADV) / 400))
    for c in ["pts", "gd", "gf", "ga", "sot", "opp_sot", "sh", "opp_sh", "corners"]:
        for w in WINDOWS:
            m[f"d_{c}_r{w}"] = m[f"{c}_r{w}_h"] - m[f"{c}_r{w}_a"]
    m["d_pts_venue5"] = m["pts_venue5_h"] - m["pts_venue5_a"]
    m["d_gd_venue5"] = m["gd_venue5_h"] - m["gd_venue5_a"]
    m["d_rest"] = m["rest_h"] - m["rest_a"]
    return m


def feature_columns(m):
    names = [c for c in m.columns if c.endswith(("_h", "_a")) and c not in ("elo_h", "elo_a")] \
            + ["elo_h", "elo_a", "elo_diff", "elo_exp"] + [c for c in m.columns if c.startswith("d_")]
    return [c for c in dict.fromkeys(names) if c not in ("gf", "ga", "result")]


def build_features(df):
    """Training table: one row per historical match.

    Returns (match table, feature names, long table, final Elo object).
    """
    df, feat_cols = add_team_features(df)
    home = df[df.venue == "Home"]
    away = df[df.venue == "Away"].rename(columns={"team": "opponent", "opponent": "team"})
    key = ["date", "team", "opponent"]
    m = home[key + ["season", "gf", "ga", "result"] + feat_cols].merge(
        away[key + feat_cols], on=key, suffixes=("_h", "_a"))
    m = m.rename(columns={"team": "home", "opponent": "away"}).sort_values("date").reset_index(drop=True)
    m["y"] = m["result"].map({"W": "H", "D": "D", "L": "A"})

    elo, last_season, eh, ea = Elo(), None, [], []
    for r in m.itertuples():
        if last_season is not None and r.season != last_season:
            elo.new_season()
        last_season = r.season
        eh.append(elo.get(r.home)); ea.append(elo.get(r.away))      # PRE-match ratings
        elo.update(r.home, r.away, r.gf, r.ga)
    m["elo_h"], m["elo_a"] = eh, ea
    m = add_match_features(m)
    return m, feature_columns(m), df, elo


def fixture_features(hist, fixtures, elo_ratings, season):
    """Features for UPCOMING fixtures, computed in bulk.

    hist      : long match history (one row per team per match)
    fixtures  : list of (home, away) or (home, away, date); default date = a week after the last match
    elo_ratings: dict team -> Elo after the last recorded match
    """
    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    default = hist.date.max() + pd.Timedelta(days=7)
    fx = pd.DataFrame([(f[0], f[1], pd.Timestamp(f[2]) if len(f) > 2 and f[2] else default) for f in fixtures],
                      columns=["home", "away", "date"])
    known = set(hist.team)
    for t in set(fx.home) | set(fx.away):
        if t not in known:
            raise ValueError(f"Unknown team '{t}'. Known teams: {sorted(known)}")
    if (fx.date <= hist.date.max()).any():
        raise ValueError("Fixture dates must be after the last match in the history.")

    def placeholder(teams, venue, date):      # stats are dummies: features only look at earlier matches
        return pd.DataFrame([dict(date=date, season=season, team=t, opponent="?", venue=venue, gf=0, ga=0, sh=1,
                                  sot=0, opp_sh=1, opp_sot=0, fouls=0, corners=0, yellow=0, red=0, result="D")
                             for t in teams])

    parts = []
    for d, sub in fx.groupby("date"):
        feats = {}
        for venue, teams in [("Home", sub.home.unique()), ("Away", sub.away.unique())]:
            long_df, cols = add_team_features(pd.concat([hist, placeholder(teams, venue, d)], ignore_index=True))
            feats[venue] = long_df[(long_df.date == d) & (long_df.venue == venue)].set_index("team")[cols]
        rows = []
        for r in sub.itertuples():
            row = {"date": d, "home": r.home, "away": r.away}
            row.update({f"{c}_h": v for c, v in feats["Home"].loc[r.home].items()})
            row.update({f"{c}_a": v for c, v in feats["Away"].loc[r.away].items()})
            row["elo_h"], row["elo_a"] = elo_ratings.get(r.home, 1500), elo_ratings.get(r.away, 1500)
            rows.append(row)
        parts.append(pd.DataFrame(rows))
    m = add_match_features(pd.concat(parts, ignore_index=True))
    return m
