"""Hand-curated Round 1 inputs, each with its source.

Everything here was read from a source on 23 Sep 2026, the day before tip-off.
The official EuroLeague injury report outranks everything else when they
disagree (e.g. Milutinov: Basketball Sphere "doubtful", official "available").
"""

# --- Bookmaker moneylines, decimal, OddsPortal average (23 Sep 2026) --------
# (home, away): (home_price, away_price)
R1_MONEYLINES = {
    ("HTA", "MUN"): (1.30, 3.54),
    ("DUB", "MAD"): (1.66, 2.21),
    ("RED", "ZAL"): (1.67, 2.20),
    ("PAN", "PRS"): (1.15, 5.39),
    ("BAR", "IST"): (1.69, 2.16),
    ("BAS", "OLY"): (4.12, 1.23),
    ("ASV", "TEL"): (2.46, 1.54),
    ("BES", "PAM"): (1.70, 2.13),
    ("ULK", "VIR"): (1.15, 5.37),
    ("PAR", "MIL"): (1.62, 2.27),
}

# Outright winner odds, decimal, OddsPortal average.
OUTRIGHTS = {
    "OLY": 4.06,
    "PAN": 4.22,
    "ULK": 7.66,
    "MAD": 8.07,
    "HTA": 14.43,
    "DUB": 15.14,
    "BAR": 18.29,
    "ZAL": 22.29,
    "IST": 29.57,
    "RED": 29.57,
    "TEL": 33.86,
    "PAM": 33.86,
    "PAR": 35.00,
    "MIL": 35.00,
    "BES": 42.00,
    "ASV": 47.43,
    "BAS": 86.00,
    "PRS": 94.00,
    "MUN": 94.00,
    "VIR": 94.00,
}

# --- Availability: P(plays) by round --------------------------------------
# Sources: official "Injury report: Round 1" (euroleaguebasketball.net, 23 Sep)
# and Basketball Sphere injury tracker (23 Sep). (r1, r2, r3)
AVAILABILITY = {
    # confirmed out for R1 by the clubs
    ("PAN", "Kendrick Nunn"): (0.0, 0.0, 0.0),  # knee surgery 19 Aug
    ("PAN", "Kostas Sloukas"): (0.0, 0.25, 0.35),  # RotoWire 22 Sep: "15-to-20 days"
    ("PAN", "Nigel Hayes-Davis"): (0.0, 0.0, 0.0),  # 6 weeks from 1 Sep
    ("PAN", "Moustapha Fall"): (0.0, 0.3, 0.5),  # hamstring, 9 Sep
    ("BAR", "Tosan Evbuomwan"): (0.0, 0.3, 0.5),  # shoulder, 4 Sep
    ("BAR", "Yoan Makoundou"): (0.0, 0.0, 0.0),  # ~6 weeks from 10 Sep
    ("IST", "Isaia Cordinier"): (0.0, 0.3, 0.5),
    ("IST", "Georgios Papagiannis"): (0.0, 0.0, 0.0),  # ACL
    ("RED", "Nikola Djurisic"): (0.0, 0.5, 0.75),  # 3-4 weeks from 27 Aug
    ("ZAL", "Deividas Sirvydis"): (0.0, 0.6, 0.8),  # did not travel
    ("ZAL", "Kaodirichi Akobundu-Ehiogu"): (0.0, 0.6, 0.8),
    ("HTA", "Tomer Ginat"): (0.0, 0.5, 0.7),
    ("HTA", "Tyler Ennis"): (0.0, 0.0, 0.0),  # Achilles
    ("MAD", "Usman Garuba"): (0.0, 0.0, 0.0),  # Achilles
    ("BES", "Mady Sissoko"): (0.0, 0.0, 0.0),  # contract frozen
    ("ASV", "Mathis Dossou-Yovo"): (0.0, 0.0, 0.0),  # back in November
    # genuinely uncertain
    ("ZAL", "Saben Lee"): (0.3, 0.6, 0.8),  # official GTD; RotoWire: out "a few weeks"
    ("BAS", "Alex Len"): (0.05, 0.6, 0.8),  # RotoWire 23 Sep: unavailable Thursday
    ("ULK", "Shane Larkin"): (0.5, 0.75, 0.85),  # missed both Super Cups
    ("DUB", "Dzanan Musa"): (0.1, 0.5, 0.7),  # RotoWire 22 Sep: out, knee
    ("DUB", "Dwayne Bacon"): (0.6, 0.85, 0.9),  # head injury in SuperCup
    ("PAM", "Neal Sako"): (0.4, 0.6, 0.8),  # groin, 22 Sep
    # cleared
    ("OLY", "Nikola Milutinov"): (
        0.85,
        0.95,
        0.95,
    ),  # official: available; RotoWire: leg, travelling
    ("TEL", "Jimmy Clark"): (0.9, 0.95, 0.95),
    ("TEL", "Daniel Theis"): (0.92, 0.95, 0.95),
    ("RED", "Semi Ojeleye"): (0.92, 0.95, 0.95),
    ("PAM", "Jasiel Rivero"): (0.8, 0.9, 0.95),  # RotoWire 22 Sep: minor muscular, limited
}

# Minutes restriction on return: Milutinov skipped the SuperCup.
MINUTES_FACTOR_R1 = {("OLY", "Nikola Milutinov"): 0.85, ("PAM", "Jasiel Rivero"): 0.85}

# --- Role evidence from preseason / expert previews ------------------------
# Multipliers on the statistical baseline, used only where a named source gives
# concrete role evidence. Kept deliberately moderate: preseason is noisy.
# Sources: euroleaguebasketball.net Round 1 tips (guards/forwards/centers,
# 21 Sep), Basketball Sphere Round 1 tips, breakout and smaller-role previews.
ROLE = {
    # bigger role than price implies
    ("PAM", "TJ Shorts"): (1.20, "leader of new team; 29 PIR in Super Cup"),
    ("PAN", "Sylvain Francisco"): (1.08, "~30 min expected with Nunn out"),
    ("BAR", "Joel Parra"): (1.30, "double-digit PIR through preseason; frontcourt depleted"),
    ("PAN", "Eleftherios Mantzoukas"): (1.40, "won real minutes in preseason"),
    ("BAS", "Stefan Joksimovic"): (1.35, "set to earn minutes immediately"),
    ("BAS", "Matteo Spagnolo"): (1.25, "~22 min, 10 pts 8 reb first official game"),
    ("PAM", "Mario Saint-Supery"): (1.30, "~18 min, 10 PIR first official game"),
    ("BAR", "Juan Nunez"): (1.20, "~15 min with thin PG depth"),
    ("ZAL", "Nigel Williams-Goss"): (1.15, "more on-ball work with Francisco gone"),
    ("PRS", "Allan Dokossi"): (1.15, "24 PIR in 22 min first official game"),
    ("PAR", "Kevarrius Hayes"): (1.10, "~20 min expected"),
    ("PAR", "Lamar Stevens"): (1.10, "~25 min in thin forward rotation"),
    ("PAR", "Arijan Lakic"): (1.20, "10-15 min in tight rotation"),
    ("BES", "Eugene Omoruyi"): (1.10, "starting PF, focal point"),
    ("MUN", "Duane Washington"): (1.10, "primary scoring role"),
    ("TEL", "Yam Madar"): (1.10, "designated backcourt leader"),
    ("ZAL", "Azuolas Tubelis"): (1.10, "inherits departed scorers' touches"),
    # smaller role than price implies
    ("DUB", "Tornike Shengelia"): (0.88, "crowded Dubai frontcourt"),
    ("ULK", "Trent Forrest"): (0.75, "~15 min in deep backcourt"),
    ("DUB", "Elie Okobo"): (0.88, "fierce backcourt competition"),
    ("OLY", "Codi Miller-McIntyre"): (0.88, "facilitator behind Montero"),
    ("ULK", "Shane Larkin"): (0.90, "deep backcourt (on top of injury doubt)"),
}

# --- Expected minutes where a source states them concretely ------------------
# Used for players with no EuroLeague history, where the price-based fallback
# cannot see a preseason role. Production = minutes x positional PIR/min x 0.9
# (a rookie/role-player discount).
MINUTES_EXPERT = {
    (
        "PAN",
        "Eleftherios Mantzoukas",
    ): 14.0,  # official tips: took the minutes expected for Mitoglou
    ("PAR", "Arijan Lakic"): 12.0,  # Basketball Sphere: 10-15 min
    ("PAM", "Mario Saint-Supery"): 17.0,  # official tips: 18 min first official game
    ("BAR", "Juan Nunez"): 14.0,  # Basketball Sphere: ~15 min
    ("BAS", "Stefan Joksimovic"): 13.0,  # both sources: immediate minutes
}
