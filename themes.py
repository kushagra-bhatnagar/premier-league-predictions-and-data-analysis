"""Shared design system for the Premier League Wiz posts.

Everything here is fixed except the accent colour, which follows the team
being analysed (see TEAM_ACCENT / accent_for).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle

# ------------------------------------------------------------------
# FIXED THEME
# ------------------------------------------------------------------
THEME = {
    "bg": "#141414",          # Matte charcoal
    "border": "#2A2A2E",      # Subtle divider line
    "accent": "#D10022",      # Default crimson (used when no team colour is given)
    "white": "#F5F5F5",
    "muted": "#8E8E93",
    "zebra": "#1A1A1A",
    "row_text": "#CFCFCF",
    "neutral": "#3A3A3F",     # Neutral fill on charcoal (draws, ties)
    "pitch_fill": "#181818",
    "pitch_line": "#4A4A50",
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Inter", "Graphik", "Helvetica Neue", "Arial", "DejaVu Sans"
]

PITCH_L, PITCH_W = 105.0, 68.0

# ------------------------------------------------------------------
# TEAM ACCENTS (tuned to read on charcoal; dark kits use a lighter tint)
# ------------------------------------------------------------------
TEAM_ACCENT = {
    "Arsenal": "#EF0107",
    "Aston Villa": "#95BFE5",
    "Bournemouth": "#DA291C",
    "Brentford": "#E30613",
    "Brighton": "#3D8BFD",
    "Burnley": "#99D6EA",
    "Chelsea": "#2E6FD8",
    "Coventry": "#5CB8E6",
    "Crystal Palace": "#C4122E",
    "Everton": "#3B6FD4",
    "Fulham": "#CC0000",
    "Hull": "#F5A21A",
    "Ipswich": "#3A64D8",
    "Leeds": "#FFCD00",
    "Leicester": "#4F8FE0",
    "Liverpool": "#C8102E",
    "Luton": "#F78F1E",
    "Manchester City": "#6CABDD",
    "Manchester United": "#DA291C",
    "Newcastle United": "#41B6E6",
    "Nottingham Forest": "#E53233",
    "Sheffield United": "#EE2737",
    "Southampton": "#D71920",
    "Sunderland": "#EB172B",
    "Tottenham": "#8FA8E8",
    "West Ham": "#8DC3EB",
    "Wolverhampton Wanderers": "#FDB913",
}


def accent_for(team):
    return TEAM_ACCENT.get(team, THEME["accent"])


# ------------------------------------------------------------------
# COLOUR HELPERS
# ------------------------------------------------------------------
def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(fg, bg, a):
    """a * fg + (1 - a) * bg, as a hex colour."""
    return "#%02X%02X%02X" % tuple(
        round(a * x + (1 - a) * y) for x, y in zip(_rgb(fg), _rgb(bg))
    )


def text_on(color):
    """Charcoal text on light fills, white text on dark fills."""
    r, g, b = _rgb(color)
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return THEME["bg"] if lum > 0.6 else THEME["white"]


# ------------------------------------------------------------------
# CHROME (canvas, header deck, dividers, footer)
# ------------------------------------------------------------------
def new_canvas():
    """4:5 portrait canvas; `ax` is a 0-100 layout space covering the figure."""
    fig = plt.figure(figsize=(8, 10), dpi=300)
    fig.patch.set_facecolor(THEME["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(THEME["bg"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def header(fig, kicker, title, subtitle, accent):
    t = THEME
    fig.text(0.10, 0.935, kicker, fontsize=9, fontweight="bold",
             color=accent, ha="left")
    fig.text(0.10, 0.895, title, fontsize=23, fontweight="heavy",
             color=t["white"], ha="left")
    fig.text(0.10, 0.860, subtitle, fontsize=10.5, color=t["muted"], ha="left")


def divider(ax, y, lw=1.0):
    ax.plot([10, 90], [y, y], color=THEME["border"], lw=lw)


def footer(fig, ax, source, brand="Premier League Wiz"):
    t = THEME
    divider(ax, 10.5)
    fig.text(0.10, 0.05, source, fontsize=8, color=t["muted"],
             ha="left", fontweight="semibold")
    fig.text(0.90, 0.05, brand, fontsize=9, color=t["white"],
             ha="right", fontweight="black")


# ------------------------------------------------------------------
# PITCH (vertical, attacking up; y_min trims the pitch from the bottom)
# ------------------------------------------------------------------
def draw_pitch(ax, y_min=PITCH_L / 2):
    t = THEME
    c, lw = t["pitch_line"], 1.3
    cx = PITCH_W / 2

    ax.add_patch(Rectangle((0, y_min), PITCH_W, PITCH_L - y_min,
                           fc=t["pitch_fill"], ec="none", zorder=0))

    # Touchlines + goal line
    ax.plot([0, 0, PITCH_W, PITCH_W], [y_min, PITCH_L, PITCH_L, y_min],
            color=c, lw=lw, zorder=1)

    # Halfway line and centre circle only if they are in view
    if y_min <= PITCH_L / 2:
        ax.plot([0, PITCH_W], [PITCH_L / 2] * 2, color=c, lw=lw, zorder=1)
        ax.add_patch(Arc((cx, PITCH_L / 2), 18.3, 18.3, theta1=0, theta2=180,
                         ec=c, lw=lw, zorder=1))

    # Penalty area, six-yard box, goal frame
    ax.add_patch(Rectangle((cx - 20.16, PITCH_L - 16.5), 40.32, 16.5,
                           fill=False, ec=c, lw=lw, zorder=1))
    ax.add_patch(Rectangle((cx - 9.16, PITCH_L - 5.5), 18.32, 5.5,
                           fill=False, ec=c, lw=lw, zorder=1))
    ax.add_patch(Rectangle((cx - 3.66, PITCH_L), 7.32, 2.0,
                           fill=False, ec=c, lw=lw, zorder=1))

    # Penalty spot, the "D", corner arcs
    ax.add_patch(Circle((cx, PITCH_L - 11), 0.35, color=c, zorder=1))
    ax.add_patch(Arc((cx, PITCH_L - 11), 18.3, 18.3, theta1=216.87, theta2=323.13,
                     ec=c, lw=lw, zorder=1))
    ax.add_patch(Arc((0, PITCH_L), 2, 2, theta1=270, theta2=360,
                     ec=c, lw=lw, zorder=1))
    ax.add_patch(Arc((PITCH_W, PITCH_L), 2, 2, theta1=180, theta2=270,
                     ec=c, lw=lw, zorder=1))

    ax.set_xlim(-2, PITCH_W + 2)
    ax.set_ylim(y_min - (2.5 if y_min <= PITCH_L / 2 else 1.0), PITCH_L + 3)
    ax.set_aspect("equal")
    ax.axis("off")