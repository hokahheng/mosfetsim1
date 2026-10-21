"""3D device drawings (matplotlib mplot3d). One function per device; register it
in DRAWERS. Dimensions in nm. Oxide / gate thicknesses are exaggerated for
visibility. Axes: x = source->drain, y = width, z = up."""
import numpy as np
import mpl_toolkits.mplot3d  # noqa: F401  (registers the 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch

COL = {"substrate": "#b8b8b8", "box": "#dcdcdc", "sd": "#2f8f5b", "body": "#d9a5a5",
       "fin": "#d9534f", "cap": "#6c757d", "oxide": "#7fb5ff", "gate": "#c78a2c"}
_FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]


def add_box(ax, x0, x1, y0, y1, z0, z1, color, alpha=0.9):
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    ax.add_collection3d(Poly3DCollection([v[f] for f in _FACES], facecolors=color,
                                         edgecolors="k", linewidths=0.4, alpha=alpha))


def finish_axes(ax, xr, yr, zr, title, legend):
    ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_zlim(*zr)
    spans = [xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]]
    big = max(spans)
    ax.set_box_aspect(tuple(max(s, 0.25 * big) for s in spans))
    ax.set_xlabel("x: source -> drain (nm)")
    ax.set_ylabel("y: width (nm)")
    ax.set_zlabel("z (nm)")
    ax.set_title(title)
    ax.view_init(elev=22, azim=-60)
    ax.legend(handles=[Patch(facecolor=COL[k], edgecolor="k", label=n) for k, n in legend],
              loc="upper left", fontsize=8)
    ax.text2D(0.02, 0.02, "oxide / gate thickness exaggerated for visibility",
              transform=ax.transAxes, fontsize=8, color="gray")


def draw_planar(ax, p):
    L, Lsd, Xj = p["L_nm"], p["Lsd_nm"], p["Xj_nm"]
    xt = 2 * Lsd + L
    W = min(p["W_um"] * 1e3, 2 * xt)                  # display width is capped
    tsub = max(2.5 * Xj, 0.4 * xt)
    tox = max(p["tox_nm"], 0.05 * L)
    tg = 0.5 * L
    add_box(ax, 0, xt, 0, W, -tsub, -Xj, COL["substrate"])
    add_box(ax, Lsd, Lsd + L, 0, W, -Xj, 0, COL["body"])
    add_box(ax, 0, Lsd, 0, W, -Xj, 0, COL["sd"])
    add_box(ax, Lsd + L, xt, 0, W, -Xj, 0, COL["sd"])
    add_box(ax, Lsd, Lsd + L, 0, W, 0, tox, COL["oxide"])
    add_box(ax, Lsd, Lsd + L, 0, W, tox, tox + tg, COL["gate"], 0.85)
    finish_axes(ax, (0, xt), (0, W), (-tsub, tox + tg), "Planar n-MOSFET",
                [("gate", "Gate"), ("oxide", "Gate oxide"), ("sd", "Source / Drain n+"),
                 ("body", "Channel (p-body)"), ("substrate", "Substrate")])


def draw_finfet(ax, p):
    H, Wf, N = p["Hfin_nm"], p["Wfin_nm"], int(p["Nfin"])
    L, Lsd = p["L_nm"], p["Lsd_nm"]
    tri = p["gate"] == "Tri-gate"
    tox = max(p["tox_nm"], 0.06 * L)
    tg = 0.6 * L
    cap = 0 if tri else 0.2 * H                       # hard-mask cap on a double-gate fin
    ztop = H + (tox if tri else cap)
    half = Wf / 2 + tox + tg
    pitch = Wf + 2 * (tox + tg) + Wf
    yc = [i * pitch for i in range(N)]
    y0, y1 = yc[0] - half, yc[-1] + half
    xs, xd, xt = Lsd, Lsd + L, 2 * Lsd + L
    add_box(ax, 0, xt, y0, y1, -0.5 * H, 0, COL["box"])
    for c in yc:
        add_box(ax, 0, xs, c - Wf / 2, c + Wf / 2, 0, H, COL["sd"])
        add_box(ax, xs, xd, c - Wf / 2, c + Wf / 2, 0, H, COL["fin"])
        add_box(ax, xd, xt, c - Wf / 2, c + Wf / 2, 0, H, COL["sd"])
        if cap:
            add_box(ax, xs, xd, c - Wf / 2, c + Wf / 2, H, H + cap, COL["cap"])
        add_box(ax, xs, xd, c - Wf / 2 - tox, c - Wf / 2, 0, H,
            COL["oxide"], 0.45)
        add_box(ax, xs, xd, c + Wf / 2, c + Wf / 2 + tox, 0, H,
            COL["oxide"], 0.45)
        for a, b in ((c - Wf / 2 - tox - tg, c - Wf / 2 - tox),
                     (c + Wf / 2 + tox, c + Wf / 2 + tox + tg)):
            add_box(ax, xs, xd, a, b, 0, H + (tox if tri else 0),
                COL["gate"], 0.55)
        if tri:
            add_box(ax, xs, xd, c - Wf / 2, c + Wf / 2, H, H + tox,
                COL["oxide"], 0.45)
            add_box(ax, xs, xd, c - Wf / 2 - tox, c + Wf / 2 + tox,
                ztop, ztop + tg, COL["gate"], 0.55)
    legend = [("gate", "Gate"), ("oxide", "Gate oxide"), ("fin", "Fin (channel)"),
              ("sd", "Source / Drain"), ("box", "BOX / substrate")]
    if cap:
        legend.insert(3, ("cap", "Fin cap"))
    zmax = ztop + tg if tri else ztop
    finish_axes(ax, (0, xt), (y0, y1), (-0.5 * H, zmax),
                f"FinFET  ({N} fin{'s' if N > 1 else ''}, {p['gate']})", legend)


DRAWERS = {"2D MOSFET": draw_planar, "FinFET": draw_finfet}


def draw_device(ax, p):
    DRAWERS[p["device"]](ax, p)
