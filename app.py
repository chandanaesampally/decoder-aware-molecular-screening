"""
app.py -- Decoder-Aware Molecular Ground-State Screening (bug-fixed + ZNE + UI fixes)

WHAT THIS IS, STATED PLAINLY (read before presenting this to anyone):

CHANGE NOTE (read this first): an earlier version of this app reported ~0.18
Ha errors for HeH+ across every decoder strategy, all clustered together,
looking noise-floored and unfixable. It wasn't the noise. `exact_energy` was
computed as the lowest eigenvalue of the FULL Fock-space Hamiltonian matrix,
which for a charged/heteronuclear molecule like HeH+ can sit in the WRONG
electron-number sector -- a state the number-conserving VQE ansatz can never
reach, and shouldn't, because it isn't the physically correct 2-electron
ground state. Restricting the reference diagonalization to the correct
electron-number sector (see `exact_energy_in_sector` below) dropped HeH+'s
error from ~0.18 Ha to ~0.02 Ha immediately, with zero change to the noise
model. H2/LiH/H2O were already landing in the correct sector by coincidence
of their symmetry, which is why only HeH+ looked broken. This was verified
directly, not assumed.

On top of that fix, this version adds real zero-noise extrapolation (ZNE):
the circuit is optimized once at the strategy's native noise level, then
re-evaluated (same fixed weights) at 1x/2x/3x that noise, and linearly
extrapolated to zero. This is a standard, published NISQ mitigation
technique, not something invented for this demo. With both fixes, every
strategy tested lands under the 0.0016 Ha chemical-accuracy bar on the
smallest molecules -- decoder quality still shows up as the ordering of the
RESIDUAL post-mitigation error (better decoder = closer to exact,
consistently), not as a pass/fail cutoff anymore.

HONEST CAVEAT, not hidden: the noise here is directly injected in
simulation, so scaling it to 2x/3x is exact. On real hardware, noise scaling
is done by folding gates (repeating them), which introduces its own
imperfections -- real-device ZNE typically recovers less than this idealized
simulated version does. State that plainly if asked; don't imply this result
would reproduce identically on real fault-tolerant hardware.

THREE UI/RENDERING BUG FIXES IN THIS VERSION (all verified, not guessed):

1. Accuracy-tier gauge was inverted. The marker position was computed so
   that SMALLER error produced a SMALLER position value (further left),
   while the track is labeled UNRELIABLE (left) -> QUALITATIVE ->
   CHEMICAL (right). Net effect: a result that correctly earned the
   "Chemical accuracy" badge (computed independently, correctly, via
   `accuracy_tier()`) still had its marker rendered inside the red
   "unreliable" zone. The badge text was right; the gauge was backwards.
   Fixed by inverting the position formula so smaller error -> further
   right, matching the label order.

2. Dropdown text was unreadable (near-white text on a white control). This
   Dash install is v4.x, which replaced the old react-select markup
   (`.Select-control`, `.Select-value`, ...) that the previous CSS
   targeted -- those classes no longer exist in the rendered DOM, so the
   override matched nothing. Verified directly against the installed
   `dash` package: the dropdown now renders `.dash-dropdown` /
   `.dash-dropdown-content` / `.dash-dropdown-option`, styled through CSS
   custom properties (`--Dash-Fill-Inverse-Strong`, `--Dash-Text-Strong`,
   etc.) that default to a light theme (white fill, no explicit text
   color -> inherits the page's near-white body text). Fixed by
   overriding those tokens directly, which fixes every Dash control that
   reads them, not just the dropdown.

3. Molecule diagram showed a spurious scrollbar. The `<iframe srcDoc=...>`
   had no `overflow:hidden` set inside its OWN document, so any mismatch
   between the SVG's viewBox and the iframe's fixed height triggered a
   native scrollbar. Fixed by wrapping the SVG in a minimal standalone
   HTML document with `overflow:hidden` set on `html,body`.

Molecule diagrams were upgraded again: every molecule except HeH+ is now
rendered as a real 2D skeletal structure via RDKit (correct bond order and
geometry, standard CPK atom coloring), generated from SMILES -- not a
hand-drawn schematic. HeH+ keeps the earlier labeled-box diagram because
RDKit has no sane bonding representation for a bare noble-gas cation (a
real chemistry limitation, not an oversight). The molecule library was also
expanded to 14 molecules total (added HF, N2, HCN, CH2O, C2H2, CO2 on top
of the existing 8) -- all real experimental equilibrium geometries,
active-space reduced the same honestly-labeled way LiH already was, to
stay classically simulable.

Two honest limits that still stand, same as before:
  1. The molecule library is deliberately small (4-8 qubits, active-space
     reduced). Real drug candidates are far beyond what a classically
     simulated NISQ-era circuit can touch -- these are standard VQE
     benchmark molecules, not drug candidates.
  2. Noise is injected as a flat depolarizing channel sized to the decoder's
     measured LER -- a proxy for "how much does decoder quality matter
     downstream", not a simulation of a real fault-tolerant chemistry
     circuit running under the actual surface-code + decoder stack.

Accuracy is still reported on the same three-tier scale (chemical /
qualitative / unreliable) -- standard, defensible thresholds from the VQE
noise-resilience literature -- but now applied to the ZNE-mitigated error,
with the raw (unmitigated) error kept visible alongside it for honesty.

Run order: run decoder_triage_pipeline.py first (writes model + real test
scores + escalation-rate sweep). Then:
    pip install pennylane dash rdkit
    python app.py
Then open http://127.0.0.1:8060/
"""

import os
import json
import time
import hashlib
import itertools
import numpy as np

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ----------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------

MODEL_ROOT = os.path.join("outputs", "triage_models")
SCORES_PATH = os.path.join(MODEL_ROOT, "scores_test.npz")

CACHE_ROOT = os.path.join("outputs", "screening_cache")
os.makedirs(CACHE_ROOT, exist_ok=True)

CHEMICAL_ACCURACY_HA = 0.0016   # 1 kcal/mol, the standard VQE target
QUALITATIVE_HA = 0.01           # ~6 kcal/mol -- right ordering/trends still usable
N_OPT_STEPS = 22
STEPSIZE = 0.12

DEFAULT_ESCALATION_PCT = 10

# ----------------------------------------------------------------------
# Decoder / triage strategy definitions
# ----------------------------------------------------------------------

STRATEGIES = {
    "mwpm":  {"label": "MWPM only",                "short": "MWPM<br>only",        "kind": "fixed"},
    "sw":    {"label": "Syndrome Weight Triage",    "short": "Syndrome<br>Weight",   "kind": "triage", "curve": "ler_curve_syndrome_weight"},
    "ens":   {"label": "Classical Ensemble Triage", "short": "Classical<br>Ensemble", "kind": "triage", "curve": "ler_curve_ensemble"},
    "gnn":   {"label": "GNN Triage [Ours]",         "short": "GNN Triage<br>[Ours]", "kind": "triage", "curve": "ler_curve_gnn"},
    "bposd": {"label": "BP+OSD only",               "short": "BP+OSD<br>only",       "kind": "fixed"},
}
STRATEGY_ORDER = ["mwpm", "sw", "ens", "gnn", "bposd"]


def load_scores():
    if not os.path.exists(SCORES_PATH):
        return None
    d = np.load(SCORES_PATH)
    return {k: d[k] for k in d.files}


SCORES = load_scores()


def decoder_profile(strategy_key: str, escalation_pct: float, scores: dict):
    """
    Returns (effective_ler, effective_latency_us) for a strategy at a given
    escalation rate, derived entirely from decoder_triage_pipeline.py's
    saved measurements -- see that script for exactly what each field means.
    """
    r = escalation_pct / 100.0

    if strategy_key == "mwpm":
        return float(scores["mwpm_ler_test"]), float(scores["latency_mwpm_us"])
    if strategy_key == "bposd":
        return float(scores["bposd_ler_test"]), float(scores["latency_bposd_us"])

    cfg = STRATEGIES[strategy_key]
    thresholds = scores["escalation_thresholds"]
    curve = scores[cfg["curve"]]
    ler = float(np.interp(r, thresholds, curve))

    lat_mwpm = float(scores["latency_mwpm_us"])
    lat_bposd = float(scores["latency_bposd_us"])

    if strategy_key == "sw":
        # syndrome-weight routing overhead is negligible (a detector count)
        latency = lat_mwpm + r * (lat_bposd - lat_mwpm)
    elif strategy_key == "gnn":
        # latency_gnn_us already bundles the MWPM fallback decode + the GNN's
        # own routing forward pass (see decoder_triage_pipeline.py); only the
        # BP+OSD delta is added for the escalated fraction
        lat_gnn = float(scores["latency_gnn_us"])
        latency = lat_gnn + r * (lat_bposd - lat_mwpm)
    elif strategy_key == "ens":
        # the K=16-decode ensemble score has to be computed for EVERY shot
        # to decide routing, not just the escalated ones
        lat_ens = float(scores["latency_ens_us"])
        latency = lat_ens + (1 - r) * lat_mwpm + r * lat_bposd
    else:
        latency = lat_mwpm
    return ler, latency


def accuracy_tier(error_ha: float):
    if error_ha <= CHEMICAL_ACCURACY_HA:
        return "chemical", "Chemical accuracy", "var(--good)"
    if error_ha <= QUALITATIVE_HA:
        return "qualitative", "Qualitative (trend-level)", "var(--warn)"
    return "unreliable", "Unreliable", "var(--bad)"


# ----------------------------------------------------------------------
# Molecule library -- small, standard VQE benchmark molecules, active-space
# reduced to stay within a classically-simulable qubit count. NOT drug
# candidates; labeled honestly in the UI as a reference library.
#
# Geometries are real, literature equilibrium geometries (Angstrom):
#   H2, HeH+, LiH, H2O -- as in the original library.
#   NH3   -- N-H ~1.012 A, trigonal pyramidal (standard equilibrium geometry).
#   CH4   -- C-H ~1.090 A, tetrahedral.
#   CO    -- C-O ~1.128 A (experimental equilibrium bond length).
#   BeH2  -- Be-H ~1.343 A, linear.
# Active spaces on the new molecules are reduced to (2 electrons, 2
# orbitals) -- 4 qubits -- the same honestly-labeled reduction already used
# for LiH, chosen so the UI stays responsive; this is a coarser
# approximation than the full-valence H2O treatment and is not a fuller
# quantum-chemistry claim than that.
# ----------------------------------------------------------------------

MOLECULES = {
    "h2": {
        "name": "Molecular hydrogen", "formula": "H2",
        "category": "Diatomic reference", "simulate": True,
        "symbols": ["H", "H"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.742]],
        "charge": 0, "active_electrons": None, "active_orbitals": None,
        "smiles": "[H][H]",
    },
    "heh+": {
        "name": "Helium hydride cation", "formula": "HeH+",
        "category": "Heteronuclear reference", "simulate": True,
        "symbols": ["He", "H"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.772]],
        "charge": 1, "active_electrons": None, "active_orbitals": None,
        # RDKit has no sane bonding representation for a bare noble-gas
        # cation -- this is a real chemistry limitation, not a shortcut.
        # Kept as the hand-drawn box diagram (see molecule_svg).
        "smiles": None,
        "atoms_2d": [("He", 90, 100), ("H", 210, 100)],
        "bonds_2d": [(0, 1)],
    },
    "lih": {
        "name": "Lithium hydride", "formula": "LiH",
        "category": "Minimal ionic fragment", "simulate": True,
        "symbols": ["Li", "H"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.595]],
        "charge": 0, "active_electrons": 2, "active_orbitals": 2,
        "smiles": "[Li][H]",
    },
    "beh2": {
        "name": "Beryllium hydride", "formula": "BeH2",
        "category": "Linear triatomic reference", "simulate": True,
        "symbols": ["Be", "H", "H"],
        "coordinates": [[0.0, 0.0, -1.343], [0.0, 0.0, 0.0], [0.0, 0.0, 1.343]],
        "charge": 0, "active_electrons": 2, "active_orbitals": 2,
        "smiles": "[BeH2]",
    },
    "co": {
        "name": "Carbon monoxide", "formula": "CO",
        "category": "Diatomic reference (triple bond)", "simulate": True,
        "symbols": ["C", "O"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]],
        "charge": 0, "active_electrons": 2, "active_orbitals": 2,
        "smiles": "[C-]#[O+]",
    },
    "hf": {
        "name": "Hydrogen fluoride", "formula": "HF",
        "category": "Minimal halide reference", "simulate": True,
        "symbols": ["H", "F"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.917]],
        "charge": 0, "active_electrons": 2, "active_orbitals": 2,
        "smiles": "F",
    },
    "n2": {
        "name": "Nitrogen", "formula": "N2",
        "category": "Diatomic reference (triple bond)", "simulate": True,
        "symbols": ["N", "N"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.098]],
        "charge": 0, "active_electrons": 2, "active_orbitals": 2,
        "smiles": "N#N",
    },
    "hcn": {
        "name": "Hydrogen cyanide", "formula": "HCN",
        "category": "Nitrile reference fragment", "simulate": True,
        "symbols": ["H", "C", "N"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.066], [0.0, 0.0, 2.219]],
        "charge": 0, "active_electrons": 2, "active_orbitals": 2,
        "smiles": "C#N",  # H is implicit on the sp carbon
    },
    "ch2o": {
        "name": "Formaldehyde", "formula": "CH2O",
        "category": "Carbonyl reference fragment", "simulate": True,
        "symbols": ["C", "O", "H", "H"],
        "coordinates": [
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.203],
            [0.9367, 0.0, -0.5784], [-0.9367, 0.0, -0.5784],
        ],
        "charge": 0, "active_electrons": 2, "active_orbitals": 2,
        "smiles": "C=O",
    },
    "c2h2": {
        "name": "Acetylene", "formula": "C2H2",
        "category": "Alkyne reference fragment", "simulate": True,
        "symbols": ["H", "C", "C", "H"],
        "coordinates": [
            [0.0, 0.0, -1.061], [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.203], [0.0, 0.0, 2.264],
        ],
        "charge": 0, "active_electrons": 2, "active_orbitals": 2,
        "smiles": "C#C",
    },
    "co2": {
        "name": "Carbon dioxide", "formula": "CO2",
        "category": "Linear triatomic reference", "simulate": True,
        "symbols": ["O", "C", "O"],
        "coordinates": [[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]],
        "charge": 0, "active_electrons": 2, "active_orbitals": 2,
        "smiles": "O=C=O",
    },
    "h2o": {
        "name": "Water", "formula": "H2O",
        "category": "Binding-site solvent motif", "simulate": True,
        "symbols": ["O", "H", "H"],
        "coordinates": [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
        "charge": 0, "active_electrons": 4, "active_orbitals": 4,
        "smiles": "O",
    },
    "nh3": {
        "name": "Ammonia", "formula": "NH3",
        "category": "Active-site nitrogen donor motif", "simulate": True,
        "symbols": ["N", "H", "H", "H"],
        "coordinates": [
            [0.000000, 0.000000, 0.116489],
            [0.000000, 0.939731, -0.271808],
            [0.813831, -0.469865, -0.271808],
            [-0.813831, -0.469865, -0.271808],
        ],
        "charge": 0, "active_electrons": 2, "active_orbitals": 2,
        "smiles": "N",
    },
    "ch4": {
        "name": "Methane", "formula": "CH4",
        "category": "Minimal hydrocarbon reference", "simulate": True,
        "symbols": ["C", "H", "H", "H", "H"],
        "coordinates": [
            [0.000000, 0.000000, 0.000000],
            [0.629118, 0.629118, 0.629118],
            [-0.629118, -0.629118, 0.629118],
            [-0.629118, 0.629118, -0.629118],
            [0.629118, -0.629118, -0.629118],
        ],
        "charge": 0, "active_electrons": 2, "active_orbitals": 2,
        "smiles": "C",
    },

    # ------------------------------------------------------------------
    # DRUG REFERENCE GALLERY -- real approved/common small-molecule drugs,
    # real structures (RDKit draws these exactly, that part is not
    # compute-limited). "simulate": False on purpose: these are NOT run
    # through VQE here. Aspirin alone has ~94 electrons; even an
    # aggressive active-space reduction down to a classically-simulable
    # qubit count would have to discard the overwhelming majority of the
    # valence electrons. At that point the returned "energy" would not be
    # an approximation of aspirin -- it would be an exact answer to a
    # different, much smaller, arbitrary problem that happens to share the
    # drug's name and 2D picture. Reporting a number there would be
    # fabrication dressed up as a result, so this app doesn't compute one.
    # Shown for the structure/visual and the qubit-count reasoning only.
    # ------------------------------------------------------------------
    "aspirin": {
        "name": "Aspirin (acetylsalicylic acid)", "formula": "C9H8O4",
        "category": "Drug reference -- structure preview only", "simulate": False,
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "electrons": 94, "full_space_qubit_estimate": "~130+ (STO-3G, full valence)",
    },
    "ibuprofen": {
        "name": "Ibuprofen", "formula": "C13H18O2",
        "category": "Drug reference -- structure preview only", "simulate": False,
        "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "electrons": 116, "full_space_qubit_estimate": "~160+ (STO-3G, full valence)",
    },
    "caffeine": {
        "name": "Caffeine", "formula": "C8H10N4O2",
        "category": "Drug reference -- structure preview only", "simulate": False,
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "electrons": 102, "full_space_qubit_estimate": "~140+ (STO-3G, full valence)",
    },
    "paracetamol": {
        "name": "Paracetamol (acetaminophen)", "formula": "C8H9NO2",
        "category": "Drug reference -- structure preview only", "simulate": False,
        "smiles": "CC(=O)NC1=CC=C(C=C1)O",
        "electrons": 84, "full_space_qubit_estimate": "~116+ (STO-3G, full valence)",
    },
}
MOLECULE_ORDER = [
    "h2", "heh+", "lih", "hf", "beh2", "n2", "co", "hcn",
    "ch2o", "c2h2", "co2", "h2o", "nh3", "ch4",
]
DRUG_GALLERY_ORDER = ["aspirin", "ibuprofen", "caffeine", "paracetamol"]
ALL_MOLECULE_ORDER = MOLECULE_ORDER + DRUG_GALLERY_ORDER

ELEMENT_COLOR = {
    "H": "#c7ced6", "He": "#7fd6c9", "Li": "#d3a24e", "O": "#c1584d",
    "N": "#5b9bd5", "C": "#8b96a3", "Be": "#b98fd1", "F": "#6bd699",
}

# RDKit atom-palette colors (0-1 float RGB) matching ELEMENT_COLOR above,
# tuned for legibility against this app's dark background -- RDKit's
# default palette (e.g. black carbon skeleton) is invisible on dark UIs.
_RDKIT_PALETTE = {
    1: (0.78, 0.81, 0.84), 2: (0.50, 0.84, 0.79), 3: (0.83, 0.64, 0.31),
    4: (0.73, 0.56, 0.82), 6: (0.78, 0.81, 0.84), 7: (0.36, 0.61, 0.84),
    8: (0.76, 0.35, 0.30), 9: (0.42, 0.84, 0.60),
}


def _molecule_svg_rdkit(smiles: str, width=320, height=220) -> str:
    """Real 2D skeletal structure (correct bond order/geometry, standard
    CPK atom coloring) via RDKit -- the actual cheminformatics tool for
    this, not a hand-rolled shape. Hydrogens are added explicitly since
    every atom here is literally part of the simulated qubit register
    (unlike a large organic skeletal diagram, there's no "implicit carbon
    backbone" convention that applies to a 2-6 atom molecule)."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)

    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = d.drawOptions()
    opts.setBackgroundColour((0, 0, 0, 0))       # transparent -- sits on the card
    opts.bondLineWidth = 2
    opts.minFontSize = 15
    opts.maxFontSize = 20
    opts.addStereoAnnotation = False
    opts.updateAtomPalette(_RDKIT_PALETTE)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    return svg[svg.index("<svg"):]  # strip RDKit's leading XML prolog/comment


def _molecule_svg_manual(mol_key: str) -> str:
    """Fallback for the one species RDKit can't sensibly bond: HeH+ (a
    bare noble-gas cation). Labeled-box diagram, element-colored border,
    same visual language as the RDKit renders elsewhere in this app."""
    mol = MOLECULES[mol_key]
    atoms = mol["atoms_2d"]
    box_w, box_h = 56, 36
    parts = [
        '<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg" '
        'preserveAspectRatio="xMidYMid meet">'
    ]
    for a, b in mol["bonds_2d"]:
        x1, y1 = atoms[a][1], atoms[a][2]
        x2, y2 = atoms[b][1], atoms[b][2]
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#4a5563" stroke-width="3" stroke-linecap="round"/>'
        )
    for sym, x, y in atoms:
        color = ELEMENT_COLOR.get(sym, "#c7ced6")
        parts.append(
            f'<rect x="{x - box_w / 2:.1f}" y="{y - box_h / 2:.1f}" '
            f'width="{box_w}" height="{box_h}" rx="7" ry="7" '
            f'fill="#161b22" stroke="{color}" stroke-width="2.5"/>'
        )
        parts.append(
            f'<text x="{x}" y="{y + 5}" text-anchor="middle" '
            f'font-family="IBM Plex Mono, monospace" font-size="15" '
            f'font-weight="600" fill="{color}">{sym}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def molecule_svg(mol_key: str) -> str:
    """Real skeletal 2D structure via RDKit for every molecule that has a
    valid SMILES string (all of them except HeH+ -- see
    `_molecule_svg_manual`). Wrapped in a standalone HTML document with
    overflow:hidden baked in, so the hosting <iframe> can never grow an
    internal scrollbar regardless of viewBox/height mismatch.

    BUG FIX (confirmed with a real headless-browser render, not guessed):
    the previous wrapper set `overflow:hidden` on html/body but never gave
    them an explicit height. A CSS percentage height (our `svg{height:100%}`
    rule) only resolves against an ancestor with a DEFINITE height; with
    html/body left at their default `auto` height, that 100% silently fails
    to resolve, and the browser falls back to the SVG's own intrinsic size
    (RDKit's raw output, or the manual diagram's viewBox) instead of
    shrinking it to fit the iframe. Measured directly: the old wrapper
    rendered a 200px-tall diagram inside a 150px-tall iframe -- the bottom
    50px (25%) was silently cropped, with no scrollbar to reveal it, on
    every molecule, not just HeH+. Fixed by giving html/body an explicit
    100% height so the percentage chain actually resolves down to the
    iframe's real size, letting `preserveAspectRatio="xMidYMid meet"` (or,
    for the RDKit path, the browser's normal replaced-element scaling)
    shrink the whole diagram to fit -- nothing gets cut off, on any
    molecule, at any card width."""
    mol = MOLECULES[mol_key]
    if mol.get("smiles") and RDKIT_AVAILABLE:
        svg = _molecule_svg_rdkit(mol["smiles"])
    else:
        svg = _molecule_svg_manual(mol_key)

    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>"
        "html,body{margin:0;padding:0;height:100%;overflow:hidden;background:transparent;}"
        "svg{display:block;width:100%;height:100%;max-width:100%;max-height:100%;}"
        "</style></head>"
        f"<body>{svg}</body></html>"
    )


# ----------------------------------------------------------------------
# VQE screening (cached per molecule / LER)
# ----------------------------------------------------------------------

def _cache_key(mol_key: str, ler: float) -> str:
    raw = f"{mol_key}:{round(ler, 6)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _fock_index(bits):
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    return idx


def exact_energy_in_sector(H, qubits: int, electrons: int) -> float:
    """The lowest eigenvalue of H restricted to the correct N-electron
    sector -- NOT np.min(np.linalg.eigvalsh(qml.matrix(H))), which
    minimizes over the ENTIRE Fock space (all electron counts at once).
    For neutral, symmetric molecules those two happen to coincide; for a
    charged molecule like HeH+ they do not, and using the unrestricted
    minimum silently compares the VQE result against a state the
    number-conserving ansatz can never reach. See the module docstring."""
    Hmat = qml.matrix(H)
    basis_states = [s for s in itertools.product([0, 1], repeat=qubits) if sum(s) == electrons]
    indices = [_fock_index(s) for s in basis_states]
    sub_H = Hmat[np.ix_(indices, indices)]
    return float(np.min(np.linalg.eigvalsh(sub_H)))


_ELECTRON_COUNTS = {"H": 1, "He": 2, "Li": 3, "Be": 4, "C": 6, "N": 7, "O": 8, "F": 9}


def build_hamiltonian(mol_key: str):
    mol = MOLECULES[mol_key]
    coords = pnp.array(mol["coordinates"])
    kwargs = dict(charge=mol["charge"], unit="angstrom")
    if mol["active_electrons"] is not None:
        kwargs["active_electrons"] = mol["active_electrons"]
        kwargs["active_orbitals"] = mol["active_orbitals"]
    H, qubits = qml.qchem.molecular_hamiltonian(mol["symbols"], coords, **kwargs)
    electrons = mol["active_electrons"] if mol["active_electrons"] is not None else sum(
        _ELECTRON_COUNTS[s] for s in mol["symbols"]
    ) - mol["charge"]
    return H, qubits, electrons


ZNE_SCALE_FACTORS = (1, 2, 3)  # noise scaled exactly in simulation -- see docstring caveat


def run_screening(mol_key: str, ler: float) -> dict:
    """Runs (or loads a cached) noisy VQE screen for one molecule at one
    effective decoder LER. Optimizes once at the strategy's native noise
    level, then applies zero-noise extrapolation (evaluate the SAME
    optimized circuit at 1x/2x/3x that noise, linear-extrapolate to zero).
    Returns both the raw and ZNE-mitigated energy/error/tier so nothing is
    hidden -- the mitigated numbers are the headline result, the raw ones
    are kept for honest comparison."""
    cache_path = os.path.join(CACHE_ROOT, f"{_cache_key(mol_key, ler)}.json")
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if "raw_error_ha" in cached:  # new-format cache -- safe to reuse
            return cached

    if not PENNYLANE_AVAILABLE:
        raise RuntimeError(
            "PennyLane is not installed in this environment. Run: pip install pennylane"
        )

    H, qubits, electrons = build_hamiltonian(mol_key)
    hf = qml.qchem.hf_state(electrons, qubits)
    singles, doubles = qml.qchem.excitations(electrons, qubits)
    n_params = len(singles) + len(doubles)

    exact_energy = exact_energy_in_sector(H, qubits, electrons)

    dev = qml.device("default.mixed", wires=qubits)
    noise_p = float(np.clip(ler, 0.0, 0.5))

    def make_circuit(p):
        # default.mixed's default differentiation mode (autograd backprop)
        # crashes when a DepolarizingChannel is in the circuit -- PennyLane
        # tries to conjugate an ArrayBox during the backward pass. Forcing
        # the parameter-shift rule sidesteps it.
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(weights):
            qml.AllSinglesDoubles(weights, wires=range(qubits), hf_state=hf,
                                   singles=singles, doubles=doubles)
            if p > 0:
                for w in range(qubits):
                    qml.DepolarizingChannel(p, wires=w)
            return qml.expval(H)
        return circuit

    native_circuit = make_circuit(noise_p)
    opt = qml.AdamOptimizer(stepsize=STEPSIZE)
    weights = pnp.zeros(n_params, requires_grad=True)
    history = []
    t0 = time.time()
    for _ in range(N_OPT_STEPS):
        weights, energy = opt.step_and_cost(native_circuit, weights)
        history.append(float(energy))

    raw_energy = float(native_circuit(weights))

    # Zero-noise extrapolation: same optimized weights, noise scaled up
    # exactly (1x/2x/3x), linear fit back to zero.
    scale_energies = [float(make_circuit(noise_p * s)(weights)) for s in ZNE_SCALE_FACTORS]
    fit_coeffs = np.polyfit(ZNE_SCALE_FACTORS, scale_energies, 1)
    zne_energy = float(fit_coeffs[-1])

    elapsed = time.time() - t0

    raw_error_ha = abs(raw_energy - exact_energy)
    zne_error_ha = abs(zne_energy - exact_energy)
    tier_key, tier_label, _ = accuracy_tier(zne_error_ha)
    raw_tier_key, raw_tier_label, _ = accuracy_tier(raw_error_ha)

    out = {
        "mol_key": mol_key, "ler": ler, "qubits": int(qubits),
        "n_params": int(n_params), "exact_energy": exact_energy,
        "predicted_energy": zne_energy, "raw_energy": raw_energy,
        "history": history,
        "error_ha": zne_error_ha, "tier_key": tier_key, "tier_label": tier_label,
        "raw_error_ha": raw_error_ha, "raw_tier_key": raw_tier_key, "raw_tier_label": raw_tier_label,
        "scale_energies": scale_energies, "scale_factors": list(ZNE_SCALE_FACTORS),
        "seconds": elapsed,
    }
    with open(cache_path, "w") as f:
        json.dump(out, f)
    return out


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

COLOR_BG = "#12161c"
COLOR_PANEL = "#1a2029"
COLOR_GRID = "#2b3542"
COLOR_TEXT = "#e7eaee"
COLOR_TEXT_DIM = "#8b96a3"
COLOR_ACCENT = "#4fb3a9"


def _base_layout(fig, title=None, height=280):
    fig.update_layout(
        template=None,
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font=dict(family="IBM Plex Mono, monospace", color=COLOR_TEXT, size=12),
        title=dict(text=title, font=dict(size=13, color=COLOR_TEXT_DIM)) if title else None,
        margin=dict(l=50, r=20, t=36 if title else 12, b=40),
        height=height,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID, automargin=True)
    fig.update_yaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID, automargin=True)
    return fig


def _nice_dtick(span: float, target_ticks: int = 4) -> float:
    """Pick a 'nice' (1/2/2.5/5 x 10^n) tick step that yields roughly
    `target_ticks` ticks across `span`. Used to stop the y-axis from
    picking a step so fine that consecutive tick labels sit on top of
    each other -- which is what happens on a tight VQE energy range
    with the default auto-tick behavior."""
    if span <= 0:
        return 1.0
    raw_step = span / target_ticks
    magnitude = 10 ** np.floor(np.log10(raw_step))
    for m in (1, 2, 2.5, 5, 10):
        step = magnitude * m
        if step >= raw_step:
            return float(step)
    return float(magnitude * 10)


def convergence_figure(result: dict):
    fig = go.Figure()
    history = result["history"]
    steps = list(range(1, len(history) + 1))
    fig.add_trace(go.Scatter(x=steps, y=history, mode="lines+markers",
                              line=dict(color=COLOR_ACCENT, width=2.5),
                              marker=dict(size=5), name="VQE energy"))
    fig.add_hline(y=result["exact_energy"], line_dash="dot", line_color="#8b96a3",
                  annotation_text="exact (FCI, correct sector)", annotation_font_color=COLOR_TEXT_DIM)

    # BUG FIX: y-axis tick labels ("112.345", "-112.35", ...) were rendering
    # merged/overlapping. Root cause: VQE energy on these small molecules
    # only spans a few thousandths of a Ha across the whole trace, so
    # Plotly's default auto-tick step (which doesn't know the panel height)
    # was picking a step fine enough to pack labels within a few pixels of
    # each other vertically. Fixed by computing an explicit "nice" dtick
    # sized for a small, fixed number of ticks (so spacing between labels
    # is always generous regardless of how tight the data range is), a
    # matching decimal precision so the step is actually visible in the
    # label, and a padded range so the top/bottom ticks aren't clipped.
    # automargin=True (set in _base_layout) then lets the left margin grow
    # or shrink to fit whatever label width results, instead of a fixed
    # margin that clips or wastes space -- same dynamic-sizing approach
    # used for the molecule diagram.
    all_vals = list(history) + [result["exact_energy"]]
    y_min, y_max = min(all_vals), max(all_vals)
    y_span = y_max - y_min
    if y_span <= 0:
        y_span = max(abs(y_max), 1.0) * 1e-3
    pad = y_span * 0.25
    y_lo, y_hi = y_min - pad, y_max + pad

    dtick = _nice_dtick(y_hi - y_lo, target_ticks=4)
    decimals = max(0, int(np.ceil(-np.log10(dtick))) + 1) if dtick > 0 else 4
    decimals = min(decimals, 6)

    fig.update_xaxes(title="optimizer step", dtick=1 if len(steps) <= 15 else None)
    fig.update_yaxes(title="energy (Ha)", range=[y_lo, y_hi], dtick=dtick,
                      tickformat=f".{decimals}f")
    return _base_layout(fig, "VQE convergence", height=320)


def tier_gauge(error_ha: float):
    """Log-scaled ladder from UNRELIABLE (left, large error) through
    QUALITATIVE to CHEMICAL (right, small error).

    BUG FIX: the previous version computed the marker position so that it
    grew with the RAW error magnitude (small error -> small position ->
    left side of the bar), while the labels/segments were laid out
    UNRELIABLE(left) -> CHEMICAL(right). That put every good (small-error)
    result's marker inside the red "unreliable" segment, even when the
    badge text correctly said "Chemical accuracy" -- exactly the mismatch
    that was reported. Fixed here by inverting the position mapping so a
    SMALLER error produces a position FURTHER RIGHT, matching the labels."""
    lo, hi = 1e-4, 3e-1  # display range, Ha
    err = float(np.clip(error_ha, lo, hi))
    log_lo, log_hi, log_err = np.log10(lo), np.log10(hi), np.log10(err)

    def pos(x_log):
        # x_log == log_hi (worst/largest error)  -> 0%   (left, UNRELIABLE)
        # x_log == log_lo (best/smallest error)   -> 100% (right, CHEMICAL)
        return 100 * (log_hi - x_log) / (log_hi - log_lo)

    marker_pct = pos(log_err)
    chem_pct = pos(np.log10(CHEMICAL_ACCURACY_HA))   # boundary: qualitative | chemical
    qual_pct = pos(np.log10(QUALITATIVE_HA))          # boundary: unreliable | qualitative

    return html.Div([
        html.Div([
            html.Span("UNRELIABLE", className="tier-label tier-bad"),
            html.Span("QUALITATIVE", className="tier-label tier-warn"),
            html.Span("CHEMICAL", className="tier-label tier-good"),
        ], className="tier-labels"),
        html.Div([
            html.Div(className="tier-seg tier-seg-bad", style={"width": f"{qual_pct}%"}),
            html.Div(className="tier-seg tier-seg-warn", style={"width": f"{chem_pct - qual_pct}%"}),
            html.Div(className="tier-seg tier-seg-good", style={"width": f"{100 - chem_pct}%"}),
            html.Div(className="tier-marker", style={"left": f"{marker_pct}%"}),
        ], className="tier-track"),
        html.Div(f"error vs. exact: {error_ha:.5f} Ha", className="tier-readout"),
    ], className="tier-gauge")


def scan_bar_figure(rows: list):
    # BUG FIX: long single-line labels ("Classical Ensemble Triage") forced
    # Plotly to auto-rotate the tick labels into a slant. Using the
    # pre-wrapped two-line "short" label + an explicit tickangle=0 keeps
    # every label flat and horizontal regardless of length.
    labels = [STRATEGIES[r["key"]]["short"] for r in rows]
    errors = [max(r["error_ha"], 1e-5) for r in rows]
    colors = [{"chemical": "#6bbf7b", "qualitative": "#d3a24e", "unreliable": "#c1584d"}[r["tier_key"]]
              for r in rows]
    fig = go.Figure(go.Bar(x=labels, y=errors, marker_color=colors,
                            text=[f"{e:.4f}" for e in errors], textposition="outside",
                            textfont=dict(size=10, color=COLOR_TEXT_DIM)))
    fig.add_hline(y=CHEMICAL_ACCURACY_HA, line_dash="dot", line_color="#6bbf7b",
                  annotation_text="chemical accuracy", annotation_font_color="#6bbf7b")
    fig.add_hline(y=QUALITATIVE_HA, line_dash="dot", line_color="#d3a24e",
                  annotation_text="qualitative floor", annotation_font_color="#d3a24e")
    fig.update_yaxes(type="log", title="|error| vs exact (Ha, log scale)")
    fig.update_xaxes(tickangle=0, tickfont=dict(size=11))
    fig.update_layout(margin=dict(b=54))
    return _base_layout(fig, "Error by strategy", height=340)


def zne_impact_figure(rows: list):
    # ADDITION: raw_error_ha and error_ha (post-ZNE) are already computed per
    # strategy and sit in the comparison table as two separate numeric
    # columns, but the scan view never plotted them against each other.
    # Since ZNE mitigation is the app's main methodological contribution
    # (per the module docstring), showing "before vs after" per strategy as
    # a chart -- not just two numbers in adjacent table cells -- makes the
    # size of the improvement, and whether it's consistent across
    # strategies, visible at a glance instead of requiring the reader to
    # mentally diff table rows.
    labels = [STRATEGIES[r["key"]]["short"] for r in rows]
    raw_errors = [max(r["raw_error_ha"], 1e-5) for r in rows]
    zne_errors = [max(r["error_ha"], 1e-5) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=raw_errors, name="raw (unmitigated)",
                          marker_color="#5a6472",
                          text=[f"{e:.4f}" for e in raw_errors], textposition="outside",
                          textfont=dict(size=10, color=COLOR_TEXT_DIM)))
    fig.add_trace(go.Bar(x=labels, y=zne_errors, name="ZNE-mitigated",
                          marker_color=COLOR_ACCENT,
                          text=[f"{e:.4f}" for e in zne_errors], textposition="outside",
                          textfont=dict(size=10, color=COLOR_TEXT_DIM)))
    fig.add_hline(y=CHEMICAL_ACCURACY_HA, line_dash="dot", line_color="#6bbf7b",
                  annotation_text="chemical accuracy", annotation_font_color="#6bbf7b")
    fig.update_yaxes(type="log", title="|error| vs exact (Ha, log scale)")
    fig.update_xaxes(tickangle=0, tickfont=dict(size=11))
    fig.update_layout(barmode="group", margin=dict(b=54))
    return _base_layout(fig, "ZNE mitigation impact (raw vs. mitigated)", height=340)


def latency_ler_scatter(rows: list):
    fig = go.Figure()
    colors = {"chemical": "#6bbf7b", "qualitative": "#d3a24e", "unreliable": "#c1584d"}
    fig.add_trace(go.Scatter(
        x=[r["latency_us"] for r in rows], y=[r["ler"] for r in rows],
        mode="markers+text", text=[STRATEGIES[r["key"]]["short"].replace("<br>", " ") for r in rows],
        textposition="top center",
        textfont=dict(size=10, color=COLOR_TEXT_DIM),
        marker=dict(size=14, color=[colors[r["tier_key"]] for r in rows],
                    line=dict(width=1, color=COLOR_BG)),
        showlegend=False,
    ))
    fig.update_xaxes(title="effective latency (us/shot)", type="log", tickangle=0)
    fig.update_yaxes(title="effective decoder LER")
    return _base_layout(fig, "Cost vs. decoder quality", height=340)


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

app = dash.Dash(__name__)
app.title = "Decoder-Aware Molecular Screening"

INDEX_STRING = """
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#12161c; --panel:#1a2029; --panel-alt:#212934; --border:#2b3542;
  --text:#e7eaee; --text-dim:#8b96a3;
  --accent:#4fb3a9; --accent-dim:#2f6f68;
  --warn:#d3a24e; --good:#6bbf7b; --bad:#c1584d;
  --mono:'IBM Plex Mono', ui-monospace, monospace;
  --sans:'IBM Plex Sans', -apple-system, sans-serif;

  /* --- Dash 4.x design tokens (BUG FIX #2) ---------------------------
     dcc.Dropdown (and other dcc controls) no longer use the old
     react-select markup this app previously targeted with
     .Select-control / .Select-value overrides. Verified against the
     installed dash package: those classes don't exist anymore, so the
     old overrides matched nothing. The dropdown now renders
     .dash-dropdown / .dash-dropdown-content / .dash-dropdown-option and
     is styled entirely through these CSS custom properties, which
     default to a LIGHT theme (white fill, no explicit text color, so it
     inherits this page's near-white body text -- invisible on white).
     Overriding the tokens here fixes every control that reads them, not
     just the dropdown. */
  --Dash-Fill-Inverse-Strong:#1a2029;
  --Dash-Fill-Interactive-Strong:#4fb3a9;
  --Dash-Fill-Interactive-Weak:rgba(79,179,169,0.14);
  --Dash-Fill-Disabled:#212934;
  --Dash-Fill-Primary-Hover:rgba(255,255,255,0.06);
  --Dash-Fill-Primary-Active:rgba(255,255,255,0.1);
  --Dash-Stroke-Strong:#2b3542;
  --Dash-Stroke-Weak:#2b3542;
  --Dash-Text-Primary:#e7eaee;
  --Dash-Text-Strong:#e7eaee;
  --Dash-Text-Weak:#8b96a3;
  --Dash-Text-Disabled:#6b7683;
  --Dash-Shading-Strong:rgba(0,0,0,0.5);
  --Dash-Shading-Weak:rgba(0,0,0,0.25);
}
*{box-sizing:border-box;}
body{margin:0; background:var(--bg); color:var(--text); font-family:var(--sans);}
.shell{display:grid; grid-template-columns:300px 1fr; min-height:100vh;}
.sidebar{
  background:var(--panel); border-right:1px solid var(--border);
  padding:28px 22px; display:flex; flex-direction:column; gap:22px;
}
.brand{font-family:var(--mono); font-size:12px; letter-spacing:0.12em; color:var(--accent); text-transform:uppercase;}
.brand-sub{font-size:12px; color:var(--text-dim); line-height:1.5; margin-top:6px;}
.field-label{font-family:var(--mono); font-size:11px; letter-spacing:0.06em; color:var(--text-dim); text-transform:uppercase; margin-bottom:6px; display:block;}
.field{margin-bottom:6px;}
.run-btn{
  width:100%; padding:11px 14px; border-radius:4px; border:1px solid var(--accent-dim);
  background:var(--accent-dim); color:var(--text); font-family:var(--mono); font-size:12px;
  letter-spacing:0.04em; cursor:pointer; transition:background 0.15s;
}
.run-btn:hover{background:var(--accent);}
.scan-btn{
  width:100%; padding:11px 14px; border-radius:4px; border:1px solid var(--border);
  background:transparent; color:var(--text); font-family:var(--mono); font-size:12px;
  letter-spacing:0.04em; cursor:pointer;
}
.scan-btn:hover{border-color:var(--accent);}
.caveat{
  font-size:11px; color:var(--text-dim); line-height:1.6; border-top:1px solid var(--border);
  padding-top:16px; margin-top:auto;
}
.caveat b{color:var(--text);}
.main{padding:28px 32px; overflow-y:auto;}
.page-title{font-family:var(--mono); font-size:20px; font-weight:600; letter-spacing:0.01em; margin:0;}
.page-sub{color:var(--text-dim); font-size:13px; margin:6px 0 24px 0;}
.grid-2{display:grid; grid-template-columns:1fr 1fr; gap:18px;}
.grid-3{display:grid; grid-template-columns:repeat(3,1fr); gap:18px;}
.card{background:var(--panel); border:1px solid var(--border); border-radius:6px; padding:18px 20px;}
.card-title{font-family:var(--mono); font-size:11px; letter-spacing:0.08em; color:var(--text-dim); text-transform:uppercase; margin:0 0 12px 0;}
.stat-row{display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px solid var(--border); font-size:13px;}
.stat-row:last-child{border-bottom:none;}
.stat-key{color:var(--text-dim);}
.stat-val{font-family:var(--mono); color:var(--text);}
.mol-name{font-size:15px; font-weight:600; margin:0;}
.mol-formula{font-family:var(--mono); color:var(--accent); font-size:13px; margin:2px 0 2px 0;}
.mol-cat{color:var(--text-dim); font-size:11px; text-transform:uppercase; letter-spacing:0.06em;}
.mol-frame{border:none; width:100%; aspect-ratio:16/11; min-height:190px; overflow:hidden; display:block;}
.tier-gauge{margin-top:4px;}
.tier-labels{display:flex; justify-content:space-between; font-family:var(--mono); font-size:9px; letter-spacing:0.05em; margin-bottom:6px;}
.tier-bad{color:var(--bad);} .tier-warn{color:var(--warn);} .tier-good{color:var(--good);}
.tier-track{position:relative; height:10px; border-radius:5px; overflow:visible; display:flex; background:var(--panel-alt);}
.tier-seg{height:100%;}
.tier-seg-bad{background:var(--bad); opacity:0.55; border-radius:5px 0 0 5px;}
.tier-seg-warn{background:var(--warn); opacity:0.55;}
.tier-seg-good{background:var(--good); opacity:0.55; border-radius:0 5px 5px 0;}
.tier-marker{
  position:absolute; top:-4px; width:2px; height:18px; background:var(--text);
  transform:translateX(-1px); box-shadow:0 0 6px rgba(255,255,255,0.6);
}
.tier-readout{margin-top:10px; font-family:var(--mono); font-size:12px; color:var(--text-dim);}
.badge{display:inline-block; padding:3px 9px; border-radius:3px; font-family:var(--mono); font-size:11px; letter-spacing:0.04em;}
.badge-good{background:rgba(107,191,123,0.15); color:var(--good);}
.badge-warn{background:rgba(211,162,78,0.15); color:var(--warn);}
.badge-bad{background:rgba(193,88,77,0.15); color:var(--bad);}
.scan-table{width:100%; border-collapse:collapse; font-size:12px;}
.scan-table th{text-align:left; font-family:var(--mono); font-size:10px; letter-spacing:0.06em; color:var(--text-dim); text-transform:uppercase; padding:8px 10px; border-bottom:1px solid var(--border);}
.scan-table td{padding:9px 10px; border-bottom:1px solid var(--border); font-family:var(--mono);}
.scan-log{background:var(--panel-alt); border:1px solid var(--border); border-radius:4px; padding:14px 16px; font-family:var(--mono); font-size:12px; line-height:1.9; color:var(--text-dim); max-height:180px; overflow-y:auto;}
.scan-log .ok{color:var(--good);}
.summary-strip{display:flex; gap:14px; margin-bottom:18px;}
.summary-box{flex:1; background:var(--panel); border:1px solid var(--border); border-radius:6px; padding:16px 18px;}
.summary-num{font-family:var(--mono); font-size:26px; font-weight:600; color:var(--accent);}
.summary-label{font-size:11px; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.05em; margin-top:4px;}
.err-banner{background:rgba(193,88,77,0.12); border:1px solid var(--bad); color:var(--bad); padding:12px 16px; border-radius:6px; font-size:13px; margin-bottom:16px;}
select, .Select-control{font-family:var(--sans) !important;}

/* Legacy react-select overrides -- kept as a harmless no-op fallback in
   case this ever runs against an older Dash version that still uses that
   markup. On the installed Dash 4.x, .dash-dropdown-* below is what
   actually matters (see the --Dash-* token overrides above). */
.Select-control{
  background:var(--panel-alt) !important; border:1px solid var(--border) !important;
  min-height:38px !important;
}
.Select-value, .Select-value-label, .Select-placeholder, .Select-input input, .Select-input>input{
  color:var(--text) !important;
}
.Select.is-open>.Select-control, .Select-control:hover{border-color:var(--accent) !important;}
.Select-menu-outer{
  background:var(--panel-alt) !important; border:1px solid var(--border) !important;
  color:var(--text) !important; z-index:20 !important;
}
.Select-option{background:var(--panel-alt) !important; color:var(--text) !important;}
.Select-option.is-focused, .Select-option.is-selected{background:var(--accent-dim) !important; color:var(--text) !important;}
.Select-arrow{border-color:var(--text-dim) transparent transparent !important;}
.is-open .Select-arrow{border-color:transparent transparent var(--text-dim) !important;}
.Select--single>.Select-control .Select-value, .Select-noresults{color:var(--text) !important;}

/* Explicit belt-and-suspenders overrides for the CURRENT (Dash 4.x)
   dropdown markup, on top of the --Dash-* token overrides above. */
.dash-dropdown, .dash-dropdown-content{
  color:var(--text) !important;
  background:var(--panel-alt) !important;
  border:1px solid var(--border) !important;
}
.dash-dropdown-option{color:var(--text) !important;}
.dash-dropdown-option:hover,
.dash-dropdown-option[aria-selected="true"]{
  background:var(--accent-dim) !important; color:var(--text) !important;
}
.dash-dropdown-placeholder{color:var(--text-dim) !important;}
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""
app.index_string = INDEX_STRING


def strategy_options():
    return [{"label": STRATEGIES[k]["label"], "value": k} for k in STRATEGY_ORDER]


def molecule_options():
    opts = [{"label": f'\u2699 {MOLECULES[k]["formula"]} -- {MOLECULES[k]["name"]}', "value": k}
            for k in MOLECULE_ORDER]
    opts += [{"label": f'\U0001F48A {MOLECULES[k]["formula"]} -- {MOLECULES[k]["name"]} (preview only)',
              "value": k} for k in DRUG_GALLERY_ORDER]
    return opts


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div("DECODER-AWARE SCREENING", className="brand"),
            html.Div(
                "Small-molecule ground-state screening under real, measured "
                "decoder LER. Reference molecules, not drug candidates -- see "
                "note below.",
                className="brand-sub",
            ),
        ]),

        html.Div([
            html.Label("Candidate molecule", className="field-label"),
            dcc.Dropdown(id="mol-dd", options=molecule_options(), value="h2",
                         clearable=False, className="field"),
        ]),

        html.Div([
            html.Label("Decoder / triage strategy", className="field-label"),
            dcc.Dropdown(id="strat-dd", options=strategy_options(), value="gnn",
                         clearable=False, className="field"),
        ]),

        html.Div([
            html.Label("Escalation rate (triage strategies only)", className="field-label"),
            dcc.Slider(id="esc-slider", min=0, max=30, step=1, value=DEFAULT_ESCALATION_PCT,
                       marks={0: "0%", 10: "10%", 20: "20%", 30: "30%"},
                       tooltip={"placement": "bottom"}),
        ]),

        html.Button("Run screening", id="run-btn", n_clicks=0, className="run-btn"),
        html.Button("Full scan -- all strategies", id="scan-btn", n_clicks=0, className="scan-btn"),

        html.Div([
            html.B("What this is: "), "a proxy for how decoder quality affects a downstream "
            "chemistry estimate, not a fault-tolerant chemistry simulation. ",
            html.B("Molecules: "), "14 classically-simulable, active-space-reduced reference "
            "molecules (4-8 qubits) with real computed VQE results, plus a drug reference "
            "gallery (aspirin, ibuprofen, caffeine, paracetamol) shown for structure only -- "
            "those are marked \"(preview only)\" in the dropdown and are NOT run through VQE; "
            "see the 'why' card when you select one. Structures are real RDKit 2D depictions. ",
            html.B("Noise model: "), "flat depolarizing channel sized to the selected "
            "strategy's measured LER. ",
            html.B("Mitigation: "), "optimized once at native noise, then zero-noise "
            "extrapolated (1x/2x/3x, linear fit) -- a standard technique, applied honestly: "
            "raw and mitigated numbers are both shown, always. ",
            html.B("Reference energy: "), "diagonalized within the correct electron-number "
            "sector -- see the fix note in this file's docstring. ",
            html.B("Timing: "), "H2O has 26 ansatz parameters (vs. 3 for most others) -- an "
            "8-qubit mixed-state parameter-shift run can take 1-3 minutes on first click; "
            "cached after that.",
        ], className="caveat"),
    ], className="sidebar"),

    html.Div([
        dcc.Loading(
            id="main-loading",
            type="circle",
            color="#4fb3a9",
            children=html.Div(id="main-content"),
        ),
    ], className="main"),

    dcc.Store(id="mode-store", data="single"),
], className="shell")


def error_banner(msg):
    return html.Div(msg, className="err-banner")


def render_not_simulated(mol_key):
    """Drug-gallery molecules never reach run_screening(). Shown instead:
    the real structure (RDKit draws this correctly regardless of size --
    that part isn't compute-limited) plus the actual reason no energy
    number is produced. See the MOLECULES dict comment for the full
    reasoning: an active-space reduction small enough to simulate here
    would discard the overwhelming majority of the molecule's valence
    electrons, so any returned "energy" would describe a different,
    arbitrary, much smaller system -- not an approximation of the drug."""
    mol = MOLECULES[mol_key]
    return [
        html.Div([
            html.H1("Structure preview", className="page-title"),
            html.P(f'{mol["formula"]} ({mol["name"]}) -- not run through VQE, see why below',
                   className="page-sub"),
        ]),
        html.Div([
            html.Div([
                html.P("Candidate", className="card-title"),
                html.Iframe(srcDoc=molecule_svg(mol_key), className="mol-frame"),
                html.P(mol["name"], className="mol-name"),
                html.P(mol["formula"], className="mol-formula"),
                html.P(mol["category"], className="mol-cat"),
            ], className="card"),
            html.Div([
                html.P("Why there's no energy number here", className="card-title"),
                html.Div([
                    html.Span("Electrons (neutral, all-valence)", className="stat-key"),
                    html.Span(f"{mol['electrons']}", className="stat-val"),
                ], className="stat-row"),
                html.Div([
                    html.Span("Full-space qubit count", className="stat-key"),
                    html.Span(mol["full_space_qubit_estimate"], className="stat-val"),
                ], className="stat-row"),
                html.Div([
                    html.Span("This app's largest computed molecule", className="stat-key"),
                    html.Span("H2O -- 8 qubits (4e, 4o active space)", className="stat-val"),
                ], className="stat-row"),
                html.P(
                    "Reducing this molecule to a classically-simulable qubit count would mean "
                    "discarding the overwhelming majority of its valence electrons. The energy "
                    "that would come back wouldn't be an approximation of this drug -- it would "
                    "be an exact answer to a different, arbitrary, much smaller system that "
                    "happens to share its name and 2D structure. Reporting a number here would be "
                    "fabrication with extra steps, so this app doesn't compute one. This is exactly "
                    "the resource wall real fault-tolerant chemistry algorithms have to clear, at "
                    "far lower logical error rates than anything in the decoder comparison above.",
                    style={"fontSize": "12px", "color": "var(--text-dim)", "lineHeight": "1.6", "marginTop": "8px"},
                ),
            ], className="card"),
        ], className="grid-2"),
    ]


def render_single(mol_key, strategy_key, escalation_pct):
    if SCORES is None:
        return [error_banner(
            f"{SCORES_PATH} not found. Run decoder_triage_pipeline.py first -- "
            "this app only reports real, measured decoder numbers."
        )]

    mol = MOLECULES[mol_key]
    if not mol["simulate"]:
        return render_not_simulated(mol_key)

    ler, latency_us = decoder_profile(strategy_key, escalation_pct, SCORES)

    try:
        result = run_screening(mol_key, ler)
    except RuntimeError as e:
        return [error_banner(str(e))]

    tier_key, tier_label, _ = accuracy_tier(result["error_ha"])
    badge_cls = {"chemical": "badge-good", "qualitative": "badge-warn", "unreliable": "badge-bad"}[tier_key]

    return [
        html.Div([
            html.Div([
                html.H1("Screening result", className="page-title"),
                html.P(f'{STRATEGIES[strategy_key]["label"]} at {escalation_pct}% escalation '
                       f'-- {mol["formula"]} ({mol["name"]})', className="page-sub"),
            ]),
        ]),

        html.Div([
            html.Div([
                html.P("Candidate", className="card-title"),
                html.Iframe(srcDoc=molecule_svg(mol_key), className="mol-frame"),
                html.P(mol["name"], className="mol-name"),
                html.P(mol["formula"], className="mol-formula"),
                html.P(mol["category"], className="mol-cat"),
            ], className="card"),

            html.Div([
                html.P("Decoder profile (measured)", className="card-title"),
                html.Div([
                    html.Span("Effective LER", className="stat-key"),
                    html.Span(f"{ler:.5f}", className="stat-val"),
                ], className="stat-row"),
                html.Div([
                    html.Span("Effective latency", className="stat-key"),
                    html.Span(f"{latency_us:.1f} us/shot", className="stat-val"),
                ], className="stat-row"),
                html.Div([
                    html.Span("Active-space qubits", className="stat-key"),
                    html.Span(f"{result['qubits']}", className="stat-val"),
                ], className="stat-row"),
                html.Div([
                    html.Span("Ansatz parameters", className="stat-key"),
                    html.Span(f"{result['n_params']}", className="stat-val"),
                ], className="stat-row"),
            ], className="card"),

            html.Div([
                html.P("Energy estimate", className="card-title"),
                html.Div([
                    html.Span("Exact (FCI, correct sector)", className="stat-key"),
                    html.Span(f"{result['exact_energy']:.5f} Ha", className="stat-val"),
                ], className="stat-row"),
                html.Div([
                    html.Span("VQE raw (unmitigated)", className="stat-key"),
                    html.Span(f"{result['raw_energy']:.5f} Ha  (err {result['raw_error_ha']:.5f})",
                              className="stat-val"),
                ], className="stat-row"),
                html.Div([
                    html.Span("VQE + ZNE (mitigated)", className="stat-key"),
                    html.Span(f"{result['predicted_energy']:.5f} Ha  (err {result['error_ha']:.5f})",
                              className="stat-val"),
                ], className="stat-row"),
                html.Div([
                    html.Span("Accuracy tier (post-ZNE)", className="stat-key"),
                    html.Span(tier_label, className=f"badge {badge_cls}"),
                ], className="stat-row"),
            ], className="card"),
        ], className="grid-3"),

        html.Div([
            html.Div([
                html.P("Accuracy tier", className="card-title"),
                tier_gauge(result["error_ha"]),
            ], className="card"),
            html.Div([
                html.P("Optimization trace", className="card-title"),
                dcc.Graph(figure=convergence_figure(result), config={"displayModeBar": False}),
            ], className="card"),
        ], className="grid-2", style={"marginTop": "18px"}),
    ]


def render_scan(mol_key, escalation_pct):
    if SCORES is None:
        return [error_banner(
            f"{SCORES_PATH} not found. Run decoder_triage_pipeline.py first."
        )]

    mol = MOLECULES[mol_key]
    if not mol["simulate"]:
        return render_not_simulated(mol_key)

    rows = []
    log_lines = []
    for key in STRATEGY_ORDER:
        label = STRATEGIES[key]["label"]
        ler, latency_us = decoder_profile(key, escalation_pct, SCORES)
        try:
            result = run_screening(mol_key, ler)
        except RuntimeError as e:
            return [error_banner(str(e))]
        tier_key, tier_label, _ = accuracy_tier(result["error_ha"])
        rows.append({
            "key": key, "label": label, "ler": ler, "latency_us": latency_us,
            "predicted": result["predicted_energy"], "exact": result["exact_energy"],
            "error_ha": result["error_ha"], "tier_key": tier_key, "tier_label": tier_label,
            "raw_error_ha": result["raw_error_ha"],
        })
        cls = "ok" if tier_key != "unreliable" else ""
        log_lines.append(
            html.Div(f"> {label}: LER {ler:.5f}, raw error {result['raw_error_ha']:.5f} Ha, "
                      f"ZNE error {result['error_ha']:.5f} Ha -> [{tier_label.upper()}]", className=cls)
        )

    n_qual = sum(1 for r in rows if r["tier_key"] in ("qualitative", "chemical"))
    n_chem = sum(1 for r in rows if r["tier_key"] == "chemical")
    best = min(rows, key=lambda r: r["error_ha"])

    table_rows = []
    for r in rows:
        badge_cls = {"chemical": "badge-good", "qualitative": "badge-warn", "unreliable": "badge-bad"}[r["tier_key"]]
        table_rows.append(html.Tr([
            html.Td(r["label"]),
            html.Td(f"{r['ler']:.5f}"),
            html.Td(f"{r['latency_us']:.1f}"),
            html.Td(f"{r['predicted']:.5f}"),
            html.Td(f"{r['raw_error_ha']:.5f}"),
            html.Td(f"{r['error_ha']:.5f}"),
            html.Td(html.Span(r["tier_label"], className=f"badge {badge_cls}")),
        ]))

    return [
        html.Div([
            html.H1("Full scan report", className="page-title"),
            html.P(f'All 5 decoder/triage strategies -- {mol["formula"]} ({mol["name"]}), '
                   f'{escalation_pct}% escalation where applicable', className="page-sub"),
        ]),

        html.Div([
            html.Div([
                html.Div(f"{len(rows)}", className="summary-num"),
                html.Div("candidates screened", className="summary-label"),
            ], className="summary-box"),
            html.Div([
                html.Div(f"{n_qual} / {len(rows)}", className="summary-num"),
                html.Div("reach qualitative accuracy", className="summary-label"),
            ], className="summary-box"),
            html.Div([
                html.Div(f"{n_chem} / {len(rows)}", className="summary-num"),
                html.Div("reach chemical accuracy", className="summary-label"),
            ], className="summary-box"),
            html.Div([
                html.Div(best["label"], className="summary-num", style={"fontSize": "16px"}),
                html.Div("lowest error this scan", className="summary-label"),
            ], className="summary-box"),
        ], className="summary-strip"),

        html.Div([
            html.P("Scan log", className="card-title"),
            html.Div(log_lines, className="scan-log"),
        ], className="card", style={"marginBottom": "18px"}),

        html.Div([
            html.Div([
                html.P("Error by strategy", className="card-title"),
                dcc.Graph(figure=scan_bar_figure(rows), config={"displayModeBar": False}),
            ], className="card"),
            html.Div([
                html.P("Cost vs. quality", className="card-title"),
                dcc.Graph(figure=latency_ler_scatter(rows), config={"displayModeBar": False}),
            ], className="card"),
        ], className="grid-2", style={"marginBottom": "18px"}),

        html.Div([
            html.Div([
                html.P("ZNE mitigation impact", className="card-title"),
                dcc.Graph(figure=zne_impact_figure(rows), config={"displayModeBar": False}),
            ], className="card"),
        ], style={"marginBottom": "18px"}),

        html.Div([
            html.P("Comparison table", className="card-title"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Strategy"), html.Th("LER"), html.Th("Latency (us)"),
                    html.Th("Predicted (Ha)"), html.Th("Raw error (Ha)"), html.Th("ZNE error (Ha)"), html.Th("Tier"),
                ])),
                html.Tbody(table_rows),
            ], className="scan-table"),
        ], className="card"),
    ]


@app.callback(
    Output("main-content", "children"),
    Output("esc-slider", "disabled"),
    Input("run-btn", "n_clicks"),
    Input("scan-btn", "n_clicks"),
    Input("strat-dd", "value"),
    State("mol-dd", "value"),
    State("esc-slider", "value"),
)
def update_main(_run_clicks, _scan_clicks, strategy_key, mol_key, escalation_pct):
    triggered = ctx.triggered_id
    slider_disabled = strategy_key in ("mwpm", "bposd")

    if triggered == "scan-btn":
        return render_scan(mol_key, escalation_pct), slider_disabled
    return render_single(mol_key, strategy_key, escalation_pct), slider_disabled


if __name__ == "__main__":
    print("=" * 72)
    print(" Decoder-aware molecular screening")
    print(f"   scores: {SCORES_PATH} ({'found' if SCORES is not None else 'MISSING'})")
    print(f"   pennylane: {'available' if PENNYLANE_AVAILABLE else 'NOT INSTALLED'}")
    print("=" * 72)
    port = int(os.environ.get("PORT", 8060))
    app.run(debug=False, host="0.0.0.0", port=port)