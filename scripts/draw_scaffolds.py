"""
Murcko scaffolds -> A4-sized PNG figures.
pip install rdkit pandas

Usage:
    python draw_scaf.py --csv path/to/data.csv --columns col1 col2 col3
    python draw_scaf.py --csv data.csv --columns smiles --out-dir figures --top-n 40

If --csv / --columns are not given, the DEFAULT_* values below are used.
"""

import argparse
import math
import os
from collections import Counter

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdCoordGen
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold

# ----------------------------------------------------------------------
# DEFAULT SETTINGS (can be overridden from the command line, see --help)
# ----------------------------------------------------------------------
DEFAULT_CSV_PATH = "mannich4ml.csv"
DEFAULT_COLUMNS = ["electrophile", "nucleophile", "organocatalyst_desalted"]
DEFAULT_OUT_DIR = "figures"

MOLS_PER_ROW = 4
TOP_N = None
GENERIC = False
KEEP_STEREO = False            # False -> плоские скаффолды, энантиомеры объединены
SORT_BY = "rows"               # "rows" или "molecules" (влияет только на порядок, не на легенду)

PAGE_W_MM, PAGE_H_MM = 210, 297
MARGIN_MM = 25
DPI = 600

BOND_WIDTH_MM = 0.18           # толщина связи на бумаге
LEGEND_MM = 3.2
LEGEND_FRACTION = 0.12
SCALE_PERCENTILE = 100

RDLogger.DisableLog("rdApp.*")

USABLE_W_MM = PAGE_W_MM - 2 * MARGIN_MM
USABLE_H_MM = PAGE_H_MM - 2 * MARGIN_MM


def mm2px(mm, dpi=DPI):
    return int(round(mm / 25.4 * dpi))


ROWS_PER_PAGE = max(1, int(round(MOLS_PER_ROW * USABLE_H_MM / USABLE_W_MM)))
PER_PAGE = MOLS_PER_ROW * ROWS_PER_PAGE


# ----------------------------------------------------------------------
# SCAFFOLDS
# ----------------------------------------------------------------------
_canon_cache = {}


def canonical(smi):
    if smi not in _canon_cache:
        mol = Chem.MolFromSmiles(str(smi))
        _canon_cache[smi] = Chem.MolToSmiles(mol) if mol is not None else None
    return _canon_cache[smi]


def scaffold_smiles(smi, generic=False, keep_stereo=False):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf.GetNumAtoms() == 0:
        return None
    if generic:
        try:
            scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
        except Exception:
            return None
    if not keep_stereo:
        Chem.RemoveStereochemistry(scaf)      # <- убирает хиральность
    return Chem.MolToSmiles(scaf)


def prepare(smiles_list):
    mols = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        if not KEEP_STEREO:
            Chem.RemoveStereochemistry(m)
        rdCoordGen.AddCoords(m)
        m = rdMolDraw2D.PrepareMolForDrawing(m, addChiralHs=False,
                                             wedgeBonds=KEEP_STEREO)
        mols.append(m)
    return mols


# ----------------------------------------------------------------------
# AUTO SCALE
# ----------------------------------------------------------------------
def geometry(mols):
    extents, bonds = [], []
    for m in mols:
        conf = m.GetConformer()
        pos = [conf.GetAtomPosition(i) for i in range(m.GetNumAtoms())]
        xs, ys = [p.x for p in pos], [p.y for p in pos]
        extents.append(max(max(xs) - min(xs), max(ys) - min(ys)))
        for b in m.GetBonds():
            a, z = pos[b.GetBeginAtomIdx()], pos[b.GetEndAtomIdx()]
            bonds.append(math.hypot(a.x - z.x, a.y - z.y))
    extents.sort()
    k = min(len(extents) - 1, int(len(extents) * SCALE_PERCENTILE / 100))
    bonds.sort()
    med = bonds[len(bonds) // 2] if bonds else 1.5
    return max(extents[k], 1e-6), med


def bond_length_px(mols, cell_px):
    extent, med = geometry(mols)
    usable = cell_px * (1 - LEGEND_FRACTION) * 0.88
    return max(4.0, usable / extent * med)


# ----------------------------------------------------------------------
# RENDER
# ----------------------------------------------------------------------
def configure(opts, cell_px, bond_px):
    ref = mm2px(USABLE_W_MM / MOLS_PER_ROW)
    opts.useBWAtomPalette()
    opts.fixedBondLength = bond_px
    opts.bondLineWidth = max(1.0, mm2px(BOND_WIDTH_MM) * cell_px / ref)
    opts.scaleBondWidth = False
    opts.minFontSize = -1
    opts.maxFontSize = -1
    opts.baseFontSize = 0.7
    opts.legendFontSize = int(mm2px(LEGEND_MM) * cell_px / ref)
    opts.legendFraction = LEGEND_FRACTION
    opts.padding = 0.04
    opts.addStereoAnnotation = KEEP_STEREO
    opts.centreMoleculesBeforeDrawing = True


def render(mols, legends, basename, bond_units):
    ncols, nrows = MOLS_PER_ROW, ROWS_PER_PAGE
    cell = mm2px(USABLE_W_MM) // ncols
    width, height = cell * ncols, cell * nrows

    if max(width, height) >= 30000:
        print(f"  {basename}: canvas too large for Cairo, lower DPI")
        return
    d = rdMolDraw2D.MolDraw2DCairo(width, height, cell, cell)
    configure(d.drawOptions(), cell, bond_units * cell)
    d.DrawMolecules(mols, legends=legends)
    d.FinishDrawing()
    png = d.GetDrawingText()
    if not png:
        print(f"  {basename}: empty PNG buffer")
        return
    with open(f"{basename}.png", "wb") as fh:
        fh.write(png)
    print(f"  -> {os.path.basename(basename)}.png  {len(mols)} structures  "
          f"[{width}x{height} px @ {DPI} dpi = {USABLE_W_MM}x{USABLE_H_MM} mm]")


def draw_pages(mols, legends, basename):
    if not mols:
        print("  nothing to draw")
        return
    cell_ref = mm2px(USABLE_W_MM) // MOLS_PER_ROW
    bond_units = bond_length_px(mols, cell_ref) / cell_ref

    npages = math.ceil(len(mols) / PER_PAGE)
    for p in range(npages):
        lo, hi = p * PER_PAGE, min((p + 1) * PER_PAGE, len(mols))
        name = basename if npages == 1 else f"{basename}_p{p+1}"
        render(mols[lo:hi], legends[lo:hi], name, bond_units)
    if npages > 1:
        print(f"  {npages} A4 pages, {PER_PAGE} per page")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Draw Murcko scaffolds from SMILES columns of a CSV into A4 PNG figures.")
    p.add_argument("--csv", default=DEFAULT_CSV_PATH,
                    help=f"Path to input CSV (default: {DEFAULT_CSV_PATH})")
    p.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS,
                    help=f"One or more column names containing SMILES "
                         f"(default: {DEFAULT_COLUMNS})")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help=f"Output directory for PNGs/CSVs (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--top-n", type=int, default=TOP_N,
                    help="Only draw the top N scaffolds per column (default: all)")
    return p.parse_args()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    csv_path = args.csv
    columns = args.columns
    out_dir = args.out_dir
    top_n = args.top_n

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns not found: {missing}\nAvailable: {list(df.columns)}")

    print(f"grid {MOLS_PER_ROW} x {ROWS_PER_PAGE} = {PER_PAGE} structures per A4 page")

    for col in columns:
        print(f"\n[{col}]")
        raw = df[col].dropna().astype(str)

        scaf_of = {}
        for smi in raw:
            c = canonical(smi)
            if c is not None and c not in scaf_of:
                scaf_of[c] = scaffold_smiles(c, GENERIC, KEEP_STEREO)

        mol_counts = Counter()   # уникальных соединений с этим скаффолдом
        row_counts = Counter()   # строк датасета с этим скаффолдом
        n_acyclic = 0

        for c, scaf in scaf_of.items():
            if scaf is None:
                n_acyclic += 1
            else:
                mol_counts[scaf] += 1

        for smi in raw:
            c = canonical(smi)
            scaf = scaf_of.get(c) if c is not None else None
            if scaf is not None:
                row_counts[scaf] += 1

        key = (lambda s: (row_counts[s], mol_counts[s])) if SORT_BY == "rows" \
            else (lambda s: (mol_counts[s], row_counts[s]))
        order = sorted(mol_counts, key=key, reverse=True)
        if top_n:
            order = order[:top_n]

        mols = prepare(order)
        legends = [f"{i} ({mol_counts[s]})"
                   for i, s in enumerate(order, start=1)][: len(mols)]

        print(f"  rows {len(raw)} | molecules {len(scaf_of)} | "
              f"scaffolds {len(mol_counts)} | acyclic {n_acyclic}")
        draw_pages(mols, legends, os.path.join(out_dir, f"scaffolds_{col}"))

        pd.DataFrame({"num": range(1, len(order) + 1),
                      "scaffold_smiles": order,
                      "n_molecules": [mol_counts[s] for s in order],
                      "n_rows": [row_counts[s] for s in order]}
                     ).to_csv(os.path.join(out_dir, f"scaffolds_{col}.csv"), index=False)


if __name__ == "__main__":
    main()