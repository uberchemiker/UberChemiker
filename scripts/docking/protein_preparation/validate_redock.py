#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_redock.py — проверка КАЧЕСТВА подготовленных рецепторов редокингом.

Идея: берём нативный лиганд (из кристалла), докуем его Vina в подготовленный
нами рецептор и смотрим RMSD топовой позы к нативной. Если нативная поза
воспроизводится (RMSD мал) — карман не испорчен подготовкой.

ВАЖНО про RMSD: считаем БЕЗ суперпозиции (in-place). Рецептор Vina не двигает,
поэтому докированная и нативная позы уже в одной системе координат. Накладывать
их (AlignMol) нельзя — это спрятало бы смещение позы в кармане. Используем
rdMolAlign.CalcRMS: он учитывает симметрию молекулы, но НЕ выравнивает координаты.

Лиганд для докинга готовим через Meeko (mk_prepare_ligand) — родной путь для Vina.

Запуск:
    python validate_redock.py \
        --results run_results_v5 \
        --root /Users/igormorgunov/Myco/Validation/Proteins_Ligands \
        --out redock_results
        [--targets 2vt4 3kk6 4djh ...]   # по умолчанию — все из --results

Ожидается:
    --results/<base>/<base>_clean.pdbqt   (наш подготовленный рецептор)
    --root/<base>/box_config              (бокс Vina)
    --root/<base>/native*.mol2|.pdb       (нативный лиганд из кристалла)

На выходе: redock_summary.csv (target, rmsd, verdict, method, status)
          и рабочие файлы редокинга в --out/<base>/
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys


VINA_CMD = "vina"
OBABEL_CMD = "obabel"
MEEKO_LIG_CMD = "mk_prepare_ligand.py"


def find_native_ligand(src_folder):
    """Найти файл нативного лиганда. Предпочитаем .mol2 (есть порядок связей —
    нужен и Meeko, и для корректного RMSD), потом .sdf, потом .pdb."""
    for name in ("native.mol2", "native_ligand.mol2", "ligand.mol2",
                 "native.sdf", "native_ligand.sdf",
                 "native_ligand.pdb", "native.pdb"):
        p = os.path.join(src_folder, name)
        if os.path.exists(p):
            return p
    return None


def prepare_ligand_meeko(native_file, out_pdbqt, log):
    """Подготовить лиганд .pdbqt через Meeko. Возвращает True/False."""
    cmd = [MEEKO_LIG_CMD, "-i", native_file, "-o", out_pdbqt]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        log.append(f"[ОШИБКА] не найден {MEEKO_LIG_CMD}")
        return False
    except subprocess.TimeoutExpired:
        log.append("[ОШИБКА] Meeko-подготовка лиганда: таймаут")
        return False
    ok = os.path.exists(out_pdbqt) and os.path.getsize(out_pdbqt) > 0
    if not ok:
        log.append("Meeko не подготовил лиганд:\n" +
                   (proc.stdout or "")[-500:] + (proc.stderr or "")[-500:])
    return ok


def run_vina(receptor_pdbqt, ligand_pdbqt, box_config, out_pdbqt, log):
    """Запустить Vina-докинг. Возвращает True/False."""
    cmd = [VINA_CMD, "--receptor", receptor_pdbqt, "--ligand", ligand_pdbqt,
           "--config", box_config, "--out", out_pdbqt]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except FileNotFoundError:
        log.append(f"[ОШИБКА] не найден {VINA_CMD}")
        return False
    except subprocess.TimeoutExpired:
        log.append("[ОШИБКА] Vina: таймаут")
        return False
    ok = os.path.exists(out_pdbqt) and os.path.getsize(out_pdbqt) > 0
    if not ok:
        log.append("Vina не создала выход:\n" +
                   (proc.stdout or "")[-500:] + (proc.stderr or "")[-500:])
    return ok


def extract_first_pose(docked_pdbqt, first_pose_pdbqt):
    """Вытащить первую (лучшую) позу из многомодельного pdbqt Vina."""
    with open(docked_pdbqt) as f:
        lines = f.readlines()
    out = []
    in_model = False
    for line in lines:
        if line.startswith("MODEL"):
            in_model = True
            out = [line]
            continue
        if line.startswith("ENDMDL"):
            out.append(line)
            break
        if in_model or line.startswith(("ATOM", "HETATM", "ROOT", "BRANCH",
                                        "ENDROOT", "ENDBRANCH", "TORSDOF")):
            out.append(line)
    # если MODEL-блоков не было, берём как есть
    if not out:
        out = lines
    with open(first_pose_pdbqt, "w") as f:
        f.writelines(out)
    return first_pose_pdbqt


def to_mol2(in_file, out_mol2, log):
    """Конвертировать в .mol2 через OpenBabel (для чтения RDKit'ом)."""
    try:
        subprocess.run([OBABEL_CMD, in_file, "-O", out_mol2],
                       capture_output=True, text=True, timeout=300, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as e:
        log.append(f"[ОШИБКА] obabel {in_file} -> mol2: {e}")
        return False
    return os.path.exists(out_mol2) and os.path.getsize(out_mol2) > 0


def compute_rmsd_inplace(native_mol2, docked_mol2, log):
    """RMSD без суперпозиции с учётом симметрии (rdMolAlign.CalcRMS).
    Возвращает (rmsd|None, status)."""
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign
    from rdkit.Chem.rdmolops import Kekulize

    native = Chem.MolFromMol2File(native_mol2, sanitize=False, removeHs=True)
    docked = Chem.MolFromMol2File(docked_mol2, sanitize=False, removeHs=True)
    if native is None or docked is None:
        return None, "mol_load_error"

    na, nd = native.GetNumAtoms(), docked.GetNumAtoms()
    if na != nd:
        log.append(f"разное число тяжёлых атомов: native={na}, docked={nd}")
        return None, "atom_mismatch"

    try:
        Kekulize(native, clearAromaticFlags=True)
        Kekulize(docked, clearAromaticFlags=True)
    except Exception:
        pass  # kekulize не критичен

    try:
        # CalcRMS: симметрия учитывается, координаты НЕ выравниваются (in-place)
        rmsd = rdMolAlign.CalcRMS(docked, native)
        return rmsd, "ok"
    except Exception as e:
        log.append(f"CalcRMS не смог сопоставить: {e}")
        return None, "rmsd_error"


def verdict(rmsd):
    if rmsd is None:
        return "н/д"
    if rmsd <= 2.0:
        return "OK (карман воспроизведён)"
    if rmsd <= 3.0:
        return "погранично"
    return "ПЛОХО (поза не воспроизведена)"


def load_methods(results_dir):
    """Подтянуть метод сборки из run_summary.csv, если он есть рядом."""
    methods = {}
    csv_path = os.path.join(results_dir, "run_summary.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                methods[row["name"]] = row.get("method", "")
    return methods


def main():
    ap = argparse.ArgumentParser(description="Редокинг-валидация подготовленных "
                                             "рецепторов (RMSD без суперпозиции).")
    ap.add_argument("--results", required=True,
                    help="папка результатов batch (с <base>/<base>_clean.pdbqt)")
    ap.add_argument("--root", required=True,
                    help="папка исходных структур (с box_config и нативным лигандом)")
    ap.add_argument("--out", default="redock_results", help="папка результатов")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="список структур (по умолчанию — все из --results)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    methods = load_methods(args.results)

    # список целей
    if args.targets:
        targets = args.targets
    else:
        targets = sorted(os.path.basename(d) for d in
                         glob.glob(os.path.join(args.results, "*"))
                         if os.path.isdir(d) and d.rstrip("/").split("/")[-1]
                         not in ("logs",))

    rows = []
    for tgt in targets:
        log = []
        receptor = os.path.join(args.results, tgt, f"{tgt}_clean.pdbqt")
        src = os.path.join(args.root, tgt)
        box = None
        for name in ("box_config", "box_config.txt", "vina_box.conf"):
            p = os.path.join(src, name)
            if os.path.exists(p):
                box = p
                break
        native = find_native_ligand(src)

        status = "ok"
        rmsd = None
        if not os.path.exists(receptor):
            status = "no_receptor"
        elif box is None:
            status = "no_box"
        elif native is None:
            status = "no_native_ligand"
        else:
            wd = os.path.join(args.out, tgt)
            os.makedirs(wd, exist_ok=True)
            lig_pdbqt = os.path.join(wd, "native_prepared.pdbqt")
            docked = os.path.join(wd, "docked_out.pdbqt")
            first = os.path.join(wd, "docked_first.pdbqt")
            native_mol2 = os.path.join(wd, "native_ref.mol2")
            docked_mol2 = os.path.join(wd, "docked_ref.mol2")

            if not prepare_ligand_meeko(native, lig_pdbqt, log):
                status = "ligand_prep_failed"
            elif not run_vina(receptor, lig_pdbqt, box, docked, log):
                status = "vina_failed"
            else:
                extract_first_pose(docked, first)
                ok1 = to_mol2(native, native_mol2, log)
                ok2 = to_mol2(first, docked_mol2, log)
                if not (ok1 and ok2):
                    status = "convert_failed"
                else:
                    rmsd, status = compute_rmsd_inplace(native_mol2,
                                                        docked_mol2, log)

        v = verdict(rmsd)
        rows.append({"target": tgt,
                     "rmsd": f"{rmsd:.3f}" if rmsd is not None else "",
                     "verdict": v, "method": methods.get(tgt, ""),
                     "status": status})
        rmsd_str = f"{rmsd:.2f} Å" if rmsd is not None else "—"
        print(f"  {tgt:6}  RMSD={rmsd_str:9}  {v:32}  "
              f"[{methods.get(tgt,'?')}]  {status}")
        # сохранить лог по структуре
        if log:
            with open(os.path.join(args.out, f"{tgt}_redock.log"), "w") as f:
                f.write("\n".join(log))

    # сводный CSV
    csv_path = os.path.join(args.out, "redock_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "rmsd", "verdict",
                                          "method", "status"])
        w.writeheader()
        w.writerows(rows)

    # сводка
    ok = [r for r in rows if r["status"] == "ok" and r["rmsd"]
          and float(r["rmsd"]) <= 2.0]
    border = [r for r in rows if r["rmsd"] and 2.0 < float(r["rmsd"]) <= 3.0]
    bad = [r for r in rows if r["rmsd"] and float(r["rmsd"]) > 3.0]
    failed = [r for r in rows if r["status"] != "ok"]

    print("\n" + "=" * 55)
    print("СВОДКА РЕДОКИНГА")
    print("=" * 55)
    print(f"  OK (RMSD<=2Å):       {len(ok)}")
    print(f"  погранично (2-3Å):   {len(border)}")
    print(f"  плохо (>3Å):         {len(bad)}")
    print(f"  не посчитано:        {len(failed)}")
    if ok:
        # сравнить качество по методам сборки
        from collections import defaultdict
        by_m = defaultdict(lambda: [0, 0])
        for r in rows:
            if r["rmsd"]:
                by_m[r["method"]][0] += 1
                if float(r["rmsd"]) <= 2.0:
                    by_m[r["method"]][1] += 1
        print("\n  Качество по методам сборки (OK/всего):")
        for m, (tot, good) in sorted(by_m.items()):
            print(f"    {m or '?':24}: {good}/{tot}")
    print(f"\nРезультаты: {csv_path}")


if __name__ == "__main__":
    main()
