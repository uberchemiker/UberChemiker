#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_run.py — ПОЛНЫЙ пакетный прогон пайплайна (Стадии 0-5) по всем структурам.

В отличие от batch_triage.py (который останавливается после Стадии 2),
этот раннер доводит каждую структуру до .pdbqt в неинтерактивном режиме и
собирает итог: каким методом собрано (Meeko / Meeko без кофактора / OpenBabel),
прошла ли валидация, или структура прервана / требует ручного разбора.

Запуск:
    python batch_run.py \
        --root /Users/igormorgunov/Myco/Validation/Proteins_Ligands \
        --script /Users/igormorgunov/protein_prepare/prepare_receptor.py \
        --out run_results \
        --fallback openbabel       # или abort (по умолчанию)

В каждой подпапке ожидается один .cif и файл box_config.

На выходе (в --out):
    run_summary.csv          — строка на структуру (метод, валидация, статус, pdbqt)
    logs/<name>.log          — полный вывод пайплайна
    <name>/                  — рабочая папка с результатами структуры
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from collections import defaultdict


def find_inputs(folder):
    cifs = glob.glob(os.path.join(folder, "*.cif")) + \
        glob.glob(os.path.join(folder, "*.mmcif"))
    box = None
    for name in ("box_config", "box_config.txt", "vina_box.conf",
                 "box_config.conf"):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            box = p
            break
    return (cifs[0] if cifs else None), box


_RE_METHOD = re.compile(r"метод сборки:\s*(\S+)")
_RE_VALID = re.compile(r"валидация формата:\s*(.+)")
_RE_PDBQT = re.compile(r"Итоговый рецептор для докинга:\s*(\S+)")
_RE_DONE = re.compile(r"ГОТОВО — пайплайн завершён")
_RE_LIG = re.compile(r"\[auto-ligand\] выбран лиганд '(\S+)'")
_RE_RDKIT_REPAIR = re.compile(r"После починки RDKit ругается")
_RE_PADDING = re.compile(r"padding/межостаточной|excess inter-residue|paddings for")
_RE_ROLLBACK = re.compile(r"\[ОТКАТ\]")
_RE_RDKIT_STAGE2 = re.compile(r"Meeko упал на RDKit-ошибке")
_RE_ABORTED = re.compile(r"прерывания|aborted|прервано|Прервано")
_RE_HEADER_METHOD = re.compile(r"Метод:\s*(\S+)")
_RE_RES = re.compile(r"Разрешение:\s*(\S+)")


def detect_failure(text, returncode):
    for needle, reason in [
        ("unrecognized arguments", "старая версия скрипта (нет флагов)"),
        ("ModuleNotFoundError", "не тот Python (нет модуля)"),
        ("No module named", "не тот Python (нет модуля)"),
        ("[TIMEOUT]", "таймаут"),
    ]:
        if needle in text:
            return reason
    if returncode not in (0, 2, 3) and "СТАДИЯ 0" not in text:
        return f"подпроцесс упал (код {returncode})"
    return None


def parse_run(text):
    d = {"method": "", "validation": "", "pdbqt": "", "status": "unknown",
         "ligand": "", "exp_method": "", "resolution": "", "note": ""}
    m = _RE_HEADER_METHOD.search(text)
    if m:
        d["exp_method"] = m.group(1)
    m = _RE_RES.search(text)
    if m:
        d["resolution"] = m.group(1)
    m = _RE_LIG.search(text)
    if m:
        d["ligand"] = m.group(1)

    if _RE_DONE.search(text):
        d["status"] = "done"
        m = _RE_METHOD.search(text)
        if m:
            d["method"] = m.group(1)
        m = _RE_VALID.search(text)
        if m:
            d["validation"] = m.group(1).strip()
        m = _RE_PDBQT.search(text)
        if m:
            d["pdbqt"] = m.group(1)
        # пометить, что была починка-откат (rdkit/padding -> исходный белок)
        if _RE_ROLLBACK.search(text):
            d["note"] = "после отката починки"
    elif _RE_RDKIT_REPAIR.search(text):
        d["status"] = "stop_rdkit_after_repair"
    elif _RE_PADDING.search(text):
        d["status"] = "stop_padding"
    elif _RE_RDKIT_STAGE2.search(text):
        d["status"] = "stop_rdkit_stage2"
    elif "aborted" in text or "Прервано" in text or "прервано" in text:
        d["status"] = "aborted"
    return d


def main():
    ap = argparse.ArgumentParser(description="Полный пакетный прогон (0-5).")
    ap.add_argument("--root", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--out", default="run_results")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--fallback", default="abort",
                    choices=["abort", "openbabel"],
                    help="что делать при падении Meeko в неинтерактиве")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logs_dir = os.path.join(args.out, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    folders = sorted([d for d in glob.glob(os.path.join(args.root, "*"))
                      if os.path.isdir(d)])
    print(f"Найдено подпапок: {len(folders)}  (fallback={args.fallback})")

    rows = []
    for folder in folders:
        name = os.path.basename(folder)
        cif, box = find_inputs(folder)
        if not cif or not box:
            print(f"  [skip] {name}: нет .cif/box_config")
            rows.append({"name": name, "status": "no_input", "method": "",
                         "validation": "", "exp_method": "", "resolution": "",
                         "ligand": "", "pdbqt": ""})
            continue

        out_sub = os.path.join(args.out, name)
        os.makedirs(out_sub, exist_ok=True)
        cmd = [args.python, args.script, "-i", cif, "-c", box, "-o", out_sub,
               "--non-interactive", "--auto-ligand",
               "--noninteractive-fallback", args.fallback]
        retcode = -1
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=1800)
            text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            retcode = proc.returncode
        except subprocess.TimeoutExpired:
            text = "[TIMEOUT]"
        with open(os.path.join(logs_dir, f"{name}.log"), "w") as f:
            f.write(text)

        fail = detect_failure(text, retcode)
        if fail:
            rows.append({"name": name, "status": "error", "method": "",
                         "validation": "", "exp_method": "", "resolution": "",
                         "ligand": "", "pdbqt": ""})
            print(f"  [ERROR ] {name}: {fail}")
            continue

        d = parse_run(text)
        rows.append({"name": name, "status": d["status"],
                     "method": d["method"], "validation": d["validation"],
                     "exp_method": d["exp_method"], "resolution": d["resolution"],
                     "ligand": d["ligand"], "pdbqt": d["pdbqt"]})
        tag = d["status"] if d["status"] != "done" else d["method"]
        print(f"  [{tag:24}] {name}  {d['exp_method']} {d['resolution']}Å  "
              f"лиганд={d['ligand'] or '?'}  "
              f"{'валид:' + d['validation'] if d['validation'] else ''}")

    # CSV
    csv_path = os.path.join(args.out, "run_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name", "status", "method", "validation", "exp_method",
            "resolution", "ligand", "pdbqt"])
        w.writeheader()
        w.writerows(rows)

    # сводка
    by_status = defaultdict(int)
    by_method = defaultdict(int)
    for r in rows:
        by_status[r["status"]] += 1
        if r["status"] == "done":
            by_method[r["method"]] += 1

    print("\n" + "=" * 55)
    print("СВОДКА ПОЛНОГО ПРОГОНА")
    print("=" * 55)
    for st, n in sorted(by_status.items()):
        print(f"  {st:26}: {n}")
    if by_method:
        print("\n  методы сборки (среди done):")
        for mth, n in sorted(by_method.items()):
            print(f"    {mth:24}: {n}")
    print(f"\nРезультаты: {csv_path}")
    print(f"           {logs_dir}/")

    # подсветить, что требует внимания
    attention = [r["name"] for r in rows
                 if r["status"] in ("stop_rdkit_after_repair",
                                    "stop_rdkit_stage2", "stop_padding",
                                    "aborted", "error", "manual_needed")]
    if attention:
        print(f"\nТребуют ручного разбора ({len(attention)}): "
              f"{', '.join(attention)}")
    ob = [r["name"] for r in rows if r["method"] == "openbabel"]
    if ob:
        print(f"\nСобраны через OpenBabel ({len(ob)}) — проверьте качество: "
              f"{', '.join(ob)}")
    nocof = [r["name"] for r in rows
             if "dropped_cofactor" in (r["method"] or "")]
    if nocof:
        print(f"\nСобраны БЕЗ кофактора ({len(nocof)}) — кофактор потерян, "
              f"нужна ручная подготовка если важен: {', '.join(nocof)}")


if __name__ == "__main__":
    main()
