#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_triage.py — пакетный ТРИАЖ структур (Стадии 0-2, без сборки, без вопросов).

Назначение: прогнать пайплайн в неинтерактивном режиме по всем подпапкам с
структурами, собрать сводную картину проблем ДО полной подготовки:
  - метод/разрешение каждой структуры,
  - какие HET-коды встретились и как классифицированы,
  - какие коды оказались НЕИЗВЕСТНЫМИ (требуют разовой классификации),
  - сколько проблемных остатков (всего/близко/далеко к боксу),
  - статус Стадии 2: clean / bad_residues / rdkit_error / прочее.

Запуск:
    python batch_triage.py \
        --root /Users/igormorgunov/Myco/Validation/Proteins_Ligands \
        --script /Users/igormorgunov/protein_prepare/prepare_receptor.py \
        --out triage_results

В каждой подпапке ожидается один .cif и файл box_config.

На выходе (в --out):
    triage_summary.csv     — строка на структуру
    unknown_het_codes.txt  — уникальные неизвестные HET-коды (частота + примеры)
    logs/<name>.log        — полный вывод пайплайна по каждой структуре
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
    """Вернуть (cif_path, box_path) или (None, None), если чего-то нет."""
    cifs = glob.glob(os.path.join(folder, "*.cif")) + \
        glob.glob(os.path.join(folder, "*.mmcif"))
    box = None
    for name in ("box_config", "box_config.txt", "vina_box.conf",
                 "box_config.conf"):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            box = p
            break
    cif = cifs[0] if cifs else None
    return cif, box


# Регексы для парсинга вывода пайплайна
_RE_METHOD = re.compile(r"Метод:\s*(\S+)")
_RE_RES = re.compile(r"Разрешение:\s*(\S+)")
_RE_HET = re.compile(r"^\s+(\S+)\s+копий=(\d+)\s+мин\.расст\.до бокса=\s*([\d.]+).*->\s*(\w+)")
_RE_AUTO_UNKNOWN = re.compile(r"\[auto\] неизвестный (\S+) -> buffer")
_RE_AUTO_LIG = re.compile(r"\[auto-ligand\] выбран лиганд '(\S+)'")
_RE_MEEKO_OK = re.compile(r"Meeko не нашёл проблемных остатков")
_RE_MEEKO_BAD = re.compile(r"Meeko сообщил о (\d+) проблемных")
_RE_CLOSE = re.compile(r"Близко к боксу \(чинить\):\s*(\d+)")
_RE_FAR = re.compile(r"Далеко \(удалить -a\):\s*(\d+)")
_RE_RDKIT = re.compile(r"RDKit ругается|Explicit valence|rdkit_error")


def detect_subprocess_failure(text, returncode):
    """Распознать явный сбой запуска пайплайна (а не нормальный результат).
    Возвращает строку-причину или None."""
    markers = [
        ("unrecognized arguments", "argparse: старая версия скрипта? "
         "нет нужных флагов"),
        ("ModuleNotFoundError", "не найден Python-модуль (gemmi и т.п.) — "
         "не тот интерпретатор"),
        ("No module named", "не найден Python-модуль — не тот интерпретатор"),
        ("command not found", "команда не найдена"),
        ("Traceback (most recent call last)", "необработанное исключение"),
        ("[TIMEOUT]", "превышен лимит времени"),
    ]
    for needle, reason in markers:
        if needle in text:
            return reason
    # ненулевой код возврата И не дошли даже до Стадии 0
    if returncode not in (0, 2, 3) and "СТАДИЯ 0" not in text:
        return f"подпроцесс завершился с кодом {returncode} без вывода Стадии 0"
    return None


def parse_log(text):
    """Извлечь из вывода пайплайна сводку по структуре."""
    d = {
        "method": "", "resolution": "", "status": "unknown",
        "n_bad": 0, "n_close": 0, "n_far": 0,
        "het": [], "unknown_codes": [], "auto_ligand": "",
    }
    m = _RE_METHOD.search(text)
    if m:
        d["method"] = m.group(1)
    m = _RE_RES.search(text)
    if m:
        d["resolution"] = m.group(1)
    m = _RE_AUTO_LIG.search(text)
    if m:
        d["auto_ligand"] = m.group(1)

    for line in text.splitlines():
        hm = _RE_HET.match(line)
        if hm:
            d["het"].append((hm.group(1), int(hm.group(2)),
                             float(hm.group(3)), hm.group(4)))
    d["unknown_codes"] = _RE_AUTO_UNKNOWN.findall(text)

    if _RE_RDKIT.search(text):
        d["status"] = "rdkit_error"
    elif _RE_MEEKO_OK.search(text):
        d["status"] = "clean"
    else:
        m = _RE_MEEKO_BAD.search(text)
        if m:
            d["status"] = "bad_residues"
            d["n_bad"] = int(m.group(1))
    m = _RE_CLOSE.search(text)
    if m:
        d["n_close"] = int(m.group(1))
    m = _RE_FAR.search(text)
    if m:
        d["n_far"] = int(m.group(1))
    return d


def main():
    ap = argparse.ArgumentParser(description="Пакетный триаж структур (Стадии 0-2).")
    ap.add_argument("--root", required=True,
                    help="корневая папка с подпапками структур")
    ap.add_argument("--script", required=True,
                    help="путь к prepare_receptor.py")
    ap.add_argument("--out", default="triage_results", help="папка результатов")
    ap.add_argument("--python", default=sys.executable,
                    help="каким Python запускать пайплайн "
                         "(по умолчанию текущий)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logs_dir = os.path.join(args.out, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    folders = sorted([d for d in glob.glob(os.path.join(args.root, "*"))
                      if os.path.isdir(d)])
    print(f"Найдено подпапок: {len(folders)}")

    rows = []
    unknown_freq = defaultdict(list)   # код -> список структур

    for folder in folders:
        name = os.path.basename(folder)
        cif, box = find_inputs(folder)
        if not cif or not box:
            print(f"  [skip] {name}: нет .cif или box_config "
                  f"(cif={bool(cif)}, box={bool(box)})")
            rows.append({"name": name, "status": "no_input",
                         "method": "", "resolution": "", "n_bad": "",
                         "n_close": "", "n_far": "", "auto_ligand": "",
                         "het_summary": "", "unknown_codes": ""})
            continue

        out_sub = os.path.join(args.out, name)
        os.makedirs(out_sub, exist_ok=True)
        cmd = [args.python, args.script, "-i", cif, "-c", box,
               "-o", out_sub, "--non-interactive", "--auto-ligand",
               "--stop-after", "stage2"]
        retcode = -1
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=1200)
            text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            retcode = proc.returncode
        except subprocess.TimeoutExpired:
            text = "[TIMEOUT]"
        # сохранить полный лог
        with open(os.path.join(logs_dir, f"{name}.log"), "w") as f:
            f.write(text)

        # сначала проверить на явный сбой запуска (а не нормальный результат)
        failure = detect_subprocess_failure(text, retcode)
        if failure:
            rows.append({"name": name, "status": "error", "method": "",
                         "resolution": "", "n_bad": "", "n_close": "",
                         "n_far": "", "auto_ligand": "", "het_summary": "",
                         "unknown_codes": ""})
            print(f"  [ERROR       ] {name}: {failure}")
            continue

        d = parse_log(text)
        for code in set(d["unknown_codes"]):
            unknown_freq[code].append(name)

        het_summary = "; ".join(f"{c}:{cat}({n})"
                                for c, n, dist, cat in d["het"])
        rows.append({
            "name": name, "status": d["status"],
            "method": d["method"], "resolution": d["resolution"],
            "n_bad": d["n_bad"], "n_close": d["n_close"], "n_far": d["n_far"],
            "auto_ligand": d["auto_ligand"],
            "het_summary": het_summary,
            "unknown_codes": ",".join(sorted(set(d["unknown_codes"]))),
        })
        print(f"  [{d['status']:12}] {name}  "
              f"{d['method']} {d['resolution']}Å  "
              f"bad={d['n_bad']} (близко={d['n_close']}/далеко={d['n_far']})  "
              f"лиганд={d['auto_ligand'] or '?'}"
              + (f"  НЕИЗВЕСТНЫЕ: {','.join(set(d['unknown_codes']))}"
                 if d["unknown_codes"] else ""))

    # --- сводный CSV ---
    csv_path = os.path.join(args.out, "triage_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name", "status", "method", "resolution",
            "n_bad", "n_close", "n_far", "auto_ligand",
            "het_summary", "unknown_codes"])
        w.writeheader()
        w.writerows(rows)

    # --- список неизвестных кодов ---
    unk_path = os.path.join(args.out, "unknown_het_codes.txt")
    with open(unk_path, "w") as f:
        f.write("# Неизвестные HET-коды по всем структурам (для разовой "
                "классификации).\n")
        f.write("# Формат: КОД  частота  примеры_структур\n")
        f.write("# Решение дописывайте в user_het_dictionary.txt как:  КОД = "
                "cofactor|buffer|ion_candidate\n\n")
        for code in sorted(unknown_freq, key=lambda c: -len(unknown_freq[c])):
            structs = unknown_freq[code]
            ex = ", ".join(structs[:5]) + ("..." if len(structs) > 5 else "")
            f.write(f"{code:6}  {len(structs):3}  {ex}\n")

    # --- сводка по статусам ---
    by_status = defaultdict(int)
    for r in rows:
        by_status[r["status"]] += 1

    print("\n" + "=" * 55)
    print("СВОДКА ТРИАЖА")
    print("=" * 55)
    for st, n in sorted(by_status.items()):
        print(f"  {st:14}: {n}")
    print(f"\n  Уникальных неизвестных HET-кодов: {len(unknown_freq)}")
    print(f"\nРезультаты:")
    print(f"  {csv_path}")
    print(f"  {unk_path}")
    print(f"  {logs_dir}/")
    if unknown_freq:
        print("\nСледующий шаг: расклассифицировать неизвестные коды из "
              f"{os.path.basename(unk_path)}\n"
              "  и дописать в user_het_dictionary.txt, затем полный прогон.")


if __name__ == "__main__":
    main()
