#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_receptor.py  —  пайплайн подготовки белка к докингу.
ИНКРЕМЕНТ 2: чтение .pdb И .cif через gemmi.
             Стадия 0 (анализ структуры + чтение бокса)
             Стадия 1 (очистка воды + инвентаризация и классификация HETATM)

Запуск (из папки со скриптами):
    python3 prepare_receptor.py -i structure.cif -c vina_box.conf [-l LIG] [-o out/]

Поддерживаются оба формата входа: .pdb и .cif (mmCIF). Формат определяется
gemmi автоматически.

На выходе:
    <name>_protein.pdb        — чистая белковая цепь (без воды и гетероатомов)
    <name>_cofactors.pdb      — сохранённые кофакторы (могут вернуться позже)
    <name>_ions.pdb           — сохранённые ион-кандидаты (с расстоянием до бокса)
    <name>_ligand_ref.pdb     — извлечённый референс-лиганд (если указан/выбран)
    <name>_prep_report.txt    — лог всех принятых решений

Зависимости: gemmi  (pip install gemmi  /  conda install -c conda-forge gemmi)
"""

import argparse
import ast
import math
import os
import re
import subprocess
import sys

try:
    import gemmi
except ImportError:
    sys.exit(
        "[ОШИБКА] Не найден модуль gemmi для текущего интерпретатора:\n"
        f"    {sys.executable}\n"
        "Похоже, скрипт запущен не тем Python, в котором установлен gemmi.\n"
        "Запустите через Python из вашего conda-окружения, например:\n"
        "    $CONDA_PREFIX/bin/python prepare_receptor.py ...\n"
        "или установите gemmi: conda install -c conda-forge gemmi")

# Конфиг ищем рядом со скриптом, чтобы запуск работал из любой папки.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep_config as cfg

# Глобальный флаг неинтерактивного режима (ставится в main по флагу --non-interactive).
NON_INTERACTIVE = False
# Поведение fallback в неинтерактиве при падении Meeko: "abort" | "openbabel".
FALLBACK_MODE = "abort"


# ======================================================================
#  СТАДИЯ 0a:  Header-инфо из сырого текста (надёжнее, чем из gemmi)
# ======================================================================

def _norm_method(txt):
    if "X-RAY" in txt:
        return "X-RAY"
    if "ELECTRON MICROSCOPY" in txt or "CRYO" in txt or txt == "EM":
        return "CRYO-EM"
    if "NMR" in txt:
        return "NMR"
    return txt or "UNKNOWN"


def analyze_header_text(path):
    """Прочитать метод и разрешение из сырого текста файла (.pdb или .cif).
    gemmi не всегда вытаскивает эти поля из header, поэтому парсим сами."""
    method = "UNKNOWN"
    resolution = None
    is_cif = path.lower().endswith((".cif", ".mmcif"))

    try:
        with open(path, "r", errors="ignore") as f:
            text_lines = f.readlines()
    except Exception:
        return {"method": method, "resolution": resolution}

    if is_cif:
        for line in text_lines:
            low = line.strip().lower()
            if low.startswith("_exptl.method"):
                parts = line.split(None, 1)
                if len(parts) > 1:
                    method = _norm_method(parts[1].strip().strip("'\"").upper())
            elif low.startswith("_refine.ls_d_res_high"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        resolution = float(parts[1])
                    except ValueError:
                        pass
            elif (low.startswith("_em_3d_reconstruction.resolution")
                  and resolution is None):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        resolution = float(parts[1])
                    except ValueError:
                        pass
    else:  # PDB
        for line in text_lines:
            if line.startswith("EXPDTA"):
                method = _norm_method(line[10:].strip().upper())
            elif line.startswith("REMARK   2 RESOLUTION"):
                after = line.split("RESOLUTION", 1)[1]
                for tok in after.split():
                    tok = tok.rstrip(".")
                    try:
                        resolution = float(tok)
                        break
                    except ValueError:
                        continue

    return {"method": method, "resolution": resolution}


# ======================================================================
#  Чтение бокса из Vina-конфига
# ======================================================================

def read_box(conf_path):
    keys = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
    vals = {}
    with open(conf_path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            k, v = [s.strip() for s in line.split("=", 1)]
            if k in keys:
                try:
                    vals[k] = float(v)
                except ValueError:
                    pass
    missing = [k for k in keys if k not in vals]
    if missing:
        sys.exit(f"[ОШИБКА] В конфиге бокса не хватает полей: {missing}")
    center = (vals["center_x"], vals["center_y"], vals["center_z"])
    size = (vals["size_x"], vals["size_y"], vals["size_z"])
    return center, size


def min_dist_to_box(coords, center, size):
    """Мин. расстояние (Å) от точек до параллелепипеда; 0, если внутри."""
    half = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
    lower = (center[0] - half[0], center[1] - half[1], center[2] - half[2])
    upper = (center[0] + half[0], center[1] + half[1], center[2] + half[2])
    best = float("inf")
    for (x, y, z) in coords:
        dx = max(lower[0] - x, 0.0, x - upper[0])
        dy = max(lower[1] - y, 0.0, y - upper[1])
        dz = max(lower[2] - z, 0.0, z - upper[2])
        best = min(best, math.sqrt(dx * dx + dy * dy + dz * dz))
    return best


# ======================================================================
#  Словарь пользовательских решений
# ======================================================================

def _user_dict_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        cfg.USER_DICT_FILE)


def load_user_dictionary():
    extra = {}
    path = _user_dict_path()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                code, cat = [s.strip() for s in line.split("=", 1)]
                extra[code.upper()] = cat.lower()
    return extra


def remember_decision(code, category):
    with open(_user_dict_path(), "a") as f:
        f.write(f"{code} = {category}\n")


def classify_het(resname, user_dict):
    rn = resname.upper()
    if rn in cfg.WATER_CODES:
        return "water"
    if rn in user_dict:
        return user_dict[rn]
    if rn in cfg.COFACTOR_CODES:
        return "cofactor"
    if rn in cfg.BUFFER_CODES:
        return "buffer"
    if rn in cfg.CATALYTIC_ION_CANDIDATES:
        return "ion_candidate"
    return "unknown"


def ask_unknown(resname, dist, count):
    if NON_INTERACTIVE:
        print(f"  [auto] неизвестный {resname} -> buffer (неинтерактивный режим)")
        return "buffer"
    print()
    print(f"  [?] Неизвестный гетерокод: {resname}  "
          f"(копий: {count}, мин. расстояние до бокса: {dist:.1f} Å)")
    print("      К какой категории отнести?")
    print("        [c] кофактор  (можно вернуть в рецептор, если близко к боксу)")
    print("        [b] буфер/мусор  (удалить навсегда)")
    print("        [l] целевой лиганд-референс  (извлечь для бокса)")
    print("        [i] каталитический ион/металл  (решим по близости)")
    print("        [k] оставить как есть в рецепторе  (не трогать)")
    mapping = {"c": "cofactor", "b": "buffer", "l": "ligand",
               "i": "ion_candidate", "k": "keep"}
    while True:
        ans = input("      Ваш выбор [c/b/l/i/k]: ").strip().lower()
        if ans in mapping:
            category = mapping[ans]
            if category in ("cofactor", "buffer", "ion_candidate"):
                rem = input("      Запомнить решение для будущих запусков? "
                            "[y/N]: ").strip().lower()
                if rem == "y":
                    remember_decision(resname.upper(), category)
            return category
        print("      Не понял ввод, повторите.")


# ======================================================================
#  Сборка/запись групп остатков через gemmi (надёжный PDB-формат)
# ======================================================================

def residues_to_structure(residue_entries, cell=None, spacegroup=None):
    """Собрать gemmi.Structure из списка (chain_name, gemmi.Residue).
    Остатки одной цепи группируются в одну gemmi-цепь."""
    st = gemmi.Structure()
    if cell is not None:
        st.cell = cell
    if spacegroup:
        st.spacegroup_hm = spacegroup
    model = gemmi.Model("1")
    chains = {}  # name -> gemmi.Chain
    for chain_name, res in residue_entries:
        cn = chain_name if chain_name else "A"
        if cn not in chains:
            chains[cn] = gemmi.Chain(cn)
        chains[cn].add_residue(res)
    for cn in chains:
        model.add_chain(chains[cn])
    st.add_model(model)
    return st


def write_structure_pdb(path, residue_entries, cell=None, spacegroup=None):
    """Записать список (chain_name, residue) в PDB через gemmi."""
    if not residue_entries:
        return
    st = residues_to_structure(residue_entries, cell, spacegroup)
    st.setup_entities()
    st.write_pdb(path)


# ======================================================================
#  СТАДИЯ 1
# ======================================================================

def run_propka_for_his(protein_pdb, outdir, base, report):
    """Запустить PROPKA на белке, вернуть {(chain, resnum): pKa} для His.
    Пусто, если PROPKA недоступна/упала."""
    pka_map = {}
    workdir = os.path.dirname(os.path.abspath(protein_pdb))
    cmd = [sys.executable, "-m", "propka", os.path.basename(protein_pdb)]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       cwd=workdir)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        report.append("  [His] PROPKA недоступна/таймаут — близкие His "
                      "получат дефолт.")
        return pka_map

    pka_file = os.path.splitext(protein_pdb)[0] + ".pka"
    if not os.path.exists(pka_file):
        report.append("  [His] PROPKA не создала .pka — близкие His получат дефолт.")
        return pka_map

    re_his = re.compile(r"^\s*HIS\s+(\d+)\s+(\w)\s+([\d.]+)\s+[\d.]+", re.M)
    try:
        with open(pka_file) as f:
            text = f.read()
        # секция SUMMARY (после строки 'SUMMARY OF THIS PREDICTION')
        if "SUMMARY OF THIS PREDICTION" in text:
            text = text.split("SUMMARY OF THIS PREDICTION", 1)[1]
        for m in re_his.finditer(text):
            resnum, chain, pka = int(m.group(1)), m.group(2), float(m.group(3))
            pka_map[(chain, resnum)] = pka
    except OSError:
        pass
    return pka_map


def tautomerize_his(structure, center, size, outdir, base, report):
    """Переименовать HIS в конкретный таутомер, чтобы Meeko не падал на ничьей.
    Дальние от бокса -> DEFAULT_HIS. Близкие -> по PROPKA (HIP если pKa высокий,
    иначе DEFAULT_HIS). Изменяет structure на месте, возвращает статистику."""
    model = structure[0]
    cutoff = cfg.POCKET_PROXIMITY_CUTOFF

    his_list = []   # (chain_name, res, dist)
    for chain in model:
        for res in chain:
            if res.name.upper() in ("HIS", "HSD", "HSE", "HSP"):
                coords = [(a.pos.x, a.pos.y, a.pos.z) for a in res]
                d = min_dist_to_box(coords, center, size)
                his_list.append((chain.name, res, d))

    if not his_list:
        return {"total": 0, "near": 0, "renamed": {}}

    near = [(c, r, d) for c, r, d in his_list if d <= cutoff]
    pka_map = {}
    if near and cfg.HIS_PROPKA_FOR_POCKET:
        # PROPKA нужен сырой белок на диске — пишем временный
        tmp_pdb = os.path.join(outdir, f"{base}_for_propka.pdb")
        st_copy = structure.clone()
        st_copy.setup_entities()
        st_copy.write_pdb(tmp_pdb)
        report.append(f"  [His] запускаю PROPKA для {len(near)} близких к боксу His")
        pka_map = run_propka_for_his(tmp_pdb, outdir, base, report)

    renamed = {}
    for chain_name, res, d in his_list:
        if d <= cutoff and (chain_name, res.seqid.num) in pka_map:
            pka = pka_map[(chain_name, res.seqid.num)]
            new = "HIP" if pka > cfg.HIS_HIP_PKA_CUTOFF else cfg.DEFAULT_HIS
            tag = f"{chain_name}:{res.seqid.num}(близкий, pKa={pka:.1f})"
        elif d <= cutoff:
            new = cfg.DEFAULT_HIS
            tag = f"{chain_name}:{res.seqid.num}(близкий, PROPKA н/д)"
        else:
            new = cfg.DEFAULT_HIS
            tag = None
        res.name = new
        renamed[new] = renamed.get(new, 0) + 1
        if tag:
            report.append(f"    His {tag} -> {new}")

    msg = (f"  [His] переименовано {len(his_list)} His "
           f"(близких к боксу: {len(near)}); итог: "
           + ", ".join(f"{k}={v}" for k, v in sorted(renamed.items())))
    print(msg)
    report.append(msg)
    return {"total": len(his_list), "near": len(near), "renamed": renamed}


def autodetect_ligand(structure, center, size):
    """Авто-детект кода лиганда: не-словарный HETATM в боксе (dist<=2 Å)
    с наибольшим числом атомов. Возвращает (resname|None, n_candidates)."""
    if len(structure) == 0:
        return None, 0
    user_dict = load_user_dictionary()
    model = structure[0]
    cand = {}   # resname -> (max_atoms, min_dist)
    for chain in model:
        for res in chain:
            rn = res.name.upper()
            if res.het_flag != "H" or rn in cfg.MODIFIED_RESIDUES:
                continue
            cat = classify_het(rn, user_dict)
            if cat in ("water", "buffer", "cofactor", "ion_candidate"):
                continue  # точно не лиганд (вода/буфер/кофактор/ион)
            coords = [(a.pos.x, a.pos.y, a.pos.z) for a in res]
            d = min_dist_to_box(coords, center, size)
            if d <= 2.0:   # в боксе
                n_atoms = len(res)
                if rn not in cand or n_atoms > cand[rn][0]:
                    cand[rn] = (n_atoms, d)
    if not cand:
        return None, 0
    # выбрать кандидата с наибольшим числом атомов
    best = max(cand.items(), key=lambda kv: kv[1][0])
    return best[0], len(cand)


def stage1_clean(structure, center, size, ligand_code, report):
    """Очистить воду, классифицировать гетероатомы.
    Возвращает: protein_entries, saved(dict) — оба как списки (chain_name, residue)."""
    user_dict = load_user_dictionary()
    model = structure[0]

    protein_entries = []
    saved = {"cofactor": [], "ion_candidate": [], "ligand": [], "keep": []}
    report.append("=== СТАДИЯ 1: классификация гетероатомов ===")

    het_by_name = {}

    for chain in model:
        for res in chain:
            rn = res.name.upper()
            is_modified = rn in cfg.MODIFIED_RESIDUES
            if res.het_flag != "H" or is_modified:
                protein_entries.append((chain.name, res))
            else:
                het_by_name.setdefault(res.name, []).append((chain.name, res))

    for resname, entries in sorted(het_by_name.items()):
        coords = [(a.pos.x, a.pos.y, a.pos.z)
                  for _, res in entries for a in res]
        dist = min_dist_to_box(coords, center, size)
        n = len(entries)

        if ligand_code and resname.upper() == ligand_code.upper():
            category = "ligand"
        else:
            category = classify_het(resname, user_dict)
            if category == "unknown":
                category = ask_unknown(resname, dist, n)

        msg = (f"  {resname:>4}  копий={n:<3} "
               f"мин.расст.до бокса={dist:6.1f} Å  -> {category}")
        report.append(msg)
        print(msg)

        if category in ("water", "buffer"):
            continue
        saved[category].extend(entries)

    return protein_entries, saved


# ======================================================================
#  СТАДИЯ 2:  Meeko-диагностика голого белка
# ======================================================================

# Регекс для строк вида:  No template matched for residue_key='A:387'
_RE_RESKEY = re.compile(r"residue_key='([^']+)'")
# Регекс для финальной строки: Template matching failed for: ['A:387', ...]
_RE_FAILLIST = re.compile(r"Template matching failed for:\s*(\[[^\]]*\])")


def run_meeko_diagnostic(protein_pdb, outprefix):
    """Запустить Meeko на голом белке БЕЗ -a (чтобы он отчитался о проблемах).
    Возвращает (returncode, combined_output). Сырой вывод также пишется в
    <outprefix>_meeko.txt для последующей диагностики."""
    cmd = [cfg.MEEKO_CMD, "-i", protein_pdb, "-o", outprefix,
           "-p", "--default_altloc", cfg.DEFAULT_ALTLOC]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        return None, (f"[ОШИБКА] Не найдена команда '{cfg.MEEKO_CMD}'. "
                      "Проверьте, что Meeko установлен и в PATH.")
    except subprocess.TimeoutExpired:
        return None, "[ОШИБКА] Meeko-диагностика превысила лимит времени."
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    try:
        with open(outprefix + "_meeko.txt", "w") as f:
            f.write(f"$ {' '.join(cmd)}\nreturncode={proc.returncode}\n\n{out}")
    except OSError:
        pass
    return proc.returncode, out


def parse_meeko_output(text):
    """Разобрать вывод Meeko.
    Возвращает dict:
       status: 'clean' | 'bad_residues' | 'rdkit_error' | 'padding_error'
               | 'unknown_error'
       bad_residues: список ключей вида 'A:387'
    """
    # 1) RDKit-ошибка валентности — отдельный случай (по договорённости СТОП)
    if "Explicit valence" in text or "AtomValenceException" in text:
        return {"status": "rdkit_error", "bad_residues": []}

    # 1b) His-неоднозначность (ничья HIE/HID/HIP) — Meeko не может выбрать таутомер
    if ("tied for fewest" in text
            or ("have passed" in text and ("HIE" in text or "HID" in text
                                           or "HIP" in text))):
        return {"status": "his_ambiguous", "bad_residues": []}

    # 2) Финальный список проблемных остатков
    bad = []
    m = _RE_FAILLIST.search(text)
    if m:
        try:
            bad = [str(x) for x in ast.literal_eval(m.group(1))]
        except (ValueError, SyntaxError):
            bad = []

    # подстраховка: собрать ключи из строк "No template matched..."
    if not bad:
        bad = sorted(set(_RE_RESKEY.findall(text)))

    if bad:
        return {"status": "bad_residues", "bad_residues": bad}

    # 3) Padding / межостаточные связи (дисульфиды, артефакты после удаления
    #    дальних остатков): "Expected N paddings", "excess inter-residue bond"
    if ("paddings for" in text or "excess inter-residue bond" in text
            or "Expected" in text and "padding" in text):
        return {"status": "padding_error", "bad_residues": []}

    # 4) Прочие ошибки Meeko (не template, не RDKit, не padding)
    if ("Error:" in text or "Traceback" in text) and "Files written" not in text:
        return {"status": "unknown_error", "bad_residues": []}

    return {"status": "clean", "bad_residues": []}


def build_residue_coord_index(protein_pdb):
    """Прочитать белок и построить {(chain, resnum, icode): [(x,y,z), ...]}."""
    st = gemmi.read_structure(protein_pdb)
    index = {}
    for chain in st[0]:
        for res in chain:
            key = (chain.name, res.seqid.num, res.seqid.icode.strip())
            coords = [(a.pos.x, a.pos.y, a.pos.z) for a in res]
            index.setdefault(key, []).extend(coords)
    return index


def split_reskey(reskey):
    """'A:387' -> ('A', 387, '')   |  'A:387B' (icode) -> ('A', 387, 'B')."""
    chain, _, rest = reskey.partition(":")
    icode = ""
    num_str = rest
    if rest and not (rest.lstrip("-").isdigit()):
        # есть insertion code на конце
        i = 0
        while i < len(rest) and (rest[i].isdigit() or (i == 0 and rest[i] == "-")):
            i += 1
        num_str, icode = rest[:i], rest[i:]
    try:
        num = int(num_str)
    except ValueError:
        num = None
    return chain, num, icode


def classify_bad_residues(bad_keys, protein_pdb, center, size, report):
    """Разложить bad residues на близкие к боксу (чинить) и далёкие (под -a)."""
    index = build_residue_coord_index(protein_pdb)
    cutoff = cfg.POCKET_PROXIMITY_CUTOFF

    to_fix, to_remove, not_found = [], [], []
    report.append("\n=== СТАДИЯ 2: классификация проблемных остатков ===")
    report.append(f"порог близости к боксу: {cutoff} Å (вариант 1: по любому атому)")

    for key in bad_keys:
        chain, num, icode = split_reskey(key)
        coords = index.get((chain, num, icode))
        if coords is None:
            coords = index.get((chain, num, ""))  # запасной поиск без icode
        if not coords:
            not_found.append(key)
            report.append(f"  {key:<10} — атомы не найдены в белке (?)")
            continue
        d = min_dist_to_box(coords, center, size)
        if d <= cutoff:
            to_fix.append(key)
            report.append(f"  {key:<10} расст.={d:6.1f} Å  -> ПОЧИНИТЬ (близко)")
        else:
            to_remove.append(key)
            report.append(f"  {key:<10} расст.={d:6.1f} Å  -> удалить через -a (далеко)")

    return {"fix": to_fix, "remove": to_remove, "not_found": not_found}


def stage2_diagnose(protein_pdb, outdir, base, center, size, report):
    """Полная Стадия 2. Возвращает dict с планом или None при необходимости СТОП."""
    print("\nСТАДИЯ 2 — Meeko-диагностика голого белка")
    tmp_prefix = os.path.join(outdir, f"{base}_meeko_diag")
    rc, out = run_meeko_diagnostic(protein_pdb, tmp_prefix)

    if rc is None:
        # инструмент не найден / таймаут
        print(out)
        report.append(f"\n[СТАДИЯ 2] {out}")
        return {"status": "tool_error"}

    parsed = parse_meeko_output(out)
    status = parsed["status"]

    if status == "his_ambiguous":
        msg = ("  [СТОП] Meeko не смог выбрать таутомер His (ничья HIE/HID/HIP),\n"
               "  несмотря на таутомеризацию. На финале сработает fallback "
               "на OpenBabel.")
        print(msg)
        report.append("\n=== СТАДИЯ 2 ===\n" + msg)
        return {"status": "his_ambiguous"}

    if status == "padding_error":
        msg = ("  [инфо] Meeko на голом белке дал padding/межостаточную ошибку\n"
               "  (дисульфид/стык). На финале сработает fallback на OpenBabel.")
        print(msg)
        report.append("\n=== СТАДИЯ 2 ===\n" + msg)
        return {"status": "padding_error"}

    if status == "rdkit_error":
        msg = ("  [СТОП] Meeko упал на RDKit-ошибке валентности (Explicit valence).\n"
               "  Это типично для cryo-EM структур с искажённой геометрией боковых\n"
               "  цепей. Список bad residues так получить нельзя.\n"
               "  Этот случай мы договорились разбирать отдельно — пайплайн\n"
               "  останавливается. Возможные пути (обсудим): минимизация геометрии\n"
               "  через OpenMM, либо сборка через OpenBabel в обход RDKit.")
        print(msg)
        report.append("\n=== СТАДИЯ 2 ===\n" + msg)
        return {"status": "rdkit_error"}

    if status == "unknown_error":
        msg = ("  [СТОП] Meeko завершился ошибкой, не распознанной как template/RDKit.\n"
               "  Загляните в вывод выше / лог. Пайплайн остановлен.")
        print(msg)
        report.append("\n=== СТАДИЯ 2 ===\n" + msg)
        report.append("Сырой вывод Meeko:\n" + out)
        return {"status": "unknown_error", "raw": out}

    if status == "clean":
        msg = ("  [OK] Meeko не нашёл проблемных остатков на голом белке.\n"
               "  Починка не требуется — можно идти к финальной сборке.")
        print(msg)
        report.append("\n=== СТАДИЯ 2 ===\n" + msg)
        return {"status": "clean", "fix": [], "remove": [], "not_found": []}

    # status == 'bad_residues'
    print(f"  Meeko сообщил о {len(parsed['bad_residues'])} проблемных остатках.")
    plan = classify_bad_residues(parsed["bad_residues"], protein_pdb,
                                 center, size, report)
    plan["status"] = "bad_residues"
    print(f"  Близко к боксу (чинить):  {len(plan['fix'])}")
    print(f"  Далеко (удалить -a):      {len(plan['remove'])}")
    if plan["not_found"]:
        print(f"  Не найдены в белке:       {len(plan['not_found'])}")
    if plan["fix"]:
        print(f"    остатки на починку: {', '.join(plan['fix'])}")
    return plan


# ======================================================================
#  СТАДИЯ 2.5:  Починка (PDBFixer достраивает ТОЛЬКО близкие; дальние под -a)
# ======================================================================

def delete_far_residues(in_pdb, out_pdb, far_keys):
    """Удалить из структуры дальние проблемные остатки (список ключей 'A:387').
    Делаем это ДО PDBFixer, чтобы он не достроил их криво и не уронил RDKit."""
    st = gemmi.read_structure(in_pdb)
    targets = set()
    for key in far_keys:
        chain, num, icode = split_reskey(key)
        targets.add((chain, num, icode))

    model = st[0]
    removed = 0
    for chain in model:
        to_del = []
        for i, res in enumerate(chain):
            k = (chain.name, res.seqid.num, res.seqid.icode.strip())
            if k in targets:
                to_del.append(i)
        for i in reversed(to_del):
            del chain[i]
            removed += 1

    st.setup_entities()
    st.write_pdb(out_pdb)
    return removed


def run_pdbfixer(in_pdb, out_pdb, report, only_residues=None):
    """Достроить недостающие тяжёлые атомы боковых цепей через PDBFixer.
    only_residues: множество ключей 'A:198' — чинить ТОЛЬКО эти остатки
    (близкие к боксу). Дальние не трогаем (их кривая достройка ломала RDKit).
    Целые отсутствующие остатки (петли) не достраиваем. Водороды НЕ добавляем.
    Возвращает (ok, info, n_fixed)."""
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError:
        return False, ("[ОШИБКА] Не найден PDBFixer/OpenMM. Установите:\n"
                       "    conda install -c conda-forge pdbfixer openmm"), 0

    try:
        fixer = PDBFixer(filename=in_pdb)
        fixer.findMissingResidues()

        if not cfg.REBUILD_MISSING_LOOPS:
            n_loops = len(fixer.missingResidues)
            fixer.missingResidues = {}
            if n_loops:
                report.append(f"  PDBFixer: пропущено {n_loops} участков "
                              f"отсутствующих остатков (петли не достраиваем)")
        else:
            kept = {}
            for k, v in fixer.missingResidues.items():
                if len(v) <= cfg.MAX_LOOP_REBUILD:
                    kept[k] = v
            dropped = len(fixer.missingResidues) - len(kept)
            fixer.missingResidues = kept
            if dropped:
                report.append(f"  PDBFixer: пропущено {dropped} длинных петель "
                              f"(> {cfg.MAX_LOOP_REBUILD} остатков)")

        fixer.findMissingAtoms()

        # Фильтр: чинить ТОЛЬКО близкие к боксу остатки. Дальние оставляем
        # обрезанными (backbone цел) — они уйдут под -a, а их достройка PDBFixer'ом
        # давала кривую геометрию и RDKit-ошибки.
        if only_residues is not None:
            kept_atoms = {}
            skipped = 0
            for res, atoms in fixer.missingAtoms.items():
                key = f"{res.chain.id}:{res.id}"
                if key in only_residues:
                    kept_atoms[res] = atoms
                else:
                    skipped += 1
            fixer.missingAtoms = kept_atoms
            if skipped:
                report.append(f"  PDBFixer: пропущено {skipped} дальних остатков "
                              f"(не достраиваем — уйдут под -a)")

        n_fixed = len(fixer.missingAtoms)
        fixer.addMissingAtoms()

        PDBFile.writeFile(fixer.topology, fixer.positions,
                          open(out_pdb, "w"), keepIds=True)
        report.append(f"  PDBFixer: достроены боковые цепи у {n_fixed} остатков")
        return True, f"достроено остатков: {n_fixed}", n_fixed
    except Exception as e:
        return False, f"[ОШИБКА] PDBFixer упал: {e}", 0


def stage25_repair(protein_pdb, outdir, base, plan, center, size, report):
    """Стадия 2.5: достроить недостающие тяжёлые атомы через PDBFixer (близкие
    к боксу обязательно сохраняем достроенными; дальние PDBFixer тоже может
    достроить — это безвредно, они уйдут под -a на финале). НЕ удаляем дальние
    физически (это рвало backbone и валило Meeko на padding). Затем повторная
    Meeko-проверка. Возвращает (repaired_pdb_path, status)."""
    print("\nСТАДИЯ 2.5 — починка (достройка близких; дальние не удаляем)")
    report.append("\n=== СТАДИЯ 2.5: починка ===")

    close = plan.get("fix", [])
    far = plan.get("remove", [])
    report.append(f"  близких к боксу: {len(close)}, дальних: {len(far)} "
                  f"(дальние НЕ удаляем — уйдут под -a с сохранением backbone)")

    # PDBFixer достраивает ТОЛЬКО близкие к боксу остатки (точечно).
    # Дальние не трогаем: ни удаления (рвёт backbone -> padding), ни достройки
    # (кривая геометрия -> RDKit). Дальние уйдут под -a с целым backbone.
    close_keys = set(close)
    if close:
        print(f"  Достраиваю ТОЛЬКО близкие к боксу остатки ({len(close)}) "
              f"через PDBFixer; дальние не трогаем...")
    repaired = os.path.join(outdir, f"{base}_repaired.pdb")
    ok, info, n_fixed = run_pdbfixer(protein_pdb, repaired, report,
                                     only_residues=close_keys)
    if not ok:
        print("  " + info)
        report.append("  " + info)
        return None, "fixer_error"
    print(f"  PDBFixer: {info}")

    # 3) повторная Meeko-проверка починенного белка
    print("  Повторная Meeko-проверка починенного белка...")
    tmp_prefix = os.path.join(outdir, f"{base}_meeko_recheck")
    rc, out = run_meeko_diagnostic(repaired, tmp_prefix)
    if rc is None:
        print("  " + out); report.append("  " + out)
        return repaired, "tool_error"

    parsed = parse_meeko_output(out)
    status = parsed["status"]

    if status == "rdkit_error":
        if n_fixed > 0:
            cause = (f"PDBFixer достроил {n_fixed} близких остаток(ов), и геометрия "
                     "одного\n  из них не принимается RDKit.")
        else:
            cause = ("PDBFixer ничего не достраивал — RDKit-проблема была в исходной\n"
                     "  геометрии (искажённые боковые цепи) и проявилась на проверке.")
        msg = ("  [инфо] После починки RDKit ругается на валентность.\n"
               f"  Причина: {cause}\n"
               "  Близкие остатки сохраняем достроенными; финал — через OpenBabel.")
        print(msg); report.append(msg)
        return repaired, "rdkit_error"

    if status == "padding_error":
        msg = ("  [инфо] После починки Meeko видит padding/межостаточную связь\n"
               "  (обычно дисульфидный мостик рядом с проблемной зоной).\n"
               "  Не форсируем OpenBabel — Meeko с -a на финале обычно справляется.")
        print(msg); report.append(msg)
        return repaired, "padding_error"

    if status == "bad_residues":
        # остались проблемные дальние остатки — они уйдут под -a (backbone цел)
        leftover = parsed["bad_residues"]
        msg = (f"  После починки осталось проблемных остатков: {len(leftover)} "
               f"(уйдут под -a на финальной сборке)")
        print(msg); report.append(msg)
        return repaired, "ok_with_a"

    if status == "clean":
        print("  [OK] После починки Meeko не видит проблемных остатков.")
        report.append("  [OK] После починки белок чистый.")
        return repaired, "clean"

    return repaired, "unknown"


# ======================================================================
#  СТАДИЯ 3:  Возврат близких кофакторов/ионов в рецептор
# ======================================================================

def ask_return(resname, category, n_close, dist):
    """Checkpoint: вернуть ли близкий кофактор/ион в рецептор. По умолчанию НЕТ."""
    if NON_INTERACTIVE:
        print(f"  [auto] {resname} не возвращён (неинтерактивный режим)")
        return False
    label = "кофактор" if category == "cofactor" else "ион/металл"
    print()
    print(f"  [?] {label} {resname}: {n_close} копий в пределах "
          f"{cfg.HETERO_RETURN_CUTOFF} Å от бокса (ближайшая {dist:.1f} Å).")
    print(f"      Вернуть в рецептор? Если это детергент/случайная молекула,")
    print(f"      возврат ИСПОРТИТ карман. По умолчанию — НЕ возвращать.")
    ans = input("      Вернуть? [y/N]: ").strip().lower()
    return ans == "y"


def stage3_return_heteroatoms(repaired_pdb, saved, center, size,
                              outdir, base, report):
    """Решить, какие кофакторы/ионы вернуть в рецептор, и собрать финальный
    рецептор (белок + возвращённые гетероатомы). Возвращает путь к рецептору."""
    print("\nСТАДИЯ 3 — возврат кофакторов/ионов к боксу")
    report.append("\n=== СТАДИЯ 3: возврат гетероатомов ===")
    cutoff = cfg.HETERO_RETURN_CUTOFF
    returned = []   # список (chain_name, res) к возврату

    for category in ("cofactor", "ion_candidate"):
        entries = saved.get(category, [])
        if not entries:
            continue
        # сгруппировать по resname
        by_name = {}
        for chain_name, res in entries:
            by_name.setdefault(res.name, []).append((chain_name, res))

        for resname, group in sorted(by_name.items()):
            # расстояние каждой копии до бокса; отобрать близкие
            close = []
            nearest = float("inf")
            for chain_name, res in group:
                coords = [(a.pos.x, a.pos.y, a.pos.z) for a in res]
                d = min_dist_to_box(coords, center, size)
                nearest = min(nearest, d)
                if d <= cutoff:
                    close.append((chain_name, res))

            if not close:
                msg = (f"  {resname} ({category}): все копии дальше {cutoff} Å "
                       f"(ближайшая {nearest:.1f} Å) — не возвращаем")
                print(msg); report.append(msg)
                continue

            # есть близкие копии — спросить подтверждение
            if ask_return(resname, category, len(close), nearest):
                returned.extend(close)
                msg = (f"  {resname} ({category}): возвращено {len(close)} копий "
                       f"в рецептор")
            else:
                msg = (f"  {resname} ({category}): пользователь отказался от возврата")
            print(msg); report.append(msg)

    # собрать финальный рецептор: репарированный белок + возвращённые гетероатомы
    receptor_path = os.path.join(outdir, f"{base}_receptor.pdb")
    st = gemmi.read_structure(repaired_pdb)
    model = st[0]
    # индекс цепей по имени
    chain_by_name = {ch.name: ch for ch in model}
    added = 0
    for chain_name, res in returned:
        cn = chain_name if chain_name else "A"
        if cn not in chain_by_name:
            new_chain = gemmi.Chain(cn)
            model.add_chain(new_chain)
            chain_by_name[cn] = model[cn]
        chain_by_name[cn].add_residue(res)
        added += 1

    st.setup_entities()
    st.write_pdb(receptor_path)

    if added:
        print(f"  Финальный рецептор: белок + {added} возвращённых гетероатомов")
        report.append(f"  возвращено гетероатомов всего: {added}")
    else:
        print("  Финальный рецептор: только белок (ничего не возвращено)")
        report.append("  возвращено гетероатомов: 0 (чистый белок)")
    report.append(f"  файл рецептора: {receptor_path}")
    return receptor_path, added


# ======================================================================
#  СТАДИЯ 4:  Финальная сборка .pdbqt
# ======================================================================

def run_meeko_assembly(receptor_pdb, out_prefix):
    """Финальная сборка через Meeko с -a. Возвращает (success, output, pdbqt)."""
    pdbqt = out_prefix + ".pdbqt"
    if os.path.exists(pdbqt):
        os.remove(pdbqt)
    cmd = [cfg.MEEKO_CMD, "-i", receptor_pdb, "-o", out_prefix,
           "-p", "--default_altloc", cfg.DEFAULT_ALTLOC, "-a"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except FileNotFoundError:
        return False, f"[ОШИБКА] Не найдена команда '{cfg.MEEKO_CMD}'.", None
    except subprocess.TimeoutExpired:
        return False, "[ОШИБКА] Meeko-сборка превысила лимит времени.", None
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    try:
        with open(out_prefix + "_meeko_assembly.txt", "w") as f:
            f.write(f"$ {' '.join(cmd)}\nreturncode={proc.returncode}\n\n{out}")
    except OSError:
        pass
    success = (proc.returncode == 0 and os.path.exists(pdbqt)
               and os.path.getsize(pdbqt) > 0)
    return success, out, (pdbqt if success else None)


def run_openbabel_assembly(receptor_pdb, pdbqt_path):
    """Fallback-сборка через OpenBabel. Возвращает (success, output, pdbqt)."""
    if os.path.exists(pdbqt_path):
        os.remove(pdbqt_path)
    cmd = [cfg.OBABEL_CMD, receptor_pdb, "-O", pdbqt_path, "-xr",
           "-p", str(cfg.PROTONATION_PH), "--partialcharge", "gasteiger"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except FileNotFoundError:
        return False, f"[ОШИБКА] Не найдена команда '{cfg.OBABEL_CMD}'.", None
    except subprocess.TimeoutExpired:
        return False, "[ОШИБКА] OpenBabel-сборка превысила лимит времени.", None
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    success = os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0
    return success, out, (pdbqt_path if success else None)


def _ask(prompt, choices, default):
    """Спросить выбор из choices (множество строк), вернуть выбор или default."""
    if NON_INTERACTIVE:
        return default
    while True:
        ans = input(prompt).strip()
        if ans == "":
            return default
        if ans in choices:
            return ans
        print(f"      Введите один из {sorted(choices)} или Enter ({default}).")


def stage4_assemble(receptor_pdb, protein_only_pdb, returned_count,
                    outdir, base, report, force_openbabel=False):
    """Финальная сборка с ветвлением по кофактору.
    force_openbabel=True: починка близких остатков не прошла RDKit-проверку
    Meeko — пробовать Meeko бесполезно, идём сразу к OpenBabel на починенном
    рецепторе (близкие остатки сохранены достроенными, кофакторы включены).
    Возвращает (pdbqt_path | None, method, status)."""
    print("\nСТАДИЯ 4 — финальная сборка .pdbqt")
    report.append("\n=== СТАДИЯ 4: финальная сборка ===")
    out_prefix = os.path.join(outdir, f"{base}_clean")

    if force_openbabel:
        print("  Починка близких остатков не прошла RDKit-проверку Meeko.")
        print("  Близкие к боксу остатки достроены и СОХРАНЕНЫ; собираю рецептор")
        print("  (с возвращёнными кофакторами, если есть) через OpenBabel.")
        report.append("  починка не прошла RDKit -> сборка починенного через "
                      "OpenBabel (близкие сохранены)")
        if NON_INTERACTIVE:
            if FALLBACK_MODE == "openbabel":
                res = _do_openbabel(receptor_pdb, out_prefix, base, report,
                                    note="починенный, близкие сохранены")
                if res[2] == "failed" and returned_count > 0:
                    return _assemble_without_cofactor(protein_only_pdb,
                                                      out_prefix, base, report)
                return res
            print("  [auto] неинтерактив: прервать (fallback=abort).")
            report.append("  [auto] прервано (fallback=abort)")
            return None, "aborted", "aborted"
        # интерактив: спросить
        print("    [1] Собрать починенный белок через OpenBabel "
              "(близкие остатки сохранены)")
        print("    [2] Прервать и выбрать другую структуру")
        choice = _ask("    Ваш выбор [1/2] (Enter = 2): ", {"1", "2"}, "2")
        if choice == "1":
            res = _do_openbabel(receptor_pdb, out_prefix, base, report,
                                note="починенный, близкие сохранены")
            if res[2] == "failed" and returned_count > 0:
                return _assemble_without_cofactor(protein_only_pdb,
                                                  out_prefix, base, report)
            return res
        print("  Прервано пользователем.")
        report.append("  пользователь прервал (после провала починки)")
        return None, "aborted", "aborted"

    # 1) основная попытка — Meeko на рецепторе (с кофакторами, если есть)
    print("  Пробую Meeko (с -a)...")
    ok, out, pdbqt = run_meeko_assembly(receptor_pdb, out_prefix)
    if ok:
        print(f"  [OK] Meeko собрал рецептор: {pdbqt}")
        report.append(f"  успех: Meeko -> {pdbqt}")
        return pdbqt, "meeko", "ok"

    # Meeko упал. Разбираем причину.
    parsed = parse_meeko_output(out)
    print(f"  [!] Meeko не справился (status={parsed['status']}).")
    report.append(f"  Meeko не справился: status={parsed['status']}")

    # 2) ветвление: есть ли возвращённые кофакторы/ионы?
    if returned_count > 0:
        print("\n  В рецепторе есть возвращённый(е) кофактор/ион.")
        print("  Вероятная причина сбоя — именно он (Meeko плохо переваривает")
        print("  нестандартные гетеромолекулы).")
        print("    [1] Кофактор ВАЖЕН  -> собрать через OpenBabel (обходной путь)")
        print("    [2] Кофактор НЕ критичен -> выкинуть его и пересобрать Meeko")
        choice = _ask("    Ваш выбор [1/2] (Enter = 2): ", {"1", "2"}, "2")

        if choice == "2":
            # выкинуть кофактор -> Meeko на белке без кофакторов
            print("\n  Убираю кофактор, пересобираю белок через Meeko...")
            report.append("  пользователь: выкинуть кофактор, пересобрать Meeko")
            ok2, out2, pdbqt2 = run_meeko_assembly(protein_only_pdb, out_prefix)
            if ok2:
                print(f"  [OK] Meeko собрал белок без кофактора: {pdbqt2}")
                report.append(f"  успех: Meeko без кофактора -> {pdbqt2}")
                return pdbqt2, "meeko_no_cofactor", "ok"
            # Meeko опять упал — значит виноват НЕ кофактор, а белок
            parsed2 = parse_meeko_output(out2)
            print(f"  [!] Meeko опять не справился (status={parsed2['status']}).")
            print("  Значит, причина НЕ в кофакторе, а в самом белке "
                  "(вероятно cryo-EM геометрия).")
            report.append(f"  Meeko без кофактора тоже упал: {parsed2['status']} "
                          "-> причина в белке")
            return _offer_obabel_or_abort(protein_only_pdb, out_prefix,
                                          base, report, with_cofactor=False)
        else:
            # кофактор важен -> OpenBabel с кофактором
            print("\n  Собираю рецептор (с кофактором) через OpenBabel...")
            report.append("  пользователь: кофактор важен -> OpenBabel")
            res = _do_openbabel(receptor_pdb, out_prefix, base, report,
                                note="с кофактором")
            # если и OpenBabel не смог с кофактором (гем/металл) -> без кофактора
            if res[2] == "failed":
                return _assemble_without_cofactor(protein_only_pdb, out_prefix,
                                                  base, report)
            return res

    # 3) кофакторов нет — причина в самом белке
    if parsed["status"] == "rdkit_error":
        print("  Причина: RDKit-ошибка валентности (геометрия белка, типично cryo-EM).")
        report.append("  причина: RDKit valence (белок)")
    elif parsed["status"] == "padding_error":
        print("  Причина: padding/межостаточная связь (дисульфид или артефакт "
              "стыков).")
        report.append("  причина: padding/inter-residue bond (белок)")
    return _offer_obabel_or_abort(receptor_pdb, out_prefix, base, report,
                                  with_cofactor=False)


def _offer_obabel_or_abort(receptor_pdb, out_prefix, base, report, with_cofactor):
    """Развилка без кофактора: OpenBabel или прервать. По умолчанию прервать."""
    if NON_INTERACTIVE:
        if FALLBACK_MODE == "openbabel":
            print("  [auto] неинтерактив: fallback на OpenBabel.")
            report.append("  [auto] fallback на OpenBabel (неинтерактив)")
            note = "с кофактором" if with_cofactor else "белок"
            return _do_openbabel(receptor_pdb, out_prefix, base, report, note=note)
        print("  [auto] неинтерактив: прервать (fallback=abort).")
        report.append("  [auto] прервано (неинтерактив, fallback=abort)")
        return None, "aborted", "aborted"
    print("\n  Meeko не смог собрать этот белок.")
    print("    [1] Собрать через OpenBabel (обходной путь, без RDKit-санитизации)")
    print("    [2] Прервать и выбрать другую структуру этого белка")
    choice = _ask("    Ваш выбор [1/2] (Enter = 2): ", {"1", "2"}, "2")
    if choice == "1":
        note = "с кофактором" if with_cofactor else "белок"
        return _do_openbabel(receptor_pdb, out_prefix, base, report, note=note)
    print("  Прервано пользователем. Попробуйте другую структуру этого белка.")
    report.append("  пользователь выбрал прервать (другая структура)")
    return None, "aborted", "aborted"


def _do_openbabel(receptor_pdb, out_prefix, base, report, note=""):
    """Выполнить OpenBabel-сборку с предупреждением."""
    print("  [предупреждение] OpenBabel — менее предпочтительный путь, "
          "чем Meeko.")
    pdbqt = out_prefix + ".pdbqt"
    ok, out, path = run_openbabel_assembly(receptor_pdb, pdbqt)
    if ok:
        print(f"  [OK] OpenBabel собрал ({note}): {path}")
        report.append(f"  успех: OpenBabel ({note}) -> {path}")
        return path, "openbabel", "ok"
    print("  [ОШИБКА] OpenBabel тоже не справился.")
    print("  " + out.strip()[:300])
    report.append("  OpenBabel не справился")
    return None, "openbabel", "failed"


def _assemble_without_cofactor(protein_only_pdb, out_prefix, base, report):
    """ПОСЛЕДНИЙ РУБЕЖ: сборка с кофактором провалилась обоими методами
    (типичный случай — гем/металл-порфирин, который ломает и Meeko, и OpenBabel).
    Пробуем собрать БЕЗ кофактора. В неинтерактиве — авто+пометка; в интерактиве
    — спрашиваем (дефолт прервать). Возвращает (pdbqt|None, method, status)."""
    print("\n  [!] Сборка С КОФАКТОРОМ не удалась ни Meeko, ни OpenBabel.")
    print("      Вероятно, кофактор — гем/металл-порфирин (ломает оба сборщика).")

    if NON_INTERACTIVE:
        print("  [auto] неинтерактив: собираю БЕЗ кофактора (кофактор потерян).")
        report.append("  [auto] сборка без кофактора (кофактор не переварился)")
        build = True
    else:
        print("    [1] Собрать рецептор БЕЗ кофактора (кофактор будет потерян!)")
        print("    [2] Прервать — кофактор критичен, нужна ручная подготовка")
        build = _ask("    Ваш выбор [1/2] (Enter = 2): ", {"1", "2"}, "2") == "1"

    if not build:
        print("  Прервано: кофактор критичен, требуется ручная подготовка.")
        report.append("  прервано: кофактор требует ручной подготовки")
        return None, "manual_needed", "manual_needed"

    # пробуем без кофактора: сначала Meeko, потом OpenBabel
    print("  Сборка без кофактора: Meeko...")
    ok, out, pdbqt = run_meeko_assembly(protein_only_pdb, out_prefix)
    method = "meeko_dropped_cofactor"
    if not ok:
        print("  Meeko без кофактора не смог, пробую OpenBabel...")
        ok, out, pdbqt = run_openbabel_assembly(protein_only_pdb,
                                                out_prefix + ".pdbqt")
        method = "openbabel_dropped_cofactor"

    if ok:
        warn = ("  [ВНИМАНИЕ] Рецептор собран БЕЗ кофактора — кофактор не удалось\n"
                "  включить ни одним сборщиком. Если кофактор (гем/металл) важен\n"
                "  для кармана, этой структуре нужна РУЧНАЯ подготовка кофактора\n"
                "  (правильные заряды металла/координация).")
        print(f"  [OK] {method} -> {pdbqt}")
        print(warn)
        report.append(f"  успех БЕЗ кофактора: {method} -> {pdbqt}")
        report.append(warn)
        return pdbqt, method, "ok_no_cofactor"

    print("  [ОШИБКА] Не удалось собрать даже без кофактора.")
    report.append("  провал: не собралось даже без кофактора")
    return None, "failed", "failed"


# ======================================================================
#  СТАДИЯ 5:  Валидация формата .pdbqt
# ======================================================================

def stage5_validate(pdbqt_path, returned_resnames, report):
    """Проверить формат .pdbqt без редокинга. Возвращает True/False."""
    print("\nСТАДИЯ 5 — валидация формата")
    report.append("\n=== СТАДИЯ 5: валидация ===")
    ok = True

    if not pdbqt_path or not os.path.exists(pdbqt_path):
        print("  [FAIL] Файл .pdbqt не создан.")
        report.append("  FAIL: нет файла")
        return False

    size = os.path.getsize(pdbqt_path)
    with open(pdbqt_path) as f:
        lines = f.readlines()
    atom_lines = [l for l in lines if l.startswith(("ATOM", "HETATM"))]

    print(f"  Файл: {pdbqt_path} ({size} байт)")
    print(f"  Атомных строк: {len(atom_lines)}")
    report.append(f"  файл={pdbqt_path} размер={size} атомов={len(atom_lines)}")

    if size < 200 or len(atom_lines) == 0:
        print("  [FAIL] Файл подозрительно мал / нет атомов.")
        report.append("  FAIL: пустой/маленький файл")
        return False

    # Проверка AutoDock-типов: последний токен ATOM-строки .pdbqt — тип атома
    bad_type = 0
    for l in atom_lines[:200]:
        toks = l.split()
        if not toks:
            bad_type += 1
            continue
        atype = toks[-1]
        if not (1 <= len(atype) <= 2 and atype[0].isalpha()):
            bad_type += 1
    if bad_type > 0:
        print(f"  [WARN] У {bad_type} строк (из первых 200) сомнительный "
              "AutoDock-тип в последней колонке.")
        report.append(f"  WARN: {bad_type} строк с сомнительным типом")
    else:
        print("  [OK] AutoDock-типы атомов проставлены.")
        report.append("  OK: атомные типы на месте")

    # Проверка сохранности возвращённых гетероатомов
    if returned_resnames:
        text = "".join(lines)
        missing = [rn for rn in returned_resnames if rn not in text]
        if missing:
            print(f"  [WARN] Возвращённые гетероатомы отсутствуют в .pdbqt: "
                  f"{missing} (Meeko/OpenBabel мог их отбросить)")
            report.append(f"  WARN: потеряны гетероатомы {missing}")
            ok = False
        else:
            print(f"  [OK] Возвращённые гетероатомы на месте: "
                  f"{sorted(returned_resnames)}")
            report.append(f"  OK: гетероатомы на месте {sorted(returned_resnames)}")

    if ok:
        print("  [OK] Базовая валидация формата пройдена.")
    return ok


# ======================================================================
#  main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Подготовка рецептора — инкремент 2 "
                    "(gemmi: .pdb/.cif, анализ + очистка + инвентаризация).")
    ap.add_argument("-i", "--input", required=True,
                    help="входная структура (.pdb или .cif)")
    ap.add_argument("-c", "--config", required=True,
                    help="Vina config-файл с center_*/size_*")
    ap.add_argument("-l", "--ligand", default=None,
                    help="HET-код целевого лиганда-референса (необязательно)")
    ap.add_argument("-o", "--outdir", default=".", help="папка вывода")
    ap.add_argument("--non-interactive", action="store_true",
                    help="не задавать вопросов; брать безопасные дефолты "
                         "(неизвестный HET->buffer, возврат кофактора->нет, "
                         "fallback->прервать)")
    ap.add_argument("--auto-ligand", action="store_true",
                    help="авто-детект кода лиганда: не-словарный HET в боксе "
                         "с наибольшим числом атомов")
    ap.add_argument("--stop-after", default=None,
                    choices=["stage2"],
                    help="остановиться после указанной стадии (для триажа)")
    ap.add_argument("--noninteractive-fallback", default="abort",
                    choices=["abort", "openbabel"],
                    help="поведение в неинтерактиве при падении Meeko: "
                         "abort (прервать, по умолчанию) или openbabel "
                         "(авто-fallback)")
    args = ap.parse_args()

    global NON_INTERACTIVE, FALLBACK_MODE
    NON_INTERACTIVE = args.non_interactive
    FALLBACK_MODE = args.noninteractive_fallback

    if not os.path.exists(args.input):
        sys.exit(f"[ОШИБКА] Нет файла: {args.input}")
    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    report = []

    try:
        structure = gemmi.read_structure(args.input)
    except Exception as e:
        sys.exit(f"[ОШИБКА] gemmi не смог прочитать структуру: {e}")

    info = analyze_header_text(args.input)
    center, size = read_box(args.config)
    n_models = len(structure)
    is_cif = args.input.lower().endswith((".cif", ".mmcif"))

    print("=" * 60)
    print(f"СТАДИЯ 0 — анализ структуры: {args.input}")
    print(f"  Формат входа: {'CIF' if is_cif else 'PDB'}")
    print(f"  Метод:        {info['method']}")
    print(f"  Разрешение:   "
          f"{info['resolution'] if info['resolution'] else 'н/д'} Å")
    print(f"  MODEL-блоков: {n_models}")
    print(f"  Бокс центр:   {center}")
    print(f"  Бокс размер:  {size}")
    if info["method"] == "CRYO-EM":
        print("  [!] cryo-EM: на финальной сборке вероятен fallback на OpenBabel.")
    print("=" * 60)

    report.append("=== СТАДИЯ 0: анализ ===")
    report.append(f"файл: {args.input}")
    report.append(f"формат: {'CIF' if is_cif else 'PDB'}")
    report.append(f"метод: {info['method']}")
    report.append(f"разрешение: {info['resolution']}")
    report.append(f"MODEL-блоков: {n_models}")
    report.append(f"бокс center={center} size={size}\n")

    if n_models > 1:
        print(f"  [i] Найдено {n_models} моделей — оставляю первую.")
        report.append(f"мульти-модель ({n_models}): оставлена первая")
        while len(structure) > 1:
            del structure[1]

    # авто-детект лиганда, если запрошен и код не задан явно
    ligand_code = args.ligand
    if args.auto_ligand and not ligand_code:
        ligand_code, n_cand = autodetect_ligand(structure, center, size)
        if ligand_code:
            msg = f"  [auto-ligand] выбран лиганд '{ligand_code}'"
            if n_cand > 1:
                msg += f" (было {n_cand} кандидатов в боксе — проверьте!)"
            print(msg); report.append(msg)
        else:
            print("  [auto-ligand] лиганд в боксе не найден (апо?)")
            report.append("  [auto-ligand] лиганд не найден")

    print("\nСТАДИЯ 1 — очистка и инвентаризация гетероатомов")
    protein, saved = stage1_clean(structure, center, size, ligand_code, report)

    # His-таутомеризация: переименовать HIS в HIE/HID/HIP, чтобы Meeko не падал
    # на неоднозначности. Остатки в `protein` — те же объекты structure[0],
    # поэтому переименование отразится в записываемом белке.
    print("\nHis-таутомеризация (чтобы Meeko не падал на HIE/HID/HIP)")
    tautomerize_his(structure, center, size, args.outdir, base, report)

    cell = structure.cell
    sg = structure.spacegroup_hm

    prot_path = os.path.join(args.outdir, f"{base}_protein.pdb")
    write_structure_pdb(prot_path, protein, cell, sg)
    outputs = [prot_path]

    if saved["cofactor"]:
        p = os.path.join(args.outdir, f"{base}_cofactors.pdb")
        write_structure_pdb(p, saved["cofactor"], cell, sg); outputs.append(p)
    if saved["ion_candidate"]:
        p = os.path.join(args.outdir, f"{base}_ions.pdb")
        write_structure_pdb(p, saved["ion_candidate"], cell, sg); outputs.append(p)
    if saved["ligand"]:
        p = os.path.join(args.outdir, f"{base}_ligand_ref.pdb")
        write_structure_pdb(p, saved["ligand"], cell, sg); outputs.append(p)
    if saved["keep"]:
        p = os.path.join(args.outdir, f"{base}_kept_hetero.pdb")
        write_structure_pdb(p, saved["keep"], cell, sg); outputs.append(p)

    # --- Стадия 2: Meeko-диагностика голого белка ---
    plan = stage2_diagnose(prot_path, args.outdir, base, center, size, report)

    # триаж: остановиться после Стадии 2
    if args.stop_after == "stage2":
        rep_path = os.path.join(args.outdir, f"{base}_prep_report.txt")
        with open(rep_path, "w") as f:
            f.write("\n".join(report) + "\n")
        print(f"\n[триаж] Остановка после Стадии 2. Лог: {rep_path}")
        # код возврата кодирует статус для batch-раннера
        st = plan.get("status", "unknown")
        sys.exit(0 if st in ("clean", "bad_residues") else 3)

    if plan.get("status") in ("rdkit_error", "unknown_error", "tool_error"):
        rep_path = os.path.join(args.outdir, f"{base}_prep_report.txt")
        with open(rep_path, "w") as f:
            f.write("\n".join(report) + "\n")
        print(f"\nЛог сохранён: {rep_path}")
        print("Пайплайн остановлен на Стадии 2 (см. сообщение выше).")
        sys.exit(2)

    # --- Стадия 2.5: починка (ТОЛЬКО если есть близкие к боксу проблемные остатки) ---
    repaired_pdb = prot_path
    repair_status = "clean"
    force_openbabel = False

    # His-неоднозначность на голом белке: -a не поможет (Meeko падает на выборе
    # таутомера до allow_bad_res) -> сразу к финалу через OpenBabel.
    if plan.get("status") == "his_ambiguous":
        print("  Статус Стадии 2 'his_ambiguous' -> финальная сборка через "
              "OpenBabel.")
        report.append("  статус 'his_ambiguous' -> финал через OpenBabel")
        force_openbabel = True

    # padding на голом белке: НЕ форсируем OpenBabel. Meeko с -a на финале,
    # скорее всего, выкинет padding-остаток через allow_bad_res и соберёт сам.
    # Если и там упадёт — сработает штатный fallback.
    if plan.get("status") == "padding_error":
        print("  Статус Стадии 2 'padding_error' -> пробуем Meeko с -a на финале "
              "(allow_bad_res обычно справляется).")
        report.append("  статус 'padding_error' -> обычная Стадия 4 (Meeko -a)")

    if plan.get("status") == "bad_residues":
        if plan.get("fix"):
            # есть что чинить рядом с боксом — запускаем починку
            repaired_pdb, repair_status = stage25_repair(
                prot_path, args.outdir, base, plan, center, size, report)
            if repaired_pdb:
                outputs.append(repaired_pdb)
            # жёсткие сбои самого PDBFixer/инструмента — остановка
            if repair_status in ("fixer_error", "tool_error"):
                rep_path = os.path.join(args.outdir, f"{base}_prep_report.txt")
                with open(rep_path, "w") as f:
                    f.write("\n".join(report) + "\n")
                print(f"\nЛог сохранён: {rep_path}")
                print("Пайплайн остановлен на Стадии 2.5 (см. сообщение выше).")
                sys.exit(2)
            # RDKit после починки: достроенные атомы дают клэш, который RDKit
            # не принимает. Meeko бесполезен -> собрать ПОЧИНЕННЫЙ через OpenBabel
            # (близкие сохранены достроенными).
            if repair_status == "rdkit_error":
                msg = (f"  Починка дала 'rdkit_error': достроенные близкие "
                       f"остатки\n  не проходят RDKit-проверку. Близкие сохраняем "
                       f"достроенными,\n  собираем через OpenBabel на Стадии 4.")
                print(msg); report.append(msg)
                force_openbabel = True
            # Padding после починки: обычно дисульфидный мостик рядом с проблемной
            # зоной (НЕ артефакт удаления — мы дальние не удаляем). Meeko с -a на
            # про достройку и НЕ про RDKit. Meeko с -a на финале, скорее всего,
            # справится сам (выкинет padding-остаток). НЕ форсируем OpenBabel —
            # идём на обычную Стадию 4 с починенным белком.
            elif repair_status == "padding_error":
                msg = (f"  Починка дала 'padding_error' (артефакт стыков). НЕ "
                       f"форсируем\n  OpenBabel — Meeko с -a на финале обычно "
                       f"справляется. Близкие\n  остатки сохранены достроенными.")
                print(msg); report.append(msg)
                # repaired_pdb остаётся починенным; force_openbabel = False
        else:
            # близких нет — чинить нечего; дальние уйдут под -a на финале
            msg = (f"\nСТАДИЯ 2.5 — пропущена: проблемных остатков рядом с боксом нет.\n"
                   f"  Все {len(plan.get('remove', []))} проблемных остатков далеко "
                   f"от кармана —\n"
                   f"  они будут удалены автоматически через -a на финальной сборке.\n"
                   f"  Структуру не трогаем (никаких удалений/достроек).")
            print(msg)
            report.append("\n=== СТАДИЯ 2.5 ===")
            report.append(f"пропущена: близких к боксу проблемных остатков нет; "
                          f"{len(plan.get('remove', []))} дальних уйдут под -a")
            repair_status = "skipped_far_only"

    # --- Стадия 3: возврат близких кофакторов/ионов в рецептор ---
    receptor_pdb, returned_count = stage3_return_heteroatoms(
        repaired_pdb, saved, center, size, args.outdir, base, report)
    outputs.append(receptor_pdb)

    # имена возвращённых гетероатомов — для проверки сохранности на Стадии 5
    returned_resnames = set()
    if returned_count > 0:
        rec_st = gemmi.read_structure(receptor_pdb)
        prot_st = gemmi.read_structure(repaired_pdb)
        prot_names = set(r.name for ch in prot_st[0] for r in ch)
        for ch in rec_st[0]:
            for r in ch:
                if r.het_flag == "H" and r.name not in prot_names:
                    returned_resnames.add(r.name)

    # --- Стадия 4: финальная сборка .pdbqt ---
    pdbqt, method, asm_status = stage4_assemble(
        receptor_pdb, repaired_pdb, returned_count, args.outdir, base, report,
        force_openbabel=force_openbabel)

    if asm_status in ("aborted", "failed") or pdbqt is None:
        rep_path = os.path.join(args.outdir, f"{base}_prep_report.txt")
        with open(rep_path, "w") as f:
            f.write("\n".join(report) + "\n")
        print(f"\nЛог сохранён: {rep_path}")
        print("Финальная сборка не завершена (см. сообщение выше).")
        sys.exit(2)

    outputs.append(pdbqt)

    # --- Стадия 5: валидация формата ---
    valid = stage5_validate(pdbqt, returned_resnames, report)

    report.append("\n=== Файлы на выходе ===")
    for o in outputs:
        report.append(f"  {o}")
    rep_path = os.path.join(args.outdir, f"{base}_prep_report.txt")
    with open(rep_path, "w") as f:
        f.write("\n".join(report) + "\n")

    print("\n" + "=" * 60)
    print("ГОТОВО — пайплайн завершён (Стадии 0-5).")
    print("=" * 60)
    print("Файлы:")
    for o in outputs:
        print(f"  {o}")
    print(f"  {rep_path}  (лог решений)")
    print(f"\nИтоговый рецептор для докинга: {pdbqt}")
    print(f"  метод сборки: {method}")
    print(f"  валидация формата: {'пройдена' if valid else 'с предупреждениями'}")
    if method.startswith("openbabel"):
        print("  [напоминание] собрано через OpenBabel (обходной путь).")
    if method == "meeko_no_cofactor":
        print("  [напоминание] кофактор был убран ради успешной сборки Meeko.")


if __name__ == "__main__":
    main()
