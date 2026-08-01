"""
Подготовка лигандов для AutoDock Vina: SMILES -> PDBQT.

Изменения относительно UPD3:
  1. Новый порядок: SMILES -> таутомеры -> протонирование (Dimorphite-DL) -> ETKDG -> PDBQT.
     (Раньше: SMILES -> ETKDG (нейтральная форма) -> obabel -p -> PDBQT, что давало
     кривые координаты для заряженных групп.)
  2. Энумерация таутомеров сохраняет стереохимию (SetRemoveSp3Stereo/BondStereo = False).
  3. xtb-фильтрация таутомеров использует 3-5 конформеров (берётся минимальная энергия).
  4. В PDBQT добавляются REMARK с Mol_ID, Tau_ID, Input_SMILES, Protonated_SMILES.
  5. mol2 сохраняются (для визуальной проверки).
"""

import glob
import os
import re
import subprocess
import argparse
import logging
import multiprocessing as mp
import tempfile
import csv
from functools import partial
from tqdm import tqdm
from datetime import datetime
import collections
import shutil

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

# ===================== НАСТРОЙКИ =====================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.FileHandler('processing.log'), logging.StreamHandler()]
)

FOLDERS = {'2d': '2d', '3d': '3d', '3d_ph': '3d_pH', 'mol2': 'mol2', 'pdbqt': 'pdbqt'}

# Элементы, которые понимает стандартный AutoDock Vina (vanilla, без кастомных
# параметров AD4). Всё, что вне этого набора — B, Si, As, Se, Te, металлы —
# вызывает у Vina ошибку парсинга "is not a valid AutoDock type" или не имеет
# scoring-параметров. Такие лиганды отсекаются предфильтром на входе.
DOCKABLE_ELEMENTS = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'}

USE_MEEKO = True

# Dimorphite-DL: универсальный адаптер для трёх вариантов API.
# - 2.x (новая, требует Python ≥ 3.10): функция protonate_smiles
# - 1.x ojmb-форк (Python ≥ 3.6): функция run(smiles=..., min_ph=..., max_ph=...)
# - 1.x оригинал durrantlab: функция run_with_mol_list([mol], min_ph=..., max_ph=...)
# Скрипт автоматически подхватит ту версию, которая установлена.
DIMORPHITE_API = None
_dl_protonate_v2 = None
_dl_module_v1 = None

try:
    # Сначала пробуем новый 2.x API
    from dimorphite_dl import protonate_smiles as _dl_protonate_v2
    DIMORPHITE_API = '2.x'
except ImportError:
    try:
        import dimorphite_dl as _dl_module_v1
        # Различаем оригинал 1.x и форк ojmb по доступным функциям
        if hasattr(_dl_module_v1, 'run') and callable(_dl_module_v1.run):
            DIMORPHITE_API = '1.x_ojmb'
        elif hasattr(_dl_module_v1, 'run_with_mol_list'):
            DIMORPHITE_API = '1.x'
        else:
            _dl_module_v1 = None
    except ImportError:
        pass

DIMORPHITE_AVAILABLE = DIMORPHITE_API is not None


# ===================== УТИЛИТЫ =====================

def run_command(cmd, timeout=None, ligand_name=None):
    try:
        result = subprocess.run(cmd, check=True, timeout=timeout, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        name = ligand_name or ' '.join(cmd)
        logging.error(f"ERROR   | {name}: {e}")
        if e.stderr:
            logging.error(f"STDERR: {e.stderr.strip()}")
        return False
    except Exception as e:
        logging.error(f"UNEXPECTED ERROR: {e}")
        return False


def extract_ligand_id(base: str, ligand_prefix: str = 'ligand_', tautomer_prefix: str = 'tau_'):
    """Извлекаем (mol_id, tau_id) из имени файла. См. UPD3 для деталей."""
    l_esc = re.escape(ligand_prefix)
    t_esc = re.escape(tautomer_prefix)
    m = re.search(rf'{l_esc}(\d+)_{t_esc}(\d+)(?:_|\.|$)', base)
    if m:
        return m.group(1), int(m.group(2))
    m = re.search(r'ligand_(\d+)_tau(\d+)', base)
    if m:
        return m.group(1), int(m.group(2))
    m = re.search(r'(\d+)_tau(\d+)', base)
    if m:
        return m.group(1), int(m.group(2))
    m = re.search(r'(\d+)', base)
    if m:
        return m.group(1), 1
    return None, None


def conformer_is_sane(mol):
    try:
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        span = positions.max(axis=0) - positions.min(axis=0)
        if span.max() < 1.0:
            return False
        import numpy as np
        if not np.isfinite(positions).all():
            return False
        return True
    except Exception:
        return False


# ===================== ПРОТОНИРОВАНИЕ =====================

def protonate_smiles_obabel_cli(smiles: str, ph: float, mol_tag: str = '') -> str:
    """
    Fallback-протонирование SMILES через obabel CLI на SMILES-уровне (не 3D).
    Хуже Dimorphite-DL по точности (у obabel простые правила pKa), но
    лучше, чем протонировать уже готовое 3D: эмбеддинг сразу строит
    геометрию под правильную ионную форму.
    Возвращает исходный SMILES, если obabel недоступен или упал.
    """
    try:
        result = subprocess.run(
            ['obabel', f'-:{smiles}', '-osmi', '-p', str(ph)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            # obabel может выдать "SMILES\tNAME\n" или просто "SMILES\n"
            line = result.stdout.strip().split('\n')[0]
            prot = line.split('\t')[0].split()[0].strip()
            if prot:
                return prot
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logging.debug(f"obabel SMILES-протонирование упало для {mol_tag}: {e}")
    return smiles


def protonate_smiles_dimorphite(smiles: str, ph: float, mol_tag: str = '') -> str:
    """
    Каскад протонирования SMILES:
      1. Dimorphite-DL (если установлен) — лучшее качество.
         Поддерживаются три API: 2.x (protonate_smiles), 1.x ojmb-форк (run),
         1.x оригинал (run_with_mol_list).
      2. obabel CLI на SMILES-уровне — fallback, среднее качество.
      3. Исходный SMILES — последний резерв.
    """
    # === Уровень 1: Dimorphite-DL ===
    if DIMORPHITE_API == '2.x':
        try:
            result = _dl_protonate_v2(
                smiles,
                ph_min=ph,
                ph_max=ph,
                precision=0.0,
                max_variants=1,
            )
            if result and len(result) > 0:
                return result[0]
        except Exception as e:
            logging.warning(f"Dimorphite-DL 2.x не справился с {mol_tag} "
                            f"({smiles[:60]}): {e}")

    elif DIMORPHITE_API == '1.x_ojmb':
        try:
            # ojmb-форк: run() принимает SMILES, возвращает list[str]
            result = _dl_module_v1.run(
                smiles=smiles,
                min_ph=ph,
                max_ph=ph,
                pka_precision=0.0,
                max_variants=1,
                silent=True,
            )
            if result and len(result) > 0:
                # Может быть list[str] (обычно) или list[Mol] (если RDKit Mol на входе)
                first = result[0]
                if isinstance(first, str):
                    return first
                else:
                    return Chem.MolToSmiles(first)
        except Exception as e:
            logging.warning(f"Dimorphite-DL ojmb не справился с {mol_tag} "
                            f"({smiles[:60]}): {e}")

    elif DIMORPHITE_API == '1.x':
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # 1.x API принимает list[Mol], возвращает list[Mol]
                protonated_mols = _dl_module_v1.run_with_mol_list(
                    [mol],
                    min_ph=ph,
                    max_ph=ph,
                    pka_precision=0.0,
                )
                if protonated_mols and len(protonated_mols) > 0:
                    return Chem.MolToSmiles(protonated_mols[0])
        except Exception as e:
            logging.warning(f"Dimorphite-DL 1.x не справился с {mol_tag} "
                            f"({smiles[:60]}): {e}")

    # === Уровень 2: obabel SMILES-уровень ===
    return protonate_smiles_obabel_cli(smiles, ph, mol_tag)


def protonate_sdf_obabel_fallback(sdf_in: str, sdf_out: str, ph: float,
                                  ligand_name: str, timeout: int) -> bool:
    """Fallback: протонирование готового 3D SDF через obabel при заданном pH."""
    return run_command(
        ['obabel', sdf_in, '-O', sdf_out, '-p', str(ph)],
        timeout=timeout, ligand_name=ligand_name
    )


# ===================== ТАУТОМЕРЫ =====================

def xtb_tautomer_energy(smiles: str, mol_tag: str, n_confs: int = 5,
                        timeout: int = 60):
    """
    Считает энергию таутомера через GFN2-xTB с ALPB(water).
    Генерирует n_confs конформеров, оптимизирует каждый, возвращает МИНИМАЛЬНУЮ энергию.
    Это устойчивее, чем 1-конформерный подход (UPD3): энергия меньше зависит
    от случайно выбранной геометрии.
    Возвращает энергию в ккал/моль или None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if not cids:
        return None

    # Лёгкая MMFF преоптимизация всех конформеров
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=300, mmffVariant='MMFF94s')
    except Exception:
        pass

    energies_kcal = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for cid in cids:
            xyz_path = os.path.join(tmpdir, f'mol_{cid}.xyz')
            try:
                Chem.MolToXYZFile(mol, xyz_path, confId=cid)
            except Exception:
                continue
            try:
                result = subprocess.run(
                    ['xtb', xyz_path, '--gfn', '2', '--opt', 'loose',
                     '--alpb', 'water', '--silent'],
                    capture_output=True, text=True, timeout=timeout, cwd=tmpdir
                )
            except subprocess.TimeoutExpired:
                logging.warning(f"xtb timeout (conf {cid}) для {mol_tag}")
                continue
            except FileNotFoundError:
                logging.error("xtb не найден в PATH! Установите: conda install -c conda-forge xtb")
                return None

            if result.returncode != 0:
                continue

            for line in result.stdout.split('\n'):
                if 'TOTAL ENERGY' in line:
                    try:
                        hartree = float(line.split()[-3])
                        energies_kcal.append(hartree * 627.5094740631)
                        break
                    except (ValueError, IndexError):
                        continue

    if not energies_kcal:
        logging.warning(f"xtb не смог обработать ни один конформер: {mol_tag}")
        return None
    return min(energies_kcal)


def _check_stereo_preserved(orig_mol, taut_mol) -> bool:
    """
    Проверяет, что хиральные центры и стерео-связи в таутомере совпадают
    с оригиналом (по количеству assigned-центров).
    """
    try:
        orig_centers = Chem.FindMolChiralCenters(orig_mol, includeUnassigned=False)
        taut_centers = Chem.FindMolChiralCenters(taut_mol, includeUnassigned=False)
        if len(orig_centers) != len(taut_centers):
            return False
        # Проверка стерео двойных связей
        orig_db = sum(1 for b in orig_mol.GetBonds()
                      if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ))
        taut_db = sum(1 for b in taut_mol.GetBonds()
                      if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ))
        return orig_db == taut_db
    except Exception:
        return True  # не валим работу из-за проверки


def enumerate_tautomers_for_smiles(smiles: str, mol_id,
                                   max_tautomers: int, score_cutoff: float,
                                   mode: str = 'score',
                                   energy_cutoff: float = 3.0,
                                   xtb_timeout: int = 60,
                                   xtb_n_confs: int = 5,
                                   strict_stereo: bool = True):
    """
    Энумерирует таутомеры с СОХРАНЕНИЕМ СТЕРЕОХИМИИ.
    strict_stereo=True (default): таутомеры с потерянной хиральностью или
    изменённой E/Z-конфигурацией ВЫБРАСЫВАЮТСЯ из выдачи.
    strict_stereo=False: оставляются с предупреждением в логе.
    Возвращает кортеж (tautomers, n_stereo_dropped), где
      tautomers = [(tau_id, tau_smiles, score, delta_energy_kcal, label), ...]
      n_stereo_dropped = сколько таутомеров отброшено в strict-stereo режиме.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.warning(f"Невалидный SMILES для mol_id={mol_id}: {smiles}")
        return [], 0

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        logging.warning(f"Sanitize failed для mol_id={mol_id}: {e}")
        return [(1, smiles, 0.0, 0.0, 'fallback_no_enum')], 0

    # Назначаем стерео, чтобы потом было что сохранять
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    enumerator = rdMolStandardize.TautomerEnumerator()

    # === СОХРАНЕНИЕ СТЕРЕО — критично! ===
    # Эти флаги появились в RDKit ~2020; проверяем через hasattr на старых сборках.
    if hasattr(enumerator, 'SetRemoveSp3Stereo'):
        enumerator.SetRemoveSp3Stereo(False)
    if hasattr(enumerator, 'SetRemoveBondStereo'):
        enumerator.SetRemoveBondStereo(False)
    if hasattr(enumerator, 'SetReassignStereo'):
        enumerator.SetReassignStereo(False)

    # Разумные лимиты (в UPD3 было 1000/2000 «для теста» — это слишком много)
    enumerator.SetMaxTautomers(200)
    enumerator.SetMaxTransforms(500)

    try:
        tauts = list(enumerator.Enumerate(mol))
    except Exception as e:
        logging.warning(f"Tautomer enumeration failed для mol_id={mol_id}: {e}")
        return [(1, smiles, 0.0, 0.0, 'fallback_no_enum')], 0

    if not tauts:
        return [(1, smiles, 0.0, 0.0, 'fallback_no_enum')], 0

    # Score + дедупликация + ПРОВЕРКА СТЕРЕО
    seen_smiles = set()
    scored = []
    n_stereo_lost = 0
    n_stereo_dropped = 0
    for t in tauts:
        try:
            # Используем isomeric SMILES для дедупликации (иначе R/S схлопываются)
            canon = Chem.MolToSmiles(t, isomericSmiles=True)
            if canon in seen_smiles:
                continue
            seen_smiles.add(canon)

            if not _check_stereo_preserved(mol, t):
                n_stereo_lost += 1
                if strict_stereo:
                    # Строгий режим: выкидываем таутомер с потерянной хиральностью
                    n_stereo_dropped += 1
                    logging.debug(f"mol_id={mol_id}: таутомер ОТБРОШЕН по стерео "
                                  f"(strict mode): {canon[:80]}")
                    continue
                else:
                    # Мягкий режим: оставляем с предупреждением
                    logging.debug(f"mol_id={mol_id}: таутомер потерял стерео "
                                  f"(оставлен): {canon[:80]}")

            score = enumerator.ScoreTautomer(t)
            scored.append((canon, score))
        except Exception:
            continue

    if n_stereo_lost > 0:
        if strict_stereo:
            logging.info(f"mol_id={mol_id}: отброшено {n_stereo_dropped} таутомеров с "
                         f"изменённой стереохимией (strict-stereo)")
        else:
            logging.info(f"mol_id={mol_id}: {n_stereo_lost} таутомеров с изменённой "
                         f"стереохимией (оставлены, требуют проверки)")

    if not scored:
        return [(1, smiles, 0.0, 0.0, 'fallback_no_enum')], n_stereo_dropped

    scored.sort(key=lambda x: -x[1])
    best_score = scored[0][1]

    logging.info(f"mol_id={mol_id}: RDKit нашёл {len(scored)} таутомеров (после дедупликации)")

    # Фильтр по score
    if score_cutoff >= 999:
        filtered = scored
    else:
        filtered = [(canon, score) for canon, score in scored
                    if (best_score - score) <= score_cutoff]
    filtered = filtered[:max_tautomers]

    # === xtb-фильтрация (с несколькими конформерами на таутомер) ===
    if mode == 'xtb' and len(filtered) > 1:
        energies = []
        for canon, score in filtered:
            e = xtb_tautomer_energy(
                canon, f"mol{mol_id}_{canon[:30]}",
                n_confs=xtb_n_confs, timeout=xtb_timeout
            )
            energies.append((canon, score, e))

        valid_e = [x for x in energies if x[2] is not None]
        if valid_e:
            min_e = min(x[2] for x in valid_e)
            final = []
            for canon, score, e in energies:
                if e is None:
                    # xtb упал — не считаем «как лучший», ставим штраф
                    final.append((canon, score, energy_cutoff + 0.01, 'xtb_failed'))
                elif (e - min_e) <= energy_cutoff:
                    final.append((canon, score, e - min_e, 'xtb'))
            # Фильтруем xtb_failed, если есть успешные
            successful = [f for f in final if f[3] == 'xtb']
            result = successful if successful else final
        else:
            result = [(canon, score, 0.0, 'score_only') for canon, score in filtered]
    else:
        result = [(canon, score, 0.0, 'score_only') for canon, score in filtered]

    output = []
    for tau_id, item in enumerate(result, start=1):
        canon, score, delta_e, label = item
        output.append((tau_id, canon, score, delta_e, label))
    return output, n_stereo_dropped


def _canonical_key_for_dedup(smiles: str) -> str:
    """
    Возвращает ключ для дедупликации таутомеров ПОСЛЕ протонирования.

    Использует InChI с опцией /FixedH (fixed-hydrogen layer). Эта вариация
    InChI работает именно так, как нужно для нашей задачи:

      - СХЛОПЫВАЕТ резонансные формы делокализованных анионов/катионов
        (имидазолат, тетразолат, бензимидазолат: заряд формально на разных
        атомах, но физически делокализован — fixed-H InChI это видит и
        даёт одинаковый ключ).
      - НЕ СХЛОПЫВАЕТ настоящие таутомеры (амид/иминол, кето/енол,
        1H-/3H-бензимидазол), где H реально стоит на разных атомах —
        FixedH-слой фиксирует позиции водородов, поэтому ключи разные.

    Почему НЕ стандартный InChIKey: его mobile-H слой агрессивно
    объединяет ЛЮБЫЕ таутомеры — амид и иминол получают один ключ,
    что приводит к ложной дедупликации настоящих таутомеров.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        inchi = Chem.MolToInchi(mol, options='/FixedH')
        if inchi:
            return Chem.InchiToInchiKey(inchi)
        return Chem.MolToSmiles(mol)
    except Exception:
        return smiles


def expand_tautomers(input_smi: str, output_smi: str, mapping_tsv: str,
                     max_tautomers: int, score_cutoff: float,
                     mode: str, energy_cutoff: float, xtb_timeout: int,
                     xtb_n_confs: int, ph: float,
                     workers: int = 1,
                     input_format: str = 'smi',
                     ligand_prefix: str = 'ligand_',
                     tautomer_prefix: str = 'tau_',
                     strict_stereo: bool = True,
                     id_col: str = None,
                     smiles_col: str = None):
    """
    Читает SMILES, энумерирует таутомеры, ПРОТОНИРУЕТ каждый таутомер при pH,
    пишет расширенный .smi (с протонированными SMILES для дальнейшего эмбеддинга).
    Возвращает кортеж (smiles_dict, stats), где
      smiles_dict = {(mol_id, tau_id): {'input': neutral, 'protonated': prot}}
      stats — dict со счётчиками для финального отчёта.
    """
    originals = _read_input_file(input_smi, input_format,
                                  id_col=id_col, smiles_col=smiles_col)

    logging.info(f"Энумерация таутомеров: {len(originals)} исходных SMILES, "
                 f"mode={mode}, max={max_tautomers}, score_cutoff={score_cutoff}"
                 + (f", energy_cutoff={energy_cutoff} kcal/mol, "
                    f"xtb_n_confs={xtb_n_confs}" if mode == 'xtb' else ''))
    if strict_stereo:
        logging.info("Strict-stereo: ВКЛ (таутомеры с потерянным стерео отбрасываются)")
    else:
        logging.info("Strict-stereo: ВЫКЛ (таутомеры с потерянным стерео сохраняются)")

    func = partial(
        _enumerate_wrapper,
        max_tautomers=max_tautomers,
        score_cutoff=score_cutoff,
        mode=mode,
        energy_cutoff=energy_cutoff,
        xtb_timeout=xtb_timeout,
        xtb_n_confs=xtb_n_confs,
        strict_stereo=strict_stereo,
    )

    if mode == 'xtb' and workers > 1:
        with mp.Pool(workers) as pool:
            all_results = list(tqdm(pool.imap(func, originals),
                                    total=len(originals), desc="Таутомеры (xtb)"))
    else:
        all_results = [func(item) for item in tqdm(originals, desc="Таутомеры (score)")]

    # === Протонирование каждого таутомера + пост-дедупликация ===
    if DIMORPHITE_AVAILABLE:
        logging.info(f"Протонирование таутомеров через Dimorphite-DL (pH={ph})...")
    else:
        logging.warning("Dimorphite-DL не установлен (pip install dimorphite-dl). "
                        "Протонирование будет fallback через obabel -p на этапе 3D SDF.")

    smiles_dict = {}
    total_tauts = 0
    mol_ids_with_multiple = 0
    n_deduped = 0
    # === Расширенная статистика для финального отчёта ===
    stats = collections.Counter()

    output_csv = os.path.join(os.path.dirname(output_smi), 'ligands_out.csv')

    with open(output_smi, 'w', encoding='utf-8') as fsmi, \
         open(output_csv, 'w', encoding='utf-8', newline='') as fcsv, \
         open(mapping_tsv, 'w', encoding='utf-8') as fmap:
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow(['ligand_id', 'tau_id', 'input_smiles', 'protonated_smiles'])
        fmap.write("mol_id\ttau_id\toriginal_smiles\ttautomer_smiles\tprotonated_smiles\t"
                   "score\tdelta_energy_kcal\tlabel\n")

        for (mol_id, orig_smiles), result_item in zip(originals, all_results):
            # all_results теперь содержит кортежи (tauts, n_stereo_dropped)
            tauts, n_stereo_dropped = result_item
            if n_stereo_dropped > 0:
                stats['stereo_dropped_tautomers'] += n_stereo_dropped
                stats['mol_ids_with_stereo_loss'] += 1

            if not tauts:
                tauts = [(1, orig_smiles, 0.0, 0.0, 'no_tautomers')]
                stats['mol_no_tautomers'] += 1

            # Учитываем label из энумерации
            for _, _, _, _, label in tauts:
                stats[f'label_{label}'] += 1

            # === Сначала протонируем ВСЕ таутомеры, потом дедуплицируем ===
            protonated_records = []
            seen_keys = set()
            for tau_id_orig, tau_smi, score, delta_e, label in tauts:
                prot_smi = protonate_smiles_dimorphite(
                    tau_smi, ph, mol_tag=f"mol{mol_id}_tau{tau_id_orig}"
                )

                # Статистика протонирования: изменился ли SMILES
                if prot_smi != tau_smi:
                    stats['protonation_changed'] += 1
                else:
                    stats['protonation_unchanged'] += 1

                dedup_key = _canonical_key_for_dedup(prot_smi)

                if dedup_key in seen_keys:
                    n_deduped += 1
                    logging.debug(
                        f"mol_id={mol_id}: таутомер #{tau_id_orig} ({tau_smi[:50]}) "
                        f"после протонирования совпал с уже существующим "
                        f"(резонансная форма) — пропускаем"
                    )
                    continue
                seen_keys.add(dedup_key)
                protonated_records.append((tau_smi, prot_smi, score, delta_e, label))

            if len(protonated_records) > 1:
                mol_ids_with_multiple += 1

            # === Перенумеровываем tau_id и записываем ===
            for new_tau_id, (tau_smi, prot_smi, score, delta_e, label) in \
                    enumerate(protonated_records, start=1):
                mol_id_str = str(mol_id)
                tag = f"{ligand_prefix}{mol_id_str}_{tautomer_prefix}{new_tau_id}"
                fsmi.write(f"{prot_smi}\t{tag}\n")
                csv_writer.writerow([mol_id_str, new_tau_id, tau_smi, prot_smi])
                fmap.write(f"{mol_id}\t{new_tau_id}\t{orig_smiles}\t{tau_smi}\t{prot_smi}\t"
                           f"{score:.3f}\t{delta_e:.3f}\t{label}\n")
                smiles_dict[(mol_id_str, new_tau_id)] = {
                    'input': tau_smi,
                    'protonated': prot_smi,
                }
                total_tauts += 1

    stats['tautomers_total'] = total_tauts
    stats['tautomers_deduped'] = n_deduped
    stats['mol_ids_with_multiple'] = mol_ids_with_multiple
    stats['input_mols'] = len(originals)

    if n_deduped > 0:
        logging.info(f"Пост-дедупликация: убрано {n_deduped} таутомеров, которые "
                     f"после протонирования стали одинаковыми (резонансные формы)")

    avg = total_tauts / max(1, len(originals))
    logging.info(f"Энумерация завершена: {len(originals)} → {total_tauts} таутомеров "
                 f"(avg {avg:.2f}/мол, {mol_ids_with_multiple} мол. имеют >1 таутомера)")
    logging.info(f"Записано: {output_smi}, {output_csv}, mapping: {mapping_tsv}")

    return smiles_dict, stats


def check_dockable(smiles: str, allowed_elements: set, max_rotatable_bonds=None):
    """
    Проверяет, можно ли задокировать молекулу стандартным Vina.
    Возвращает (is_dockable: bool, reason: str).
    reason пустой, если молекула проходит.

    max_rotatable_bonds: если задано, молекулы с числом вращаемых связей
    выше порога отсеиваются (Vina надёжна примерно до 20-30 торсий;
    жёсткий лимит AutoDock4 — 32). None = не проверять гибкость.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, 'invalid_smiles'

    bad_elements = set()
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in allowed_elements:
            bad_elements.add(sym)

    if bad_elements:
        return False, 'unsupported_elements:' + ','.join(sorted(bad_elements))

    if max_rotatable_bonds is not None:
        n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol, strict=True)
        if n_rot > max_rotatable_bonds:
            return False, f'too_flexible:{n_rot}_rotatable_bonds'

    return True, ''


def prefilter_undockable(input_file: str, input_format: str,
                         allowed_elements: set, output_dir: str,
                         max_rotatable_bonds=None,
                         id_col: str = None, smiles_col: str = None):
    """
    Предфильтр: отсекает лиганды, которые Vina не сможет задокировать
    (неподдерживаемые элементы, невалидные SMILES, чрезмерная гибкость).

    Создаёт:
      <output_dir>/ligands_dockable.<ext>   — прошедшие фильтр (идут в пайплайн)
      <output_dir>/ligands_rejected.csv     — отсеянные, с колонкой reason

    Возвращает (путь_к_dockable_файлу, n_kept, n_rejected).
    """
    originals = _read_input_file(input_file, input_format,
                                  id_col=id_col, smiles_col=smiles_col)
    if not originals:
        return input_file, 0, 0

    ext = 'csv' if input_format == 'csv' else 'smi'
    dockable_path = os.path.join(output_dir, f'ligands_dockable.{ext}')
    rejected_path = os.path.join(output_dir, 'ligands_rejected.csv')

    kept = []
    rejected = []
    reason_counter = collections.Counter()

    for mol_id, smi in originals:
        is_ok, reason = check_dockable(smi, allowed_elements, max_rotatable_bonds)
        if is_ok:
            kept.append((mol_id, smi))
        else:
            rejected.append((mol_id, smi, reason))
            # Группируем причины: для too_flexible убираем конкретное число,
            # чтобы статистика была читаемой
            if reason.startswith('too_flexible'):
                reason_counter['too_flexible'] += 1
            elif reason.startswith('unsupported_elements'):
                reason_counter[reason] += 1
            else:
                reason_counter[reason] += 1

    # Пишем dockable в исходном формате
    if input_format == 'csv':
        with open(dockable_path, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['ligand_id', 'SMILES'])
            w.writerows(kept)
    else:
        with open(dockable_path, 'w', encoding='utf-8') as f:
            for mol_id, smi in kept:
                f.write(f"{smi}\t{mol_id}\n")

    # Пишем rejected всегда (даже если пусто — для прозрачности)
    with open(rejected_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ligand_id', 'SMILES', 'reason'])
        w.writerows(rejected)

    logging.info(f"Предфильтр Vina-совместимости: {len(originals)} → "
                 f"{len(kept)} годных, {len(rejected)} отсеяно")
    if rejected:
        logging.info(f"Отсеянные сохранены в {rejected_path}. Причины:")
        for reason, n in reason_counter.most_common():
            logging.info(f"    {n:5d}  {reason}")

    return dockable_path, len(kept), len(rejected)


def _read_input_file(path: str, input_format: str,
                     id_col: str = None, smiles_col: str = None):
    """
    Читает SMI или CSV, возвращает [(mol_id, smiles), ...].

    Для CSV:
    - Если id_col/smiles_col заданы явно — используются эти колонки.
    - Иначе ридер ищет колонки по типичным синонимам (без учёта регистра).
    - Дополнительные колонки игнорируются (можно иметь activity, source и т.п.).
    - Поддерживается разделитель ',' или ';' (автоопределение).
    """
    originals = []
    if input_format == 'csv':
        # Автоопределение разделителя по первой строке
        with open(path, encoding='utf-8-sig') as f:
            first_line = f.readline()
        if first_line.count(';') > first_line.count(','):
            delimiter = ';'
        else:
            delimiter = ','

        with open(path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            headers = reader.fieldnames or []
            if not headers:
                logging.error(f"CSV пустой или без заголовка: {path}")
                return []

            # Карта: lower(заголовок) -> оригинальное имя
            header_map = {h.strip().lower(): h for h in headers if h is not None}

            # === Определение колонки с SMILES ===
            smiles_key = None
            if smiles_col:
                # Явно задано — берём как есть (с учётом регистра поиска)
                if smiles_col in headers:
                    smiles_key = smiles_col
                elif smiles_col.lower() in header_map:
                    smiles_key = header_map[smiles_col.lower()]
                else:
                    logging.error(f"Колонка '{smiles_col}' не найдена в CSV. "
                                  f"Доступные: {headers}")
                    return []
            else:
                # Автопоиск по синонимам
                smiles_synonyms = [
                    'smiles', 'canonical_smiles', 'isomeric_smiles', 'smi',
                    'structure', 'mol', 'smiles_str',
                ]
                for syn in smiles_synonyms:
                    if syn in header_map:
                        smiles_key = header_map[syn]
                        break
                if smiles_key is None:
                    logging.error(f"Не найдена колонка со SMILES в CSV. "
                                  f"Ожидаемые имена: {smiles_synonyms}. "
                                  f"Доступные: {headers}. "
                                  f"Укажите явно через --smiles-col.")
                    return []

            # === Определение колонки с ID ===
            id_key = None
            if id_col:
                if id_col in headers:
                    id_key = id_col
                elif id_col.lower() in header_map:
                    id_key = header_map[id_col.lower()]
                else:
                    logging.error(f"Колонка '{id_col}' не найдена в CSV. "
                                  f"Доступные: {headers}")
                    return []
            else:
                # Автопоиск по синонимам
                id_synonyms = [
                    'ligand_id', 'id', 'mol_id', 'compound_id', 'name',
                    'compound', 'molecule_name', 'molid', 'cid', 'zinc_id',
                    'chembl_id',
                ]
                for syn in id_synonyms:
                    if syn in header_map:
                        id_key = header_map[syn]
                        break
                # Если не нашли — используем порядковый номер строки
                if id_key is None:
                    logging.warning(f"Не найдена колонка с ID в CSV. Ожидаемые: "
                                    f"{id_synonyms}. Доступные: {headers}. "
                                    f"Буду использовать порядковые номера.")

            logging.info(f"CSV: delimiter='{delimiter}', smiles_col='{smiles_key}', "
                         f"id_col='{id_key or '<auto-number>'}'")

            for i, row in enumerate(reader, start=1):
                smi = (row.get(smiles_key) or '').strip()
                if not smi:
                    continue
                if id_key:
                    mol_id = (row.get(id_key) or '').strip() or str(i)
                else:
                    mol_id = str(i)
                originals.append((mol_id, smi))
    else:
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split()
                if parts:
                    smi = parts[0]
                    if len(parts) > 1:
                        tag = parts[1]
                        m = re.search(r'(\d+)', tag)
                        mol_id = m.group(1) if m else str(i)
                    else:
                        mol_id = str(i)
                    originals.append((mol_id, smi))
    return originals


def _enumerate_wrapper(item, max_tautomers, score_cutoff, mode,
                       energy_cutoff, xtb_timeout, xtb_n_confs,
                       strict_stereo=True):
    """Возвращает кортеж (tautomers_list, n_stereo_dropped)."""
    mol_id, smiles = item
    try:
        return enumerate_tautomers_for_smiles(
            smiles, mol_id, max_tautomers, score_cutoff,
            mode=mode, energy_cutoff=energy_cutoff,
            xtb_timeout=xtb_timeout, xtb_n_confs=xtb_n_confs,
            strict_stereo=strict_stereo,
        )
    except Exception as e:
        logging.warning(f"Tautomer enumeration crash для mol_id={mol_id}: {e}")
        return [(1, smiles, 0.0, 0.0, 'crash_fallback')], 0


def rename_split_files(split_dir: str, originals_order,
                       ligand_prefix: str = 'ligand_', tautomer_prefix: str = 'tau_'):
    sdf_files = sorted(
        glob.glob(os.path.join(split_dir, '*.sdf')),
        key=lambda p: int(re.search(r'(\d+)\.sdf$', p).group(1))
        if re.search(r'\d+\.sdf$', p) else 0
    )
    if len(sdf_files) != len(originals_order):
        logging.warning(
            f"Число split-файлов ({len(sdf_files)}) не совпадает с числом "
            f"записей в .smi ({len(originals_order)})!"
        )
    for sdf_path, (mol_id, tau_id) in zip(sdf_files, originals_order):
        new_name = os.path.join(split_dir, f"{ligand_prefix}{mol_id}_{tautomer_prefix}{tau_id}.sdf")
        if sdf_path != new_name:
            os.rename(sdf_path, new_name)


# ===================== КОНВЕРСИЯ В PDBQT =====================

def sdf_to_pdbqt_meeko(sdf_3d: str, pdbqt_path: str, ligand_name: str):
    return run_command(
        ['mk_prepare_ligand.py', '-i', sdf_3d, '-o', pdbqt_path],
        timeout=60, ligand_name=ligand_name
    )


def add_provenance_to_pdbqt(pdbqt_path: str, mol_id, tau_id,
                            input_smiles: str, protonated_smiles: str = None) -> bool:
    """
    Добавляет REMARK с информацией о происхождении в начало PDBQT-файла.
    Не трогает существующие REMARK от Meeko (включая SMILES IDX, которые
    нужны для обратной сборки молекулы из позы).
    """
    if not os.path.exists(pdbqt_path):
        return False
    try:
        with open(pdbqt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logging.warning(f"Не удалось прочитать {pdbqt_path}: {e}")
        return False

    header_lines = [
        f"REMARK  Mol_ID: {mol_id}",
        f"REMARK  Tau_ID: {tau_id}",
        f"REMARK  Input_SMILES: {input_smiles}",
    ]
    if protonated_smiles and protonated_smiles != input_smiles:
        header_lines.append(f"REMARK  Protonated_SMILES: {protonated_smiles}")

    new_content = '\n'.join(header_lines) + '\n' + content

    try:
        with open(pdbqt_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    except Exception as e:
        logging.warning(f"Не удалось записать {pdbqt_path}: {e}")
        return False
    return True


# ===================== ОБРАБОТКА ЛИГАНДА =====================

def process_single_ligand(key, args, smiles_dict):
    """
    Обрабатывает одного лиганда по ключу (mol_id, tau_id).
    ETKDG идёт напрямую из protonated SMILES в smiles_dict.
    2D SDF создаётся ЛЕНИВНО, только при необходимости obabel-fallback gen3D.
    """
    mol_id, tau_id = key
    base = f"{args.ligand_prefix}{mol_id}_{args.tautomer_prefix}{tau_id}"

    pdbqt_path = os.path.join(FOLDERS['pdbqt'], f"{base}.pdbqt")
    if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 100:
        return "skipped"

    sdf_3d = os.path.join(FOLDERS['3d'], f"{base}_3D.sdf")
    mol2_path = os.path.join(FOLDERS['mol2'], f"{base}.mol2")
    embed_method = "fallback"
    pdbqt_backend = "none"

    # key уже задан, не нужно извлекать из имени файла
    # Достаём оба SMILES (входной таутомер + протонированный)
    input_smiles = None
    protonated_smiles = None
    if key in smiles_dict:
        entry = smiles_dict[key]
        if isinstance(entry, dict):
            input_smiles = entry.get('input')
            protonated_smiles = entry.get('protonated')
        else:
            input_smiles = entry
            protonated_smiles = entry

    smiles_for_embed = protonated_smiles or input_smiles
    has_se = False
    if smiles_for_embed:
        m = Chem.MolFromSmiles(smiles_for_embed)
        if m:
            has_se = any(atom.GetSymbol() == 'Se' for atom in m.GetAtoms())

    ff_obabel = 'UFF' if has_se else 'MMFF94'
    charge_method = 'qeq' if has_se else 'gasteiger'

    # === RDKit ETKDGv3 на ПРОТОНИРОВАННОМ SMILES ===
    mmff_uff_fallback = False
    if smiles_for_embed:
        mol = Chem.MolFromSmiles(smiles_for_embed)
        if mol:
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            try:
                mol_id_int = int(mol_id)
            except (TypeError, ValueError):
                mol_id_int = abs(hash(str(mol_id))) % 1000000
            params.randomSeed = 42 + mol_id_int * 100 + tau_id
            params.useSmallRingTorsions = True
            params.useMacrocycleTorsions = True

            success = False
            best_cid = -1
            if args.num_confs > 1:
                cids = AllChem.EmbedMultipleConfs(mol, numConfs=args.num_confs, params=params)
                if cids:
                    if has_se:
                        energies = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=1000)
                    else:
                        energies = AllChem.MMFFOptimizeMoleculeConfs(
                            mol, maxIters=1000, mmffVariant='MMFF94s'
                        )
                    best_idx = min(
                        range(len(energies)),
                        key=lambda i: energies[i][1] if energies[i][0] == 0 else 999999
                    )
                    best_cid = cids[best_idx]
                    success = True
                    embed_method = f"ETKDG_{args.num_confs}confs"
            else:
                if AllChem.EmbedMolecule(mol, params) == 0:
                    if has_se:
                        status = AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                    else:
                        status = AllChem.MMFFOptimizeMolecule(
                            mol, maxIters=1000, mmffVariant='MMFF94s'
                        )
                    if status == -1 and not has_se:
                        logging.info(f"MMFF не параметризовал {base}, пробуем UFF")
                        AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
                        mmff_uff_fallback = True
                    best_cid = mol.GetConformer().GetId()
                    success = True
                    embed_method = "ETKDG"

            if success and not conformer_is_sane(mol):
                logging.warning(f"Конформер схлопнулся/некорректен: {base}, уходим в fallback")
                success = False

            if success:
                writer = Chem.SDWriter(sdf_3d)
                writer.write(mol, confId=best_cid)
                writer.close()

    # === Fallback OpenBabel (gen3D через ленивый 2D) ===
    # Сюда попадаем ТОЛЬКО если ETKDG не справился. Создаём 2D SDF на лету
    # из protonated SMILES, потом obabel gen3D на нём.
    if not os.path.exists(sdf_3d) or os.path.getsize(sdf_3d) == 0:
        if not smiles_for_embed:
            logging.warning(f"Нет SMILES для fallback {base}")
            return "failed"
        logging.info(f"Fallback OpenBabel gen3D для {base}")
        # Ленивая 2D-генерация из SMILES (на лету для одной молекулы)
        sdf_2d_lazy = os.path.join(FOLDERS['2d'], f"{base}_2D.sdf")
        if not run_command(
            ['obabel', f'-:{smiles_for_embed}', '-O', sdf_2d_lazy, '--gen2D', '-h'],
            timeout=args.timeout, ligand_name=base
        ):
            logging.warning(f"Ленивая gen2D не удалась для {base}")
            return "failed"
        if not run_command(
            ['obabel', sdf_2d_lazy, '-O', sdf_3d,
             '--gen3D', '--minimize', '--ff', ff_obabel, '--steps', '1000'],
            timeout=args.timeout, ligand_name=base
        ):
            return "failed"
        embed_method = "obabel_gen3D"

    # === Протонирование SDF (нужно ТОЛЬКО если Dimorphite-DL недоступен) ===
    if DIMORPHITE_AVAILABLE:
        sdf_for_conversion = sdf_3d
    else:
        sdf_protonated = os.path.join(FOLDERS['3d_ph'], f"{base}_3D_pH.sdf")
        protonated_ok = protonate_sdf_obabel_fallback(
            sdf_3d, sdf_protonated, args.ph, base, args.timeout
        )
        sdf_for_conversion = sdf_protonated if protonated_ok and \
            os.path.getsize(sdf_protonated) > 0 else sdf_3d
        if not protonated_ok:
            logging.warning(f"obabel-протонирование не удалось для {base}, используем исходный SDF")

    # === mol2 (для визуальной проверки, не идёт в докинг) ===
    run_command(
        ['obabel', sdf_for_conversion, '-O', mol2_path,
         '--partialcharge', charge_method],
        timeout=args.timeout, ligand_name=base
    )
    if not os.path.exists(mol2_path) or os.path.getsize(mol2_path) == 0:
        logging.warning(f"mol2 не создан для {base}")

    # === PDBQT через Meeko ===
    pdbqt_ok = False
    if USE_MEEKO:
        pdbqt_ok = sdf_to_pdbqt_meeko(sdf_for_conversion, pdbqt_path, base)
        if pdbqt_ok:
            pdbqt_backend = "meeko"
        else:
            logging.warning(f"Meeko не справился с {base}, пробуем obabel fallback")

    # === Fallback obabel sdf→pdbqt ===
    if not pdbqt_ok:
        pdbqt_ok = run_command(
            ['obabel', sdf_for_conversion, '-O', pdbqt_path],
            timeout=args.timeout, ligand_name=base
        )
        if pdbqt_ok:
            pdbqt_backend = "obabel"

    if not pdbqt_ok:
        return "failed"

    # === Финальная проверка + добавление SMILES в PDBQT ===
    if os.path.exists(pdbqt_path):
        with open(pdbqt_path) as f:
            content = f.read()
        if 'ROOT' not in content:
            logging.warning(f"pdbqt без ROOT (возможно пустой): {base}")
            return "failed"

        # Добавляем REMARK с SMILES и провенансом
        if input_smiles:
            add_provenance_to_pdbqt(
                pdbqt_path, mol_id, tau_id,
                input_smiles=input_smiles,
                protonated_smiles=protonated_smiles,
            )
        else:
            logging.warning(f"Не нашли SMILES для {base} в smiles_dict — REMARK не добавлен")

    # Возвращаем структурированный код: "<embed>|<pdbqt>[|mmff_uff]"
    # Парсится в финальной статистике для подсчёта по каждой компоненте.
    code = f"{embed_method}|{pdbqt_backend}"
    if mmff_uff_fallback:
        code += "|mmff_uff"
    return code


def parse_torsdof_from_pdbqt(pdbqt_path: str):
    """
    Читает число активных торсий (TORSDOF) из готового PDBQT-файла.
    Это именно то число степеней свободы кручения, которое использует Vina
    (с учётом того, что Meeko фиксирует амидные связи, отбрасывает терминальные
    группы и т.п. — поэтому TORSDOF обычно меньше «сырых» вращаемых связей).
    Возвращает int или None, если не нашлось.
    """
    try:
        with open(pdbqt_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('TORSDOF'):
                    return int(line.split()[1])
    except Exception:
        pass
    return None


def finalize_docking_set(pdbqt_dir: str, smiles_dict, output_dir: str,
                         ligand_prefix: str, tautomer_prefix: str,
                         max_torsions=None, dock_id_prefix='D',
                         prefilter_rejected_path=None):
    """
    Финальная постобработка набора для докинга:
      1. Сканирует все PDBQT в pdbqt_dir.
      2. Читает TORSDOF (реальное число активных торсий).
      3. Если max_torsions задан — отсекает слишком гибкие (перемещает в
         pdbqt_rejected/ и пишет в отчёт отказов).
      4. Присваивает сквозной dock_id годным лигандам.
      5. Пишет два CSV:
         - docking_set.csv    : годные, проиндексированы (идут в докинг)
         - rejected_ligands.csv: отсеянные (элементы из предфильтра + торсии),
                                  с указанием стадии и причины.

    Возвращает (n_dockable, n_rejected_torsions).
    """
    pdbqt_files = sorted(glob.glob(os.path.join(pdbqt_dir, '*.pdbqt')))

    dockable = []      # (dock_id, fname, mol_id, tau_id, torsdof, in_smi, prot_smi)
    rejected_tors = [] # (mol_id, tau_id, smi, torsdof)

    rejected_dir = os.path.join(os.path.dirname(pdbqt_dir.rstrip('/')) or '.',
                                'pdbqt_rejected')

    # Сначала собираем все годные/гибкие
    parsed = []
    for pf in pdbqt_files:
        base = os.path.splitext(os.path.basename(pf))[0]
        mol_id, tau_id = extract_ligand_id(base, ligand_prefix, tautomer_prefix)
        torsdof = parse_torsdof_from_pdbqt(pf)
        entry = smiles_dict.get((str(mol_id), tau_id), {})
        in_smi = entry.get('input', '') if isinstance(entry, dict) else ''
        prot_smi = entry.get('protonated', '') if isinstance(entry, dict) else ''
        parsed.append((pf, base, mol_id, tau_id, torsdof, in_smi, prot_smi))

    # Применяем порог по торсиям
    too_flexible = []
    for pf, base, mol_id, tau_id, torsdof, in_smi, prot_smi in parsed:
        if (max_torsions is not None and torsdof is not None
                and torsdof > max_torsions):
            too_flexible.append((pf, base, mol_id, tau_id, torsdof, in_smi, prot_smi))
        else:
            dockable.append((pf, base, mol_id, tau_id, torsdof, in_smi, prot_smi))

    # Перемещаем слишком гибкие в отдельную папку
    if too_flexible:
        os.makedirs(rejected_dir, exist_ok=True)
        for pf, base, mol_id, tau_id, torsdof, in_smi, prot_smi in too_flexible:
            try:
                shutil.move(pf, os.path.join(rejected_dir, os.path.basename(pf)))
            except Exception as e:
                logging.warning(f"Не удалось переместить {pf}: {e}")
            rejected_tors.append((mol_id, tau_id, in_smi or prot_smi, torsdof))

    # Сортируем годные по (mol_id, tau_id) для стабильной индексации
    def _sort_key(x):
        _, _, mid, tid, _, _, _ = x
        try:
            return (int(mid), int(tid))
        except (TypeError, ValueError):
            return (10**9, 0)
    dockable.sort(key=_sort_key)

    # === docking_set.csv: годные с сквозным dock_id ===
    docking_csv = os.path.join(output_dir, 'docking_set.csv')
    width = max(6, len(str(len(dockable))))
    with open(docking_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dock_id', 'pdbqt_file', 'mol_id', 'tau_id',
                    'n_torsions', 'input_smiles', 'protonated_smiles'])
        for i, (pf, base, mol_id, tau_id, torsdof, in_smi, prot_smi) in \
                enumerate(dockable, start=1):
            dock_id = f"{dock_id_prefix}{i:0{width}d}"
            w.writerow([dock_id, os.path.basename(pf), mol_id, tau_id,
                        torsdof if torsdof is not None else '',
                        in_smi, prot_smi])

    # === rejected_ligands.csv: объединяем предфильтр (элементы) + торсии ===
    rejected_csv = os.path.join(output_dir, 'rejected_ligands.csv')
    with open(rejected_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mol_id', 'tau_id', 'smiles', 'stage', 'reason'])
        # Из предфильтра (по элементам / невалидные)
        if prefilter_rejected_path and os.path.exists(prefilter_rejected_path):
            with open(prefilter_rejected_path, encoding='utf-8') as pf_in:
                reader = csv.DictReader(pf_in)
                for row in reader:
                    w.writerow([row.get('ligand_id', ''), '',
                                row.get('SMILES', ''),
                                'prefilter', row.get('reason', '')])
        # Из постфильтра (по торсиям)
        for mol_id, tau_id, smi, torsdof in rejected_tors:
            w.writerow([mol_id, tau_id, smi, 'postfilter',
                        f'too_flexible:{torsdof}_torsions'])

    logging.info(f"Финальный набор для докинга: {len(dockable)} лигандов "
                 f"-> {docking_csv}")
    if rejected_tors:
        logging.info(f"Отсеяно по гибкости (>{max_torsions} торсий): "
                     f"{len(rejected_tors)} -> перемещены в {rejected_dir}/")
    logging.info(f"Полный список отсеянных -> {rejected_csv}")

    return len(dockable), len(rejected_tors)


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(
        description='Подготовка лигандов (.smi -> .pdbqt) для AutoDock Vina. '
                    'Порядок: SMILES -> таутомеры -> Dimorphite-DL -> ETKDG -> Meeko.'
    )
    parser.add_argument('input_file', help='Файл со SMILES, например ligands.smi')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Таймаут на один лиганд (сек)')
    parser.add_argument('--ph', type=float, default=7.4,
                        help='pH для протонирования (default: 7.4)')
    parser.add_argument('--workers', type=int, default=mp.cpu_count())
    parser.add_argument('--num_confs', type=int, default=1,
                        help='Число конформеров ETKDG для эмбеддинга (1=быстро, 3-5=лучше)')
    parser.add_argument('--no-meeko', action='store_true')

    tau_group = parser.add_argument_group('Таутомеры')
    tau_group.add_argument('--tautomers', action='store_true')
    tau_group.add_argument('--tautomer-mode', choices=['score', 'xtb'], default='score')
    tau_group.add_argument('--max-tautomers', type=int, default=3)
    tau_group.add_argument('--score-cutoff', type=float, default=3.0)
    tau_group.add_argument('--energy-cutoff', type=float, default=3.0,
                           help='Отсечение xtb по энергии, ккал/моль (default: 3.0)')
    tau_group.add_argument('--xtb-timeout', type=int, default=60)
    tau_group.add_argument('--xtb-n-confs', type=int, default=5,
                           help='Сколько конформеров на таутомер для xtb (default: 5)')
    tau_group.add_argument('--strict-stereo', dest='strict_stereo',
                           action='store_true', default=True,
                           help='Выбрасывать таутомеры с потерянным стерео (по умолчанию ВКЛ)')
    tau_group.add_argument('--no-strict-stereo', dest='strict_stereo',
                           action='store_false',
                           help='Оставлять таутомеры с потерянным стерео (с warning в логе)')

    parser.add_argument('--input-format', choices=['smi', 'csv'], default='smi')
    parser.add_argument('--id-col', default=None,
                        help='Имя колонки с ID лиганда в CSV (по умолчанию '
                             'автопоиск среди: ligand_id, id, mol_id, '
                             'compound_id, name, и т.п.)')
    parser.add_argument('--smiles-col', default=None,
                        help='Имя колонки со SMILES в CSV (по умолчанию '
                             'автопоиск среди: smiles, canonical_smiles, '
                             'structure, и т.п.)')
    parser.add_argument('--ligand-prefix', default='ligand_')
    parser.add_argument('--tautomer-prefix', default='tau_')

    filter_group = parser.add_argument_group('Предфильтр Vina-совместимости')
    filter_group.add_argument('--no-prefilter', action='store_true',
                              help='Отключить отсев недокируемых лигандов '
                                   '(по умолчанию фильтр ВКЛ)')
    filter_group.add_argument('--allowed-elements', default=None,
                              help='Список разрешённых элементов через запятую '
                                   '(по умолчанию: H,C,N,O,F,P,S,Cl,Br,I). '
                                   'Указывайте, если ваша сборка Vina знает доп. типы.')
    filter_group.add_argument('--max-rotatable-bonds', type=int, default=None,
                              help='ГРУБЫЙ предотсев на входе по числу вращаемых '
                                   'связей (по SMILES). Экономит время на заведомо '
                                   'гигантских молекулах. По умолчанию ВЫКЛ. '
                                   'Точный фильтр — --max-torsions (по PDBQT).')
    filter_group.add_argument('--max-torsions', type=int, default=None,
                              help='ТОЧНЫЙ постфильтр по числу активных торсий '
                                   '(TORSDOF из готового PDBQT — именно это число '
                                   'использует Vina). Жёсткий лимит AutoDock4 = 32 '
                                   '(Morris 2009); Vina надёжна до ~20-30 '
                                   '(Forli 2016). По умолчанию ВЫКЛ. Рекомендация: 32.')
    filter_group.add_argument('--dock-id-prefix', default='D',
                              help='Префикс сквозного индекса в docking_set.csv '
                                   "(по умолчанию 'D' -> D000001, D000002, ...)")

    args = parser.parse_args()

    global USE_MEEKO
    if args.no_meeko:
        USE_MEEKO = False
        logging.info("Meeko отключён — используется obabel для pdbqt")
    else:
        result = subprocess.run(['mk_prepare_ligand.py', '--help'],
                                capture_output=True)
        if result.returncode != 0 and b'usage' not in result.stdout.lower():
            logging.warning("mk_prepare_ligand.py не найден! Установите: pip install meeko")
            logging.warning("Переключаемся на obabel fallback")
            USE_MEEKO = False

    if DIMORPHITE_AVAILABLE:
        logging.info(f"Протонирование: Dimorphite-DL {DIMORPHITE_API} при pH {args.ph}")
    else:
        logging.info(f"Протонирование: obabel CLI на SMILES-уровне при pH {args.ph} "
                     f"(Dimorphite-DL не установлен; для лучшего качества: "
                     f"pip install dimorphite-dl или dimorphite-ojmb)")

    if args.tautomers:
        logging.info(f"Энумерация таутомеров: ВКЛ (mode={args.tautomer_mode}, "
                     f"max={args.max_tautomers}, score_cutoff={args.score_cutoff}"
                     + (f", energy_cutoff={args.energy_cutoff} kcal/mol, "
                        f"xtb_n_confs={args.xtb_n_confs}"
                        if args.tautomer_mode == 'xtb' else '') + ")")
    else:
        logging.info("Энумерация таутомеров: ВЫКЛ")

    for d in FOLDERS.values():
        os.makedirs(d, exist_ok=True)

    # === Этап 0: предфильтр Vina-совместимости ===
    input_file = args.input_file
    prefilter_rejected_path = None
    if not args.no_prefilter:
        if args.allowed_elements:
            allowed = {e.strip() for e in args.allowed_elements.split(',') if e.strip()}
        else:
            allowed = DOCKABLE_ELEMENTS
        logging.info(f"Предфильтр: разрешённые элементы = "
                     f"{', '.join(sorted(allowed))}")
        if args.max_rotatable_bonds is not None:
            logging.info(f"Предфильтр (грубый): макс. вращаемых связей = "
                         f"{args.max_rotatable_bonds}")
        filtered_file, n_kept, n_rejected = prefilter_undockable(
            args.input_file, args.input_format, allowed, FOLDERS['2d'],
            max_rotatable_bonds=args.max_rotatable_bonds,
            id_col=args.id_col, smiles_col=args.smiles_col,
        )
        if n_kept == 0:
            logging.error("После предфильтра не осталось ни одного лиганда — выход.")
            return
        input_file = filtered_file
        prefilter_rejected_path = os.path.join(FOLDERS['2d'], 'ligands_rejected.csv')
    else:
        logging.info("Предфильтр Vina-совместимости: ОТКЛЮЧЁН (--no-prefilter)")

    # === Этап 1: таутомеры + протонирование ===
    tautomer_stats = collections.Counter()
    if args.tautomers:
        expanded_smi = os.path.join(FOLDERS['2d'], 'ligands_tautomers.smi')
        mapping_tsv = os.path.join(FOLDERS['2d'], 'tautomer_mapping.tsv')
        smiles_dict, tautomer_stats = expand_tautomers(
            input_file, expanded_smi, mapping_tsv,
            max_tautomers=args.max_tautomers,
            score_cutoff=args.score_cutoff,
            mode=args.tautomer_mode,
            energy_cutoff=args.energy_cutoff,
            xtb_timeout=args.xtb_timeout,
            xtb_n_confs=args.xtb_n_confs,
            ph=args.ph,
            workers=args.workers,
            input_format=args.input_format,
            ligand_prefix=args.ligand_prefix,
            tautomer_prefix=args.tautomer_prefix,
            strict_stereo=args.strict_stereo,
            id_col=args.id_col,
            smiles_col=args.smiles_col,
        )
        working_smi = expanded_smi
    else:
        # Без энумерации: каждый SMILES = tau_id=1, но всё равно протонируем
        smiles_dict = {}
        working_smi = os.path.join(FOLDERS['2d'], 'ligands_single.smi')
        output_csv = os.path.join(FOLDERS['2d'], 'ligands_out.csv')
        originals = _read_input_file(input_file, args.input_format,
                                      id_col=args.id_col, smiles_col=args.smiles_col)

        with open(working_smi, 'w', encoding='utf-8') as fsmi, \
             open(output_csv, 'w', encoding='utf-8', newline='') as fcsv:
            csv_writer = csv.writer(fcsv)
            csv_writer.writerow(['ligand_id', 'tau_id', 'input_smiles', 'protonated_smiles'])
            for mol_id, smi in originals:
                prot_smi = protonate_smiles_dimorphite(smi, args.ph, mol_tag=f"mol{mol_id}")
                tag = f"{args.ligand_prefix}{mol_id}_{args.tautomer_prefix}1"
                fsmi.write(f"{prot_smi}\t{tag}\n")
                csv_writer.writerow([mol_id, 1, smi, prot_smi])
                smiles_dict[(str(mol_id), 1)] = {'input': smi, 'protonated': prot_smi}
                if prot_smi != smi:
                    tautomer_stats['protonation_changed'] += 1
                else:
                    tautomer_stats['protonation_unchanged'] += 1

    logging.info(f"Всего структур для обработки: {len(smiles_dict)}")

    # === Список ключей для обработки ===
    # ETKDG идёт напрямую из smiles_dict (protonated SMILES), без массового gen2D.
    # 2D SDF создаётся лениво в process_single_ligand только при obabel-fallback.
    keys = sorted(
        smiles_dict.keys(),
        key=lambda k: (
            int(k[0]) if str(k[0]).isdigit() else 10**9,
            int(k[1]) if k[1] is not None else 0,
        )
    )

    if not keys:
        logging.error("Нет лигандов для обработки!")
        return

    logging.info(f"Обработка {len(keys)} структур | workers={args.workers} | "
                 f"num_confs={args.num_confs} | meeko={USE_MEEKO} | "
                 f"dimorphite={DIMORPHITE_AVAILABLE}")

    with mp.Pool(args.workers) as pool:
        func = partial(process_single_ligand, args=args, smiles_dict=smiles_dict)
        results = list(tqdm(pool.imap(func, keys), total=len(keys), desc="Обработка"))

    sdf_list = keys  # для совместимости с отчётом ниже (len(sdf_list))

    # ===================== РАСШИРЕННЫЙ ОТЧЁТ =====================
    # Распарсим коды результатов "<embed>|<pdbqt>[|mmff_uff]"
    embed_counter = collections.Counter()
    pdbqt_counter = collections.Counter()
    mmff_uff_fallbacks = 0
    skipped = 0
    failed = 0
    for code in results:
        if code == 'skipped':
            skipped += 1
            continue
        if code == 'failed':
            failed += 1
            continue
        parts = str(code).split('|')
        if len(parts) >= 2:
            embed_counter[parts[0]] += 1
            pdbqt_counter[parts[1]] += 1
            if 'mmff_uff' in parts:
                mmff_uff_fallbacks += 1
        else:
            embed_counter['unknown'] += 1

    total = len(sdf_list)
    success = total - failed

    logging.info("=" * 70)
    logging.info("ГОТОВО — расширенная статистика")
    logging.info("=" * 70)

    # === Блок 1: общие итоги обработки ===
    logging.info(f"Всего структур на обработке  : {total}")
    if args.tautomers:
        n_originals = tautomer_stats.get('input_mols',
                          len(set(mol_id for (mol_id, _) in smiles_dict.keys())))
        logging.info(f"Исходных молекул             : {n_originals}")
        logging.info(f"Среднее таутомеров/мол       : {total/max(1,n_originals):.2f}")
    logging.info(f"Успешно создано PDBQT        : {success}/{total} "
                 f"({100*success//max(1,total)}%)")
    logging.info(f"Уже были (пропущено)         : {skipped}")
    logging.info(f"Не удалось                   : {failed}")
    logging.info("")

    # === Блок 2: эмбеддинг 3D ===
    logging.info("--- 3D-эмбеддинг ---")
    for method, n in sorted(embed_counter.items(), key=lambda x: -x[1]):
        logging.info(f"  {method:30s}: {n}")
    if mmff_uff_fallbacks > 0:
        logging.info(f"  (из них с UFF-fallback от MMFF: {mmff_uff_fallbacks})")
    logging.info("")

    # === Блок 3: бэкенд PDBQT ===
    logging.info("--- Бэкенд PDBQT ---")
    for backend, n in sorted(pdbqt_counter.items(), key=lambda x: -x[1]):
        logging.info(f"  {backend:30s}: {n}")
    logging.info("")

    # === Блок 4: статистика таутомеров и протонирования ===
    if args.tautomers and tautomer_stats:
        logging.info("--- Энумерация таутомеров ---")
        logging.info(f"  всего создано таутомеров   : {tautomer_stats.get('tautomers_total', 0)}")
        logging.info(f"  молекул с >1 таутомером    : "
                     f"{tautomer_stats.get('mol_ids_with_multiple', 0)}")
        logging.info(f"  схлопнуто (резонансные)    : "
                     f"{tautomer_stats.get('tautomers_deduped', 0)}")
        logging.info(f"  молекул без таутомеров     : "
                     f"{tautomer_stats.get('mol_no_tautomers', 0)}")

        # === Стерео-отброс (strict-stereo) ===
        stereo_dropped = tautomer_stats.get('stereo_dropped_tautomers', 0)
        stereo_mols = tautomer_stats.get('mol_ids_with_stereo_loss', 0)
        if args.strict_stereo:
            if stereo_dropped > 0:
                logging.info(f"  отброшено по стерео        : {stereo_dropped} "
                             f"(в {stereo_mols} молекулах)")
            else:
                logging.info(f"  отброшено по стерео        : 0 "
                             f"(strict-stereo ВКЛ, ни один не потерял хиральность)")
        else:
            logging.info(f"  таутомеров с потерянным стерео: {stereo_dropped} "
                         f"(ОСТАВЛЕНЫ: --no-strict-stereo)")

        xtb_failed = tautomer_stats.get('label_xtb_failed', 0)
        xtb_ok = tautomer_stats.get('label_xtb', 0)
        score_only = tautomer_stats.get('label_score_only', 0)
        if args.tautomer_mode == 'xtb':
            logging.info(f"  xtb успешно                : {xtb_ok}")
            logging.info(f"  xtb упал (penalty 999)     : {xtb_failed}")
        elif score_only > 0:
            logging.info(f"  score-only (без xtb)       : {score_only}")

        crash_fb = tautomer_stats.get('label_crash_fallback', 0)
        no_enum = tautomer_stats.get('label_fallback_no_enum', 0)
        if crash_fb > 0:
            logging.info(f"  fallback (crash в RDKit)   : {crash_fb}")
        if no_enum > 0:
            logging.info(f"  fallback (no enumeration)  : {no_enum}")
        logging.info("")

    # === Блок 5: протонирование ===
    if tautomer_stats:
        p_changed = tautomer_stats.get('protonation_changed', 0)
        p_unchanged = tautomer_stats.get('protonation_unchanged', 0)
        if p_changed + p_unchanged > 0:
            logging.info("--- Протонирование ---")
            logging.info(f"  SMILES изменился (pH 7.4)  : {p_changed}")
            logging.info(f"  SMILES не изменился        : {p_unchanged}")
            logging.info("")

    logging.info(f"Время завершения             : {datetime.now().strftime('%H:%M:%S')}")
    logging.info("=" * 70)

    # ===================== ФИНАЛИЗАЦИЯ: индексация + торсионный постфильтр =====================
    if args.max_torsions is not None:
        logging.info(f"Постфильтр по торсиям: макс. {args.max_torsions} активных "
                     f"торсий (TORSDOF из PDBQT)")
    n_dockable, n_tors_rejected = finalize_docking_set(
        FOLDERS['pdbqt'], smiles_dict, FOLDERS['2d'],
        ligand_prefix=args.ligand_prefix,
        tautomer_prefix=args.tautomer_prefix,
        max_torsions=args.max_torsions,
        dock_id_prefix=args.dock_id_prefix,
        prefilter_rejected_path=prefilter_rejected_path,
    )
    logging.info("=" * 70)
    logging.info(f"ИТОГО годных для докинга : {n_dockable}")
    logging.info(f"docking_set.csv          : 2d/docking_set.csv")
    logging.info(f"rejected_ligands.csv     : 2d/rejected_ligands.csv")
    logging.info("=" * 70)


if __name__ == '__main__':
    main()
