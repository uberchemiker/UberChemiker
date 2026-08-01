#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_setup.py — проверка, что окружение готово к работе пайплайна.

Проверяет:
  - какой Python запущен (ловушка python vs python3),
  - Python-модули (gemmi, meeko, pdbfixer, openmm, rdkit, numpy, propka),
  - CLI-инструменты (mk_prepare_receptor.py, mk_prepare_ligand.py, obabel,
    vina, propka/pdb2pqr).

Запуск:   python check_setup.py
ВАЖНО: запускайте ИМЕННО тем python, которым будете запускать пайплайн!
"""

import importlib
import shutil
import subprocess
import sys


def check_module(name, import_name=None):
    """Проверить, импортируется ли модуль. Вернуть (ok, версия|сообщение)."""
    try:
        mod = importlib.import_module(import_name or name)
        ver = getattr(mod, "__version__", "?")
        return True, ver
    except Exception as e:
        return False, str(e).split("\n")[0][:60]


def check_cli(cmd):
    """Проверить, есть ли команда в PATH."""
    path = shutil.which(cmd)
    return (path is not None), (path or "не найдена в PATH")


def main():
    print("=" * 64)
    print("ПРОВЕРКА ОКРУЖЕНИЯ ПАЙПЛАЙНА ПОДГОТОВКИ РЕЦЕПТОРОВ")
    print("=" * 64)

    # 0) какой Python
    print("\n[Python]")
    print(f"  Интерпретатор: {sys.executable}")
    print(f"  Версия: {sys.version.split()[0]}")
    print("  ВАЖНО: запускайте пайплайн ИМЕННО этим python (не обязательно "
          "python3!).")
    print("  Если 'python' и 'python3' указывают на разные окружения —")
    print("  используйте то имя, при котором эта проверка проходит.")

    # 1) Python-модули
    print("\n[Python-модули]")
    modules = [
        ("gemmi", None, "чтение .pdb/.cif, геометрия"),
        ("meeko", None, "подготовка рецептора/лиганда для AutoDock"),
        ("pdbfixer", None, "достройка боковых цепей (Стадия 2.5)"),
        ("openmm", None, "нужен для PDBFixer"),
        ("rdkit", "rdkit", "RMSD в редокинг-валидации"),
        ("numpy", None, "вычисления"),
        ("propka", None, "pKa для His-таутомеризации близких остатков"),
    ]
    all_mod_ok = True
    for name, imp, descr in modules:
        ok, info = check_module(name, imp)
        mark = "OK  " if ok else "НЕТ "
        print(f"  [{mark}] {name:12} {('v'+info) if ok else info:24} — {descr}")
        if not ok:
            all_mod_ok = False

    # 2) CLI-инструменты
    print("\n[CLI-инструменты]")
    clis = [
        ("mk_prepare_receptor.py", "Meeko: сборка рецептора .pdbqt"),
        ("mk_prepare_ligand.py", "Meeko: подготовка лиганда (редокинг)"),
        ("obabel", "OpenBabel: fallback-сборка и конвертация"),
        ("vina", "AutoDock Vina: редокинг-валидация (опционально)"),
    ]
    all_cli_ok = True
    for cmd, descr in clis:
        ok, info = check_cli(cmd)
        mark = "OK  " if ok else "НЕТ "
        print(f"  [{mark}] {cmd:24} — {descr}")
        if not ok and cmd != "vina":   # vina нужна только для редокинга
            all_cli_ok = False
        if not ok:
            print(f"           ({info})")

    # 3) propka как модуль запуска
    print("\n[PROPKA как 'python -m propka']")
    try:
        r = subprocess.run([sys.executable, "-m", "propka", "--version"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0 or "propka" in (r.stdout + r.stderr).lower():
            print("  [OK  ] python -m propka запускается")
        else:
            print("  [?   ] python -m propka вернул код", r.returncode)
    except Exception as e:
        print(f"  [НЕТ ] python -m propka не запускается: {e}")

    # Итог
    print("\n" + "=" * 64)
    if all_mod_ok and all_cli_ok:
        print("ИТОГ: всё основное на месте. Можно запускать пайплайн.")
        if not check_cli("vina")[0]:
            print("Примечание: 'vina' не найдена — нужна только для "
                  "validate_redock.py.")
    else:
        print("ИТОГ: чего-то не хватает (см. [НЕТ] выше).")
        print("Установка: см. README.md, раздел «Установка окружения».")
    print("=" * 64)


if __name__ == "__main__":
    main()
