"""
Разбивает CSV с лигандами (ligand_id, SMILES) на группы по числу тяжёлых атомов.

Зачем: xtb-фильтрация таутомеров вычислительно осмысленна только для
мелких/средних молекул. Для крупных гибких молекул (>~40 атомов) xtb и
слишком медленный, и ненадёжный (5 конформеров не покрывают конформационное
пространство), поэтому для них лучше использовать быстрый score-режим.

Использование:
    python split_by_size.py ligands.csv --cutoff 40

Создаёт:
    ligands_small.csv  (<= cutoff атомов)  -> запускать с --tautomer-mode xtb
    ligands_large.csv  (>  cutoff атомов)  -> запускать с --tautomer-mode score
    ligands_problem.csv (As/металлы и пр.) -> обработать отдельно / проверить
"""
import argparse
import csv
import sys
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# Элементы, которые MMFF94 не параметризует и которые требуют осторожности
PROBLEM_ELEMENTS = {'As', 'Se', 'Te', 'B', 'Si', 'Sb', 'Bi'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv')
    parser.add_argument('--cutoff', type=int, default=40,
                        help='Граница по числу тяжёлых атомов (default: 40)')
    parser.add_argument('--id-col', default='ligand_id')
    parser.add_argument('--smiles-col', default='SMILES')
    args = parser.parse_args()

    rows = []
    with open(args.input_csv, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row[args.id_col].strip(), row[args.smiles_col].strip()))

    small, large, problem, invalid = [], [], [], []

    for lid, smi in rows:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append((lid, smi))
            continue
        elements = {a.GetSymbol() for a in mol.GetAtoms()}
        if elements & PROBLEM_ELEMENTS:
            problem.append((lid, smi))
            continue
        if mol.GetNumHeavyAtoms() <= args.cutoff:
            small.append((lid, smi))
        else:
            large.append((lid, smi))

    def write_csv(path, data):
        with open(path, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['ligand_id', 'SMILES'])
            w.writerows(data)

    write_csv('ligands_small.csv', small)
    write_csv('ligands_large.csv', large)
    write_csv('ligands_problem.csv', problem)
    if invalid:
        write_csv('ligands_invalid.csv', invalid)

    print(f"Всего: {len(rows)}")
    print(f"  Мелкие (<={args.cutoff} ат)  -> ligands_small.csv   : {len(small)}  [xtb]")
    print(f"  Крупные (>{args.cutoff} ат)  -> ligands_large.csv   : {len(large)}  [score]")
    print(f"  Проблемные (As/Se/...)      -> ligands_problem.csv : {len(problem)}  [проверить]")
    if invalid:
        print(f"  Невалидные SMILES           -> ligands_invalid.csv : {len(invalid)}")
    print()
    print("Дальше:")
    print("  1. xtb для мелких:")
    print("     python prepare_ligands_UPD4.py ligands_small.csv --input-format csv \\")
    print("         --tautomers --tautomer-mode xtb --max-tautomers 3 --xtb-n-confs 5 \\")
    print("         --energy-cutoff 3.0 --num_confs 5 --ph 7.4 --workers 6")
    print("  2. score для крупных:")
    print("     python prepare_ligands_UPD4.py ligands_large.csv --input-format csv \\")
    print("         --tautomers --tautomer-mode score --max-tautomers 3 \\")
    print("         --num_confs 5 --ph 7.4 --workers 6")


if __name__ == '__main__':
    main()
