# Подготовка лигандов для AutoDock Vina — руководство

Скрипт `prepare_ligands_UPD4.py` готовит наборы лигандов для докинга в AutoDock
Vina / Vina-GPU. На входе SMILES, на выходе PDBQT-файлы с правильной 3D-геометрией,
протонированием при физиологическом pH и опциональной таутомерной обработкой.

---

## Что делает скрипт (пайплайн)

```
ВХОД: SMI или CSV со SMILES
   │
   ▼
[Этап 0] Предфильтр Vina-совместимости (по SMILES)
   • Невалидные SMILES → reject
   • Элементы вне {H,C,N,O,F,P,S,Cl,Br,I} → reject
   • (опц.) Слишком много вращаемых связей → reject
   │
   ▼
[Этап 1] Энумерация таутомеров (RDKit)
   • Стерео сохраняется (R/S и E/Z)
   • Дедупликация по isomeric SMILES
   • Ранжирование по ScoreTautomer
   • Pre-cap: топ-N по score идут дальше (N = --max-tautomers)
   │
   ▼
[Этап 2 — опционально] xtb-фильтр таутомеров
   • Для каждого таутомера: N конформеров → MMFF94s → GFN2-xTB+ALPB(water)
   • Берётся минимум по конформерам
   • Отсев: ΔE > energy_cutoff (по умолч. 3 ккал/моль)
   │
   ▼
[Этап 3] Протонирование при pH 7.4 (Dimorphite-DL)
   • Каскад: Dimorphite-DL → obabel fallback → исходный
   │
   ▼
[Этап 4] Пост-дедупликация (fixed-H InChIKey)
   • Резонансные формы анионов схлопываются
   • Реальные таутомеры (амид/иминол) сохраняются
   │
   ▼
[Этап 5-6] 3D-эмбеддинг
   • RDKit ETKDGv3 напрямую из SMILES (без gen2D)
   • MMFF94s оптимизация (UFF fallback для Se)
   • num_confs конформеров, выбор лучшего
   • Fallback: obabel gen3D, если ETKDG не справился
   │
   ▼
[Этап 7-8] PDBQT через Meeko
   • mk_prepare_ligand.py
   • Fallback: obabel
   │
   ▼
[Этап 9] REMARK SMILES + provenance в PDBQT
   │
   ▼
[Этап 10] Финализация
   • Подсчёт TORSDOF из готовых PDBQT
   • Постфильтр: >--max-torsions → pdbqt_rejected/
   • Сквозная индексация D000001, D000002, …
   │
   ▼
ВЫХОД:
   pdbqt/*.pdbqt             — лиганды для Vina
   2d/docking_set.csv        — индекс годных (с mol_id/tau_id/SMILES/n_torsions)
   2d/rejected_ligands.csv   — отсеянные с причинами
   2d/tautomer_mapping.tsv   — провенанс таутомеров
   3d/*.sdf, mol2/*.mol2     — промежуточные структуры
```

---

## Установка зависимостей

Скрипт нужен Python 3.9+. Все зависимости лучше ставить в отдельный conda env.

### Создание окружения
```bash
conda create -n meeko python=3.9 -y
conda activate meeko
```

### Обязательные пакеты
```bash
# RDKit (химия + 3D)
conda install -c conda-forge rdkit -y

# OpenBabel (резервный путь, форматы)
conda install -c conda-forge openbabel -y

# Meeko (PDBQT-генерация по AutoDock-стандарту)
pip install meeko

# tqdm (прогресс-бар)
pip install tqdm
```

### Сильно рекомендуемые
```bash
# Dimorphite-DL (протонирование при pH; для Python 3.9 используем форк ojmb)
pip install dimorphite-ojmb

# xtb (полуэмпирическая квантовая химия для отбора таутомеров)
conda install -c conda-forge xtb -y
```

### Проверка установки
```bash
python -c "from rdkit import Chem; print('RDKit OK')"
python -c "import meeko; print('Meeko OK')"
obabel -V
which xtb
python -c "import dimorphite_dl; print('Dimorphite OK')"
```

---

## Формат входных данных

Скрипт принимает два формата:

### `.smi` — один SMILES на строку
```
CCO ethanol
c1ccccc1 benzene
CC(=O)O acetic_acid
```
Имя лиганда (второй столбец) опционально. Если нет — будет порядковый номер.

### `.csv` — стандартный CSV с заголовками
```csv
ligand_id,SMILES
00001,CC(=O)Oc1ccccc1C(=O)O
00002,CN1CCCC1c1cccnc1
```

**Колонки определяются автоматически.** Скрипт ищет SMILES среди:
`smiles`, `canonical_smiles`, `isomeric_smiles`, `structure`, `mol`, `smiles_str`.
ID — среди: `ligand_id`, `id`, `mol_id`, `compound_id`, `name`, `compound`, `cid`,
`chembl_id`, `zinc_id`, и т.п.

Регистр не важен. Разделитель `,` или `;` определяется автоматически.
Лишние колонки (`activity`, `source`, `mw`...) игнорируются.

Если автопоиск не сработал — укажите явно:
```bash
--id-col my_id --smiles-col canonical_smiles
```

---

## Базовые команды

### Самый простой запуск (без таутомеров)
```bash
python prepare_ligands_UPD4.py ligands.smi --input-format smi \
    --ph 7.4 --workers 6
```
На каждый входной SMILES получите один протонированный лиганд в PDBQT.

### С быстрой таутомерной обработкой (рекомендуется по умолчанию)
```bash
python prepare_ligands_UPD4.py ligands.smi --input-format smi \
    --tautomers --tautomer-mode score \
    --max-tautomers 3 --num_confs 5 \
    --ph 7.4 --workers 6 \
    --max-torsions 32
```
- `--tautomers` — включает энумерацию
- `--tautomer-mode score` — отбор по эмпирическому скору RDKit (быстро)
- `--max-tautomers 3` — оставить до 3 топовых таутомеров на молекулу
- `--max-torsions 32` — отсечь сверхгибкие лиганды (Vina ненадёжна, см. Morris 2009)

### С качественной xtb-фильтрацией таутомеров (для важных молекул)
```bash
python prepare_ligands_UPD4.py ligands.smi --input-format smi \
    --tautomers --tautomer-mode xtb \
    --max-tautomers 3 --xtb-n-confs 5 --energy-cutoff 3.0 \
    --num_confs 5 --ph 7.4 --workers 6 \
    --max-torsions 32
```
xtb считает GFN2-xTB энергии каждого таутомера в воде (ALPB) и отбраковывает
формы, лежащие выше `--energy-cutoff` ккал/моль от минимума.

### Для CSV с нестандартными именами колонок
```bash
python prepare_ligands_UPD4.py library.csv --input-format csv \
    --id-col compound_name --smiles-col canonical_smiles \
    --tautomers --tautomer-mode score \
    --max-tautomers 3 --ph 7.4 --workers 6 \
    --max-torsions 32
```

---

## Ключевые флаги — что зачем

| Флаг | Назначение | По умолчанию |
|---|---|---|
| `--input-format` | smi или csv | smi |
| `--id-col` / `--smiles-col` | Явные имена CSV-колонок | автопоиск |
| `--tautomers` | Включить энумерацию таутомеров | выкл |
| `--tautomer-mode` | `score` (быстро) или `xtb` (точно) | score |
| `--max-tautomers` | Топ-N таутомеров на молекулу | 3 |
| `--score-cutoff` | Отсечка по score (от лучшего) | 3.0 |
| `--energy-cutoff` | Отсечка xtb по ΔE (ккал/моль) | 3.0 |
| `--xtb-n-confs` | Конформеров на таутомер для xtb | 5 |
| `--xtb-timeout` | Таймаут xtb на один конформер (сек) | 60 |
| `--strict-stereo` / `--no-strict-stereo` | Отбрасывать таутомеры с потерянным стерео (R/S и E/Z) | ВКЛ |
| `--ph` | pH для протонирования | 7.4 |
| `--num_confs` | Конформеров для финального 3D-эмбеддинга | 5 |
| `--workers` | Параллельных процессов | 4 |
| `--max-torsions` | Постфильтр по TORSDOF из PDBQT | None (выкл) |
| `--max-rotatable-bonds` | Грубый предотсев на входе по SMILES | None (выкл) |
| `--no-prefilter` | Отключить фильтр по элементам | предфильтр ВКЛ |
| `--allowed-elements` | Кастомный набор элементов | H,C,N,O,F,P,S,Cl,Br,I |
| `--dock-id-prefix` | Префикс для сквозной индексации | `D` |
| `--no-meeko` | Использовать только obabel для PDBQT | Meeko предпочтителен |

---

## Структура выходных данных

После запуска в рабочей директории появятся:

```
2d/
  ligands_dockable.smi       — то, что прошло предфильтр (идёт в обработку)
  ligands_rejected.csv       — отсеянные предфильтром (элементы)
  ligands_tautomers.smi      — расширенный набор с таутомерами
  tautomer_mapping.tsv       — провенанс: mol_id → tau_id → SMILES + score + ΔE + label
  ligands_out.csv            — таблица всех (mol_id, tau_id, input_smi, prot_smi)
  docking_set.csv            — ⭐ ФИНАЛЬНЫЙ ИНДЕКС для докинга
  rejected_ligands.csv       — ⭐ ВСЕ отсеянные (prefilter + postfilter) с причинами
3d/
  ligand_*_tau_*_3D.sdf      — 3D-структуры из ETKDG
3d_pH/
  (используется только если Dimorphite-DL не установлен)
mol2/
  ligand_*_tau_*.mol2        — для визуальной проверки (НЕ для докинга)
pdbqt/
  ligand_*_tau_*.pdbqt       — ⭐ ВХОД для Vina
pdbqt_rejected/
  (если задан --max-torsions, сюда уезжают сверхгибкие)
```

### Главные файлы для дальнейшей работы

**`2d/docking_set.csv`** — главный индекс. Колонки:
```
dock_id, pdbqt_file, mol_id, tau_id, n_torsions, input_smiles, protonated_smiles
D000001, ligand_00001_tau_1.pdbqt, 00001, 1, 5, CCO, CCO
```
Сквозной `dock_id` (D000001, D000002, …) — для удобной трассировки в докинге.

**`2d/rejected_ligands.csv`** — что не дошло до Vina:
```
mol_id, tau_id, smiles, stage, reason
00135, , C[Si]..., prefilter, unsupported_elements:Si
00004, 1, CCCC..., postfilter, too_flexible:45_torsions
```
Колонка `stage` различает: `prefilter` (по элементам на входе) vs `postfilter`
(по торсиям на выходе).

---

## Расширенная статистика в логе

После завершения скрипт выводит структурированный отчёт:

```
======================================================================
ГОТОВО — расширенная статистика
======================================================================
Всего структур на обработке  : 88
Исходных молекул             : 71
Среднее таутомеров/мол       : 1.24
Успешно создано PDBQT        : 88/88 (100%)
Уже были (пропущено)         : 0
Не удалось                   : 0

--- 3D-эмбеддинг ---
  ETKDG_5confs                  : 88

--- Бэкенд PDBQT ---
  meeko                         : 88

--- Энумерация таутомеров ---
  всего создано таутомеров   : 88
  молекул с >1 таутомером    : 17
  схлопнуто (резонансные)    : 1
  молекул без таутомеров     : 0
  отброшено по стерео        : 516 (в 20 молекулах)
  xtb успешно                : 56

--- Протонирование ---
  SMILES изменился (pH 7.4)  : 70
  SMILES не изменился        : 19
```

Эта секция показывает реальную картину: где пайплайн пошёл по основному
пути, где пришлось переключиться на fallback.

---

## Стереохимия — важный нюанс

По умолчанию `--strict-stereo` **ВКЛЮЧЕН**: скрипт выбрасывает таутомеры,
у которых при таутомеризации:
- Потерян sp3-хиральный центр (R/S) — `[C@H]` → плоский `=C<`
- Потеряна E/Z-конфигурация двойной связи — `/C=C/` → одинарная связь без стерео

**Это важно знать:** даже если у вас в SMILES нет `[C@H]/[C@@H]`, но есть `/` или
`\` для E/Z двойных связей — стерео-проверка их тоже учитывает. Это правильно:
изомеры по двойной связи часто отличаются по биологической активности.

Если хотите видеть **все** таутомеры независимо от стерео — используйте
`--no-strict-stereo`. Будьте готовы, что:
- В докинг попадут «плоские» таутомеры без привязки к исходной хиральности
- Время xtb увеличится (больше таутомеров для расчёта)

---

## Ориентиры по времени

На MacBook Pro M2 (8 ядер) с `--workers 6`:

| Задача | Размер выборки | Время |
|---|---|---|
| score-режим, drug-like (≤40 атомов) | 100 мол | 1-2 мин |
| score-режим, drug-like | 1000 мол | 10-15 мин |
| score-режим, drug-like | 10000 мол | 1.5-2 ч |
| xtb-режим, drug-like (≤40 атомов) | 71 мол | 15-20 мин |
| xtb-режим, drug-like | 1000 мол | 3-4 ч |
| xtb-режим, drug-like | 5000+ мол | overnight |

**xtb-режим в 10-30× медленнее score-режима**, поэтому используйте его только
когда таутомерная определённость критична. Для большого скрининга — `score`.

### Если хочется быстрее с xtb
- `--xtb-n-confs 3` вместо 5 (даёт +60% скорости, теряете немного точности)
- `--score-cutoff 1.5` вместо 3.0 (строже предотсев до xtb)
- На M2/M3 убедитесь, что `export OMP_NUM_THREADS=1` (иначе xtb сам распараллелится
  и будет драться с воркерами за CPU)

---

## Стратегия для больших разнородных библиотек

Если у вас 8000+ лигандов смешанного размера (от drug-like до пептидоподобных),
используйте `split_by_size.py` для разделения:

```bash
python split_by_size.py library.csv --cutoff 40
```

Получите:
- `ligands_small.csv` (≤40 атомов) — годятся для xtb-режима
- `ligands_large.csv` (>40 атомов) — для score-режима (xtb на крупных молекулах
  и медленный, и научно ненадёжный — 5 конформеров не покрывают пространство)
- `ligands_problem.csv` (As, Se, Si и пр.) — отдельная обработка

Дальше:
```bash
# Мелкие: xtb (overnight)
nohup python prepare_ligands_UPD4.py ligands_small.csv --input-format csv \
    --tautomers --tautomer-mode xtb \
    --max-tautomers 3 --xtb-n-confs 5 \
    --num_confs 5 --ph 7.4 --workers 6 \
    --max-torsions 32 \
    > xtb.log 2>&1 &

# Крупные: score (быстро)
python prepare_ligands_UPD4.py ligands_large.csv --input-format csv \
    --tautomers --tautomer-mode score \
    --max-tautomers 3 --num_confs 3 \
    --ph 7.4 --workers 6 \
    --max-torsions 32
```

---

## Типичные проблемы и их диагностика

### `Dimorphite-DL not found, протонирование через obabel`
Не критично, но протонирование будет менее точным. Установите:
```bash
# Для Python 3.9
pip install dimorphite-ojmb
# Для Python 3.10+
pip install dimorphite-dl
```

### `Meeko не справился, пробуем obabel fallback`
Иногда Meeko падает на экзотических группах. obabel подхватит. Если процент
fallback в финальном отчёте >5% — стоит проверить, что Meeko установлен правильно
(`pip install --upgrade meeko`).

### `xtb timeout (conf X) для mol_Y`
Для крупных или экзотических молекул xtb может не уложиться в 60 секунд.
Поднимите таймаут: `--xtb-timeout 180`.

### `Fallback OpenBabel gen3D для ligand_X`
ETKDG не справился — обычно для очень специфичных макроциклов. obabel сделает
3D, но качество ниже.

### Молекул в `docking_set.csv` меньше, чем ожидалось
Проверьте `rejected_ligands.csv` — там будут причины. Группируйте по `stage`
и `reason` чтобы понять, что и где отсеялось.

---

## Литературные ссылки для отчётности

- AutoDock Vina (основная): Trott, O.; Olson, A.J. **AutoDock Vina: improving
  the speed and accuracy of docking with a new scoring function**. *J. Comput.
  Chem.* **31**, 455–461 (2010).
- AutoDock 4 (откуда `MAX_TORS=32`): Morris, G.M. *et al.* **AutoDock4 and
  AutoDockTools4: Automated docking with selective receptor flexibility**.
  *J. Comput. Chem.* **30**, 2785–2791 (2009).
- Протокол AutoDock suite: Forli, S. *et al.* **Computational protein-ligand
  docking and virtual drug screening with the AutoDock suite**. *Nat. Protoc.*
  **11**, 905–919 (2016).
- xTB (GFN2): Bannwarth, C.; Ehlert, S.; Grimme, S. **GFN2-xTB**. *J. Chem.
  Theory Comput.* **15**, 1652–1671 (2019).
- ETKDG: Riniker, S.; Landrum, G.A. **Better Informed Distance Geometry**.
  *J. Chem. Inf. Model.* **55**, 2562–2574 (2015).
- Dimorphite-DL: Ropp, P.J. *et al.* **Dimorphite-DL**. *J. Cheminform.*
  **11**, 14 (2019).
- Meeko: https://github.com/forlilab/Meeko

---

## Минимальный чек-лист перед запуском

1. ✅ Conda env `meeko` создан, активирован
2. ✅ RDKit, Meeko, OpenBabel установлены, проверены
3. ✅ Dimorphite-DL установлен (`dimorphite-ojmb` для Python 3.9)
4. ✅ xtb установлен (если планируется `--tautomer-mode xtb`)
5. ✅ Входной файл подготовлен (SMILES в первой колонке валидны)
6. ✅ Свободно >5 ГБ на диске на каждые 5000 лигандов
7. ✅ Для xtb-прогонов: `export OMP_NUM_THREADS=1`
8. ✅ Для долгих прогонов: `nohup ... &` + контроль автосна системы

После запуска первое, что смотрите — расширенная статистика в конце лога.
Она покажет, где пайплайн отработал чисто, а где пришлось задействовать
fallback или отбрасывать молекулы.
