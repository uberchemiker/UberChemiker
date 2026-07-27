# Scripts

## draw_scaf.py
 
Extracts Murcko scaffolds from SMILES columns in a CSV and draws them as
A4 PNG grids. Also saves a per-column CSV with scaffold counts.
 
### Run
 
```bash
python draw_scaffolds.py --csv data.csv --columns col1 col2 col3
```
 
| Flag        | Description                                  | Default              |
|-------------|-----------------------------------------------|-----------------------|
| `--csv`     | input CSV path                                 | `DEFAULT_CSV_PATH`    |
| `--columns` | one or more SMILES columns (space-separated)   | `DEFAULT_COLUMNS`     |
| `--out-dir` | output folder for PNGs/CSVs                    | `figures`             |
| `--top-n`   | only draw the N most common scaffolds/column   | all                   |
 
No flags needed if you set `DEFAULT_CSV_PATH` / `DEFAULT_COLUMNS` at the
top of the script.
 
### Output (per column, in `--out-dir`)
 
- `scaffolds_<column>.png` (or `_p1.png`, `_p2.png`, ... if it spans
  multiple A4 pages)
- `scaffolds_<column>.csv` with `num`, `scaffold_smiles`, `n_molecules`, `n_rows`
Each structure's label is `id (n molecules with this scaffold)`.
 
### Other settings (top of script)
 
`MOLS_PER_ROW`, `GENERIC`, `KEEP_STEREO`, `SORT_BY` (`"rows"` or
`"molecules"`, doesn't affect the label), `DPI`, page size/margins.
