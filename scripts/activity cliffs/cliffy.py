from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
import numpy as np
import pandas as pd


def _fp_matrix(fps):
    n_bits = fps[0].GetNumBits()
    arr = np.zeros((len(fps), n_bits), dtype=np.float32)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def _tanimoto_matrix(arr):
    intersection = arr @ arr.T
    row_sums = arr.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def find_cliff(smiles, activities=None, maccs_thr=0.8, ecfp4_thr=0.4, act_thr=None):
    non_str = [i for i, s in enumerate(smiles) if not isinstance(s, str)]
    if non_str:
        raise ValueError(f"non-string SMILES (e.g. NaN) at indices: {non_str}")

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    bad = [s for s, m in zip(smiles, mols) if m is None]
    if bad:
        raise ValueError(f"invalid SMILES: {bad}")

    maccs_fps = [MACCSkeys.GenMACCSKeys(m) for m in mols]
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    ecfp4_fps = [morgan_gen.GetFingerprint(m) for m in mols]

    maccs_sim = _tanimoto_matrix(_fp_matrix(maccs_fps))
    ecfp4_sim = _tanimoto_matrix(_fp_matrix(ecfp4_fps))

    n = len(smiles)
    i_idx, j_idx = np.triu_indices(n, k=1)
    mask = (maccs_sim[i_idx, j_idx] >= maccs_thr) | (ecfp4_sim[i_idx, j_idx] >= ecfp4_thr)

    smiles_arr = np.asarray(smiles, dtype=object)
    out = {
        "smiles1": smiles_arr[i_idx[mask]],
        "smiles2": smiles_arr[j_idx[mask]],
        "maccs_tanimoto": maccs_sim[i_idx, j_idx][mask],
        "ecfp4_tanimoto": ecfp4_sim[i_idx, j_idx][mask],
    }

    if activities is not None:
        act = np.asarray(activities, dtype=float)
        out["activity_diff"] = np.abs(act[i_idx] - act[j_idx])[mask]

    df = pd.DataFrame(out)
    if activities is not None and act_thr is not None:
        df = df[df["activity_diff"] >= act_thr].reset_index(drop=True)
    return df
