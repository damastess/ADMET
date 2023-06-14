import prolif as plf
import os
import pandas as pd
import numpy as np
import re
import yaml

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

with open('configs/paths.yaml', "r") as f:
    config = yaml.safe_load(f)

bundled_fp_dir = config['bundled_fp_dir']
separate_fps_dir = config['separate_fps_dir']
proteins_dir = config['proteins_dir']
ligands_dir = config['ligands_dir']

def prepare_df(target):
    separate_fp_files_list = sorted(os.listdir(separate_fps_dir + '/' + target))
    fp_bundled = plf.Fingerprint.from_pickle(bundled_fp_dir + '/' + target + '_fingerprints.pkl')
    df_bundled = fp_bundled.to_dataframe()

    chembl_ids = [str.split('_')[2] for str in separate_fp_files_list]
    df_bundled['chembl_id'] = chembl_ids

    with open('configs/docking.yaml', "r") as f:
        config2 = yaml.safe_load(f)[target]

    data_path = config2['data_path'][1:]
    protein_path = config2['protein_path'][1:]

    df = pd.read_csv(data_path, sep=';')
    df_pr = df[['Molecule ChEMBL ID', 'Smiles', 'Standard Value']].dropna().groupby(['Molecule ChEMBL ID', 'Smiles'], as_index=False).mean().copy()

    second_df = df_pr[df_pr['Molecule ChEMBL ID'].isin(chembl_ids)].reset_index(drop=True)
    second_df.rename(columns={'Molecule ChEMBL ID': 'chembl_id'}, inplace=True)
    new_df = df_bundled.join(second_df)
    new_df['pIC50'] = -np.log10(new_df['Standard Value'])

    return new_df


def prepare_np_xy(df):
    columns = df.columns[:-5].tolist() + [df.columns[-1]]
    df_sel = df[columns]
    np_arr = df_sel.to_numpy().astype(np.float32)
    X = np_arr[:, :-1]
    y = np_arr[:, -1]
    return X, y


def get_kfold_split(X, y, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    splits = kf.split(X)
    return splits