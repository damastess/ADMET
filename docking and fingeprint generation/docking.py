import sys
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import re
import yaml

import traceback
import logging


def optimize_conformation(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def dock(mol, out_path, protein_path, centers, program='smina'):
    Chem.MolToMolFile(mol, f'molecule.mol')
    os.system("obabel -imol molecule.mol -omol2 -O molecule.mol2")
    os.remove(f'molecule.mol')
    os.system(program + " -r " + protein_path + " -l molecule.mol2 --center_x " + centers['x'] + " --center_y " + centers['y'] + " --center_z " + centers['z'] + " --size_x 30 --size_y 30 --size_z 30 --exhaustiveness 8 --out " + out_path)
    os.remove(f'molecule.mol')
    

def dock_molecules(target, start_idx, mol_count):
    end_idx = start_idx + mol_count - 1

    with open('../configs/docking.yaml', "r") as f:
        config = yaml.safe_load(f)[target]

    data_path = config['data_path']
    protein_path = config['protein_path']
    centers = config['centers']

    df = pd.read_csv(data_path, sep=';')
    df_pr = df[['Molecule ChEMBL ID', 'Smiles', 'Standard Value']].dropna().groupby(['Molecule ChEMBL ID', 'Smiles'], as_index=False).mean().copy()

    print('df_pr shape: ', df_pr.shape)
        
    for i, row in df_pr.loc[start_idx:end_idx].iterrows():

        try:
            idx_string = str(row.name)
            print("idx: ", idx_string)
            smiles = row['Smiles']
            mol = Chem.MolFromSmiles(smiles)
            optimize_conformation(mol)
            idx_string_zeros = idx_string.zfill(5)
            out_path = './docked/' + target + '/' + target + '_' + idx_string_zeros + '_' + row['Molecule ChEMBL ID'] + '.mol2'
            dock(mol, out_path, protein_path, centers)

        except Exception as e:
            logging.error(traceback.format_exc())


if __name__=='__main__':

    target = sys.argv[1]
    start_idx = int(sys.argv[2])
    mol_count = int(sys.argv[3])

    dock_molecules(target, start_idx, mol_count)