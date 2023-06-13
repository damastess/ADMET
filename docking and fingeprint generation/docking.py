import sys
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import re

import traceback
import logging


def optimize_conformation(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def dock_herg(mol, out_path, program='smina'):
    Chem.MolToMolFile(mol, f'molecule.mol')
    os.system("obabel -imol molecule.mol -omol2 -O molecule.mol2")
    os.remove(f'molecule.mol')
    os.system(program + " -r ../data/proteins/7cn1_hERG.pdb -l molecule.mol2 --center_x 141.475 --center_y 141.38225 --center_z 166.96675 --size_x 30 --size_y 30 --size_z 30 --exhaustiveness 8 --out " + out_path)

def dock_cyp3a4(mol, out_path, program='smina'):
    Chem.MolToMolFile(mol, f'molecule.mol')
    os.system("obabel -imol molecule.mol -omol2 -O molecule.mol2")
    os.remove(f'molecule.mol')
    os.system(program + " -r ../data/proteins/1w0g_CYP3A4.pdb -l molecule.mol2 --center_x 58.633 --center_y 81.856 --center_z 7.439 --size_x 30 --size_y 30 --size_z 30 --exhaustiveness 8 --out " + out_path)

def dock_cyp2d6(mol, out_path, program='smina'):
    Chem.MolToMolFile(mol, f'molecule.mol')
    os.system("obabel -imol molecule.mol -omol2 -O molecule.mol2")
    os.remove(f'molecule.mol')
    os.system(program + " -r ../data/proteins/3qm4_CYP2D6.pdb -l molecule.mol2 --center_x -5.464 --center_y -7.289 --center_z 30.982 --size_x 30 --size_y 30 --size_z 30 --exhaustiveness 8 --out " + out_path)

def dock_cyp2d6_new(mol, out_path, program='smina'):
    Chem.MolToMolFile(mol, f'molecule.mol')
    os.system("obabel -imol molecule.mol -omol2 -O molecule.mol2")
    os.remove(f'molecule.mol')
    os.system(program + " -r ../data/proteins/cyp2d6_nowy01_4wnw.pdb -l molecule.mol2 --center_x -4.664 --center_y -6.289 --center_z 33.282 --size_x 30 --size_y 30 --size_z 30 --exhaustiveness 8 --out " + out_path)


def dock_molecules(target, start_idx, mol_count):
    end_idx = start_idx + mol_count - 1

    if(target == 'herg'):
        data_path = "../data/ligands/herg_ic50.csv"
    elif(target == 'cyp3a4'):
        data_path = "../data/ligands/cyp3a4_ic50.csv"
    elif(target == 'cyp2d6'):
        data_path = "../data/ligands/cyp2d6_ic50.csv"
    elif(target == 'cyp2d6_new'):
        data_path = "../data/ligands/cyp2d6_ic50.csv"
    else:
        pass

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

            if(target == 'herg'):
                out_path = './docked/herg/' + 'herg_' + idx_string_zeros + '_' + row['Molecule ChEMBL ID'] + '.mol2'
                dock_herg(mol, out_path)

            elif(target == 'cyp3a4'):
                out_path = './docked/cyp3a4/' + 'cyp3a4_' + idx_string_zeros + '_' + row['Molecule ChEMBL ID'] + '.mol2'
                dock_cyp3a4(mol, out_path)

            elif(target == 'cyp2d6'):
                out_path = './docked/cyp2d6/' + 'cyp2d6_' + idx_string_zeros + '_' + row['Molecule ChEMBL ID'] + '.mol2'
                dock_cyp2d6(mol, out_path)

            elif(target == 'cyp2d6_new'):
                out_path = './docked/cyp2d6_new/' + 'cyp2d6_new_' + idx_string_zeros + '_' + row['Molecule ChEMBL ID'] + '.mol2'
                dock_cyp2d6_new(mol, out_path)

            else:
                pass

        except Exception as e:
            logging.error(traceback.format_exc())


if __name__=='__main__':

    target = sys.argv[1]
    start_idx = int(sys.argv[2])
    mol_count = int(sys.argv[3])

    dock_molecules(target, start_idx, mol_count)