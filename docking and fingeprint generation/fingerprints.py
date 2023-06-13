import sys
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import re

import traceback
import logging

import prolif as plf

def generate_fingerprints(target, start_idx, mol_count):

    if target == 'herg':
        target_pdb = '../data/proteins/7cn1_hERG.pdb'
    elif target == 'cyp3a4':
        target_pdb = '../data/proteins/1w0g_CYP3A4.pdb'
    elif target == 'cyp2d6':
        target_pdb = '../data/proteins/3qm4_CYP2D6.pdb'
    elif target == 'cyp2d6_new':
        target_pdb = '../data/proteins/cyp2d6_nowy01_4wnw.pdb'

    docked_dir = 'docked'

    list = sorted(os.listdir(docked_dir + '/' + target))
    lmols = []
    mol = Chem.MolFromPDBFile(target_pdb, removeHs=False)
    pmol = plf.Molecule(mol)
    err_num = 0

    for file_name in list[start_idx:mol_count]:
        try:

            with open(docked_dir + '/' + target +'/' + file_name, 'r') as file:
                mol_blocks = ['@<TRIPOS>MOLECULE'+block for block in file.read().split('@<TRIPOS>MOLECULE')][1:]
                mol = Chem.MolFromMol2Block(mol_blocks[0])
            lmol = plf.Molecule.from_rdkit(mol)

            fp_1 = plf.Fingerprint()
            fp_1.run_from_iterable([lmol], pmol)

            lmols.append(lmol)

        except Exception as e:
            logging.error(traceback.format_exc())
            err_num += 1

    fp = plf.Fingerprint()
    fp.run_from_iterable(lmols, pmol)
    fp.to_pickle("fingerprints/" + target + "_fingerprints.pkl")
    print(start_idx, mol_count, len(lmols), err_num)

if __name__=='__main__':

    target = sys.argv[1]
    start_idx = int(sys.argv[2])
    mol_count = int(sys.argv[3])

    generate_fingerprints(target, start_idx, mol_count)