# Structure based ADMET prediction

This repository contains the code and data used in the ADMET property prediction project carried out as part of the "Machine learning in drug design" course. The aim of this project was to predict activity values for protein-ligand pairs using structure based approach. The three proteins we worked with are:

- hERG - blocking of potassium ion channels causes cardiotoxicity
- and two subtypes of cytochrome P450 which are important for the clearance of various compounds:
    - CYP3A4
    - CYP2D6

## Molecular docking and interaction fingerprints

In order to obtain interaction fingerprints which serve as input to our machine learning models we conducted molecular docking for protein-ligand pairs for which we have their activity measurements. We used SMINA which is a fork of Autodock Vina docking tool.

Molecular docking is the process of finding the optimal position of the ligand in the protein's binding pocket by optimizing the energy function.

In the next step we generated interaction fingerprint using ProLIF package. This way we obtain binary vectors describing the types of interactions between the atoms of the ligand and the amino acids of the protein.

## Code

In order to use the code install and activate the conda environmant provided in environment.yml file: `conda env create -f environment.yml`

See the notebook `example_training.ipynb` for an example data preparation of input and machine learning models training.

To obtain a pandas dataframe with full information about each ligand interacting with a chosen protein use a function from `data_preparation.py`:
```
PROTEIN = 'herg'
df = prepare_df(PROTEIN)
```
To get binary input X and y pIC50 activity labels use:
```
X, y = prepare_np_xy(df)
```
`example_docking_and_fingerprint_generation.ipynb` notebook shows the example use of provided docking and fingerprint generation scripts. `docking.py` allows for parallelization of the docking process by specifing multiple ranges of indexes for various compute nodes.