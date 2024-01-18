
# imports
'''
this file is meant to store all the functions meant for mutating the giving the protein 
'''
from biopandas.pdb import PandasPdb
import numpy as np
import random
import pandas as pd







def pdb2df(file_path):
    '''
    this function takes in a pdb file's file path and returns a dataframe for the rows corresponding to "ATOM"

    Parameters
    ----------
    file_path : str
        file path of the input file in PDB format

    Returns
    -------
    A Pandas DataFrame

    '''
    # defining the amino acids
    # In the list below there are only 20 natural amino acids - selenocysteine and pyrroleucine are excluded.

    three_to_one = {
        'ALA': 'A',
        'ARG': 'R',
        'ASN': 'N',
        'ASP': 'D',
        'CYS': 'C',
        'GLN': 'Q',
        'GLU': 'E',
        'GLY': 'G',
        'HIS': 'H',
        'ILE': 'I',
        'LEU': 'L',
        'LYS': 'K',
        'MET': 'M',
        'PHE': 'F',
        'PRO': 'P',
        'SER': 'S',
        'THR': 'T',
        'TRP': 'W',
        'TYR': 'Y',
        'VAL': 'V',
        'CEN': 'Z',
        'NEN': 'X'
    }
    protein = PandasPdb()
    protein.read_pdb(file_path)
    protein_df = protein.df['ATOM']
    protein_df['residue_id'] = protein_df['residue_name'].map(three_to_one)
    return protein_df




def pdb2fasta(PDBFile):
    '''
    this function takes in a PDB file and outputs a FASTA format file

    Parameters
    ----------
    PDBDFile : str
        file path of the input file in PDB format.
    '''
    with open(PDBFile, 'r') as pdb_file:
        for record in SeqIO.parse(pdb_file, 'pdb-atom'):
            print('>' + record.id)
            print(record.seq)





def sequence_df(protein_df):
    '''
    takes in the protein DataFrame and outputs a DataFrame containing only residue_number and residue_id

    Parameters
    ----------
    protein_df : Pandas DataFrame Object
    '''
    protein_sequence_df = protein_df[['residue_number','residue_id']].drop_duplicates()
    protein_sequence_df = protein_sequence_df.reset_index()
    protein_sequence_df = protein_sequence_df.drop(columns = 'index')
    return protein_sequence_df




def df2sequence(pdb_df):
    '''
    takes in a sequence DataFrame and outputs a string containing the sequence in string format

    Parameters
    ----------
    pdb_df : Pandas DataFrame Object
    '''
    sequence = ''.join(pdb_df['residue_id'])
    return sequence




def biased_coin_toss(p_heads):
    '''
    returns a boolean value based on a input probability of heads.

    Parameters
    ----------
    p_heads : float
        input the desired probability of heads
    '''
    random_number = random.random()  # Generate a random number between 0 and 1

    if random_number < p_heads:
        return True  # Heads
    else:
        return False  # Tails






def read_pdb_file(file_path):
    '''
    takes in a pdb file and gives back the sequence in string format 
    '''
    chars_to_remove = "ZX"
    input_string = df2sequence(sequence_df(pdb2df(file_path)))
    result_string = ''.join(char for char in input_string if char not in chars_to_remove)
    return result_string





