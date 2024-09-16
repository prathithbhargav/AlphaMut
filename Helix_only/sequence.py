
# imports
'''
this file is meant to store all the functions meant for mutating the giving the protein 
'''
from biopandas.pdb import PandasPdb


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

def read_pdb_file(file_path):
    '''
    takes in a pdb file and gives back the sequence in string format 
    '''
    chars_to_remove = "ZX"
    input_string = df2sequence(sequence_df(pdb2df(file_path)))
    result_string = ''.join(char for char in input_string if char not in chars_to_remove)
    return result_string


