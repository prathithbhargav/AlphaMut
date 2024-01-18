import numpy as np
import biovec
def protein_to_indices(protein_sequence):
    # Define the mapping of amino acids to indices

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    # this has been verified manually to correspond to the single letter amino acids
    
    amino_acid_indices = {aa: idx for idx, aa in enumerate(amino_acids, 1)}

    # Convert the protein sequence to a list of indices
    indices_list = [amino_acid_indices.get(aa, 0) for aa in protein_sequence]
    indices_list = np.array(indices_list)

    return indices_list

def indices_to_protein(indices_list):
    
    # Define the mapping of indices to amino acids
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    # Create a dictionary that maps indices to amino acids
    index_to_amino_acid = {idx: aa for idx, aa in enumerate(amino_acids, 1)}

    # Convert the list of indices to a protein sequence
    protein_sequence = ''.join([index_to_amino_acid[idx] for idx in indices_list if idx in index_to_amino_acid])
    # protein_sequence = ''.join([index_to_amino_acid.get(idx, 'X') for idx in indices_list])

    return protein_sequence

def convert_sequence_to_embeddings(sequence):
    '''
    takes in sequence as a string as an input and outputs a column matrix with 100 values
    it uses the biovec model. 
    '''
    trained_model_biovec = biovec.models.load_protvec(model_fname='swissprot-reviewed-protvec.model')
    predicted_biovec_embeddings= trained_model_biovec.to_vecs(sequence)
    return np.mean(predicted_biovec_embeddings,axis=0)
    
def return_amino_acid_of_index(amino_acid_index):
    # Define the mapping of indices to amino acids
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    # Create a dictionary that maps indices to amino acids
    index_to_amino_acid = {idx: aa for idx, aa in enumerate(amino_acids, 1)}
    return index_to_amino_acid[amino_acid_index]