import numpy as np
# import biovec
from transformers import AutoTokenizer,EsmModel
import torch

esm_seq_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
encoder_model_esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")


def protein_to_indices(protein_sequence):
    '''
    this is a dummy function needed in the reinforcement learning model. converts each amino acid into a number between 1 and 20.
    '''
    # Define the mapping of amino acids to indices

    amino_acids = "ACDEFGHIKLMNQRSTVWYP"
    # this has been verified manually to correspond to the single letter amino acids
    
    amino_acid_indices = {aa: idx for idx, aa in enumerate(amino_acids, 1)}

    # Convert the protein sequence to a list of indices
    indices_list = [amino_acid_indices.get(aa, 0) for aa in protein_sequence]
    indices_list = np.array(indices_list)

    return indices_list

def indices_to_protein(indices_list):
    '''
    this is a dummy function needed in the reinforcement learning model. converts each amino acid into a number between 1 and 20.
    '''
    
    # Define the mapping of indices to amino acids
    amino_acids = "ACDEFGHIKLMNQRSTVWYP"
    
    # Create a dictionary that maps indices to amino acids
    index_to_amino_acid = {idx: aa for idx, aa in enumerate(amino_acids, 1)}

    # Convert the list of indices to a protein sequence
    protein_sequence = ''.join([index_to_amino_acid[idx] for idx in indices_list if idx in index_to_amino_acid])
    # protein_sequence = ''.join([index_to_amino_acid.get(idx, 'X') for idx in indices_list])

    return protein_sequence

def convert_sequence_to_embeddings(sequence,embedding_type):
    '''
    takes in sequence as a string as an input and outputs a column matrix.  
        
    Parameters
    ----------
    sequence : str
    embedding_type : str
    
    Returns
    -------
    numpy array or torch tensor
    '''
    # if embedding_type == 'biovec':
    #     trained_model_biovec = biovec.models.load_protvec(model_fname='swissprot-reviewed-protvec.model')
    #     predicted_biovec_embeddings= trained_model_biovec.to_vecs(sequence)
    #     return np.mean(predicted_biovec_embeddings,axis=0)
    
    if embedding_type == 'esm':
        inputs = esm_seq_tokenizer(sequence, return_tensors="pt")
        outputs = encoder_model_esm(**inputs)
        last_hidden_states = outputs.last_hidden_state 
        vector_representation = torch.mean(last_hidden_states, dim=1).flatten()
        return vector_representation.detach().numpy()
        
def return_amino_acid_of_index(amino_acid_index):
    '''
    dummy function required during validation
    '''
    # Define the mapping of indices to amino acids
    amino_acids = "ACDEFGHIKLMNQRSTVWYP"
    
    # Create a dictionary that maps indices to amino acids
    index_to_amino_acid = {idx: aa for idx, aa in enumerate(amino_acids, 1)}
    return index_to_amino_acid[amino_acid_index]
