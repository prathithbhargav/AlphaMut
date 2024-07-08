import torch
import numpy as np
from only_helix_utils.encoder_decoder import *
from only_helix_utils.sequence import *
from only_helix_utils.reward import *
from gymnasium import Env
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box
import random
import os
import glob


class PeptideEvolution(Env):
    '''
    This is to modify the class to make it compatible with the biovec embedding. 
    '''
    def __init__(self,
                 folder_containing_pdb_files,
                 folder_to_save_validation_files,
                 reward_cutoff,
                 unique_path_to_give_for_file,
                 sequence_encoding_type,
                 secondary_structure_to_disrupt='helix',
                 maximum_number_of_allowed_mutations_per_episode=15,
                 validation=False,
                use_proline=True):
        '''
        This part initialises the environment - give the arguements such as what is the corpus of structures that you'd like to train on,
        reward cutoff, maximum number of mutations allowed per episode, unique path to give for training file, etc. 
        '''
        # this selects the folder in which we have the PDBs that we can use to Train. 
        self.folder_of_initial_pdb_structures = folder_containing_pdb_files   
        # selecting the initial pdb file
        list_of_initial_pdb_files = glob.glob(f'{self.folder_of_initial_pdb_structures}/*.pdb')
        # choosing the initial template pdb file on which mutations will be performed
        self.path_of_template_pdb_file = random.choice(list_of_initial_pdb_files)
        # giving a unique path - this is just to ensure that when run paralelly, there are no issues. Anything can be given as a unique path
        self.unique_path_to_give_for_file = unique_path_to_give_for_file
        # what is the encoding? esm or biovec
        self.sequence_encoding_type = sequence_encoding_type
        # choosing the secondary structure to disrupt
        self.secondary_structure_to_disrupt = secondary_structure_to_disrupt
        # getting the length of the file so as to 
        initial_pdb_path = self.path_of_template_pdb_file
        initial_pdb_structure_state = protein_to_indices(read_pdb_file(initial_pdb_path))
        length = len(initial_pdb_structure_state)
        self.length = length
        # this is to initialise the folder where validation structures are saved
        self.folder_to_save_validation_files = folder_to_save_validation_files
        # reward cutoff - to determine when to stop the mutation 
        self.cutoff_for_the_reward = reward_cutoff
        # this is to initialise how many maximum mutations to perform per episode. 
        self.maximum_number_of_allowed_mutations_per_episode = maximum_number_of_allowed_mutations_per_episode

    
                
        # if proline is being used there are only 19 possible amino acid substitutions, else it is 20. 
        if use_proline == False:
            self.no_of_amino_acid = 19
        if use_proline == True:
            self.no_of_amino_acid = 20
        
        self.action_space = Discrete(self.length*self.no_of_amino_acid)
        if self.sequence_encoding_type == 'biovec':
            self.observation_space =Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32)
        if self.sequence_encoding_type == 'esm':
            self.observation_space =Box(low=-np.inf, high=np.inf, shape=(320,), dtype=np.float32)
        # to get the new state for mutator 
        self.dummy_state_for_mutator = protein_to_indices(read_pdb_file(initial_pdb_path))
        # to get the initial state
        self.state = convert_sequence_to_embeddings(read_pdb_file(initial_pdb_path),embedding_type=self.sequence_encoding_type)
        # initialising the number of mutations
        self.number_of_mutations = 0
        # whether to use the environment for validation or not. 
        self.use_environment_for_validation = validation

    def step(self,action):
        
        self.number_of_mutations += 1 # adding one mutation every time an action is taken

        # getting the 

        amino_acid_position = action // self.no_of_amino_acid

        amino_acid_new = (action % self.no_of_amino_acid) + 1
        
        # this part is for mutating and obtaining the new state
        self.dummy_state_for_mutator[amino_acid_position] = amino_acid_new
        # finding the mutated sequence. 
        protein_sequence = indices_to_protein(self.dummy_state_for_mutator)
        # embedding the mutated sequence as a state. 
        self.state = convert_sequence_to_embeddings(protein_sequence,embedding_type=self.sequence_encoding_type)
        # Reward

        if self.use_environment_for_validation == False:
            reward = reward_function(template_protein_structure_path=self.path_of_template_pdb_file,
                                    protein_sequence=protein_sequence,
                                    folder_containing_pdb_files=self.folder_of_initial_pdb_structures,
                                    reward_cutoff=self.cutoff_for_the_reward,
                                    unique_name_to_give=self.unique_path_to_give_for_file,
                                    secondary_structure_type_from_env =self.secondary_structure_to_disrupt,
                                    validation=False,
                                    folder_to_save_validation_files=None)
            
        if self.use_environment_for_validation == True:
            reward = reward_function(template_protein_structure_path=self.path_of_template_pdb_file,
                                    protein_sequence=protein_sequence,
                                    folder_containing_pdb_files=self.folder_of_initial_pdb_structures,
                                    reward_cutoff=self.cutoff_for_the_reward,
                                    unique_name_to_give=self.unique_path_to_give_for_file,
                                    secondary_structure_type_from_env =self.secondary_structure_to_disrupt,
                                    validation=True,
                                    folder_to_save_validation_files=self.folder_to_save_validation_files)

        
        if reward != 10 and self.number_of_mutations <self.maximum_number_of_allowed_mutations_per_episode:
            actual_reward = reward
            terminated = False
        if reward == 10:
            terminated= True
            actual_reward=reward

        if self.number_of_mutations>=self.maximum_number_of_allowed_mutations_per_episode and reward!=10:
            terminated=True
            actual_reward = -25

        # this is intended to give information on what actually the amino acid position and the new substitution is.
        info = {'amino_acid_position':amino_acid_position+1,
                'new_amino_acid':return_amino_acid_of_index(amino_acid_new)}

        return self.state, actual_reward, terminated, False, info
    
    def render(self):
        None

    def reset(self,seed=None, options=None):
        try:
            os.remove(f'NEW_{self.unique_path_to_give_for_file}.pdb')
        except:
            None
            
        list_of_initial_pdb_files = glob.glob(f'{self.folder_of_initial_pdb_structures}/*.pdb')
        self.path_of_template_pdb_file =random.choice(list_of_initial_pdb_files)
        initial_pdb_path = self.path_of_template_pdb_file
        self.state = convert_sequence_to_embeddings(read_pdb_file(initial_pdb_path),embedding_type=self.sequence_encoding_type)
        self.number_of_mutations = 0
        self.dummy_state_for_mutator = protein_to_indices(read_pdb_file(initial_pdb_path))
        info = {}
        return self.state, info

