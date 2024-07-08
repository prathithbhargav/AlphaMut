import torch
import numpy as np
from whole_protein_utils.encoder_decoder import *
from whole_protein_utils.sequence import *
from whole_protein_utils.reward import *
from gymnasium import Env
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box
import random
import os
import pandas as pd
import glob


class ProteinEvolution(Env):
    '''
    Class for the helix breaker with the entire protein
    '''
    def __init__(self,
                 file_containing_sequence_database, # this is to replace the earlier code that required that you have a folder. 
                 protein_length_limit, # this is to choose those proteins that can actually be modelled by ESM on the given cluster --- 25GB GPU can model upto 500 length protein
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
        dataframe_of_dataset = pd.read_csv(file_containing_sequence_database)
        keyword_to_choose = 'training' if validation == False else 'validate'
        dataframe_of_dataset = dataframe_of_dataset[dataframe_of_dataset['dataset'] == keyword_to_choose ]
        dataframe_of_dataset = dataframe_of_dataset[dataframe_of_dataset['length_of_whole_sequence']<= protein_length_limit]
        self.dataset_of_initial_sequences = dataframe_of_dataset 
        # selecting the initial pdb file
        self.row_of_template_sequence = self.dataset_of_initial_sequences.sample()
        # giving a unique path - this is just to ensure that when run paralelly, there are no issues. Anything can be given as a unique path
        self.unique_path_to_give_for_file = unique_path_to_give_for_file
        # what is the encoding? esm or biovec
        self.sequence_encoding_type = sequence_encoding_type
        # choosing the secondary structure to disrupt
        self.secondary_structure_to_disrupt = secondary_structure_to_disrupt
        # getting the length of the file so as to 
        self.initial_helix_sequence = self.row_of_template_sequence['Seq'].values[0]
        # print(type(self.initial_helix_sequence))
        initial_sequence_state = protein_to_indices(self.initial_helix_sequence)
        length = len(initial_sequence_state)
        self.entire_initial_protein_sequence =  self.row_of_template_sequence['whole_protein_sequence'].values[0]
        self.length = length
        # this is to initialise the folder where validation structures are saved
        self.folder_to_save_validation_files = folder_to_save_validation_files
        # reward cutoff - to determine when to stop the mutation 
        self.cutoff_for_the_reward = reward_cutoff
        # this is to initialise how many maximum mutations to perform per episode. 
        self.maximum_number_of_allowed_mutations_per_episode = maximum_number_of_allowed_mutations_per_episode
        self.path_of_template_pdb_file = self.row_of_template_sequence['PDB'].values[0]
        # if proline is being used there are only 19 possible amino acid substitutions, else it is 20. 
        if use_proline == False:
            self.no_of_amino_acid = 19
        if use_proline == True:
            self.no_of_amino_acid = 20
        
        self.action_space = Discrete(self.length*self.no_of_amino_acid)
        if self.sequence_encoding_type == 'biovec':
            # since we're gonna accomodate both the individual sequence and the protein sequence. 
            self.observation_space =Box(low=-np.inf, high=np.inf, shape=(200,), dtype=np.float32)
        if self.sequence_encoding_type == 'esm':
            self.observation_space =Box(low=-np.inf, high=np.inf, shape=(640,), dtype=np.float32)
        # to get the new state for mutator 
        self.dummy_state_for_mutator = protein_to_indices(self.initial_helix_sequence)
        # to get the initial state
        

        # @TODO needs to be modified. 
        # initialising the number of mutations
        self.number_of_mutations = 0
        # whether to use the environment for validation or not. 
        self.use_environment_for_validation = validation

        self.starting_residue_in_protein = self.row_of_template_sequence['starting_residue'].values[0]
        self.ending_residue_in_protein = self.row_of_template_sequence['ending_residue'].values[0]
        
        self.pre_helix_of_protein = self.entire_initial_protein_sequence[:self.starting_residue_in_protein - 1]
        self.post_helix_of_protein = self.entire_initial_protein_sequence[self.ending_residue_in_protein:]
        self.entire_protein_sequence = self.pre_helix_of_protein + self.initial_helix_sequence + self.post_helix_of_protein # the reason we are defining this is so that we consistently keep doing this 
        helix_embedding = convert_sequence_to_embeddings(self.initial_helix_sequence,embedding_type=self.sequence_encoding_type)
        whole_protein_embedding = convert_sequence_to_embeddings(self.entire_protein_sequence,embedding_type = self.sequence_encoding_type)
        self.state = np.concatenate([helix_embedding,whole_protein_embedding])        
        
    def step(self,action):
        
        self.number_of_mutations += 1 # adding one mutation every time an action is taken

        # getting the 
            
        amino_acid_position = action // self.no_of_amino_acid

        amino_acid_new = (action % self.no_of_amino_acid) + 1
        
        # this part is for mutating and obtaining the new state
        self.dummy_state_for_mutator[amino_acid_position] = amino_acid_new
        # finding the mutated sequence. 
        mutated_helix_sequence = indices_to_protein(self.dummy_state_for_mutator)
        # print(mutated_helix_sequence)
        # embedding the mutated sequence as a state. 
        mutated_helix_state = convert_sequence_to_embeddings(mutated_helix_sequence,embedding_type=self.sequence_encoding_type)
        mutated_whole_protein_sequence = self.pre_helix_of_protein + mutated_helix_sequence + self.post_helix_of_protein
        mutated_whole_protein_state = convert_sequence_to_embeddings(mutated_whole_protein_sequence,embedding_type=self.sequence_encoding_type)
        self.state = np.concatenate([mutated_helix_state,mutated_whole_protein_state])
        # Reward

        if self.use_environment_for_validation == False:
            reward = reward_function(template_protein_structure_path=self.path_of_template_pdb_file,
                                    protein_sequence=mutated_whole_protein_sequence,
                                    reward_cutoff=self.cutoff_for_the_reward,
                                    unique_name_to_give=self.unique_path_to_give_for_file,
                                    starting_residue_id = self.starting_residue_in_protein,
                                    ending_residue_id = self.ending_residue_in_protein,
                                    secondary_structure_type_from_env =self.secondary_structure_to_disrupt,
                                    validation=False,
                                    folder_to_save_validation_files=None)
            
        if self.use_environment_for_validation == True:
            reward = reward_function(template_protein_structure_path=self.path_of_template_pdb_file,
                                    protein_sequence=mutated_whole_protein_sequence,
                                    reward_cutoff=self.cutoff_for_the_reward,
                                    unique_name_to_give=self.unique_path_to_give_for_file,
                                    starting_residue_id = self.starting_residue_in_protein,
                                    ending_residue_id = self.ending_residue_in_protein,
                                    secondary_structure_type_from_env =self.secondary_structure_to_disrupt,
                                    validation=True,
                                    folder_to_save_validation_files=self.folder_to_save_validation_files)

        if reward != 10 and self.number_of_mutations < self.maximum_number_of_allowed_mutations_per_episode:
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
        # print('resetting_happening')


        self.row_of_template_sequence = self.dataset_of_initial_sequences.sample()
        # print(self.row_of_template_sequence.index)
        self.path_of_template_pdb_file = self.row_of_template_sequence['PDB'].values[0]
        self.entire_initial_protein_sequence =  self.row_of_template_sequence['whole_protein_sequence'].values[0]
        self.starting_residue_in_protein = self.row_of_template_sequence['starting_residue'].values[0]
        self.ending_residue_in_protein = self.row_of_template_sequence['ending_residue'].values[0]
        self.pre_helix_of_protein = self.entire_initial_protein_sequence[:self.starting_residue_in_protein - 1]
        self.post_helix_of_protein = self.entire_initial_protein_sequence[self.ending_residue_in_protein:]
        self.initial_helix_sequence = self.row_of_template_sequence['Seq'].values[0]
        self.entire_protein_sequence = self.pre_helix_of_protein + self.initial_helix_sequence + self.post_helix_of_protein
        helix_embedding = convert_sequence_to_embeddings(self.initial_helix_sequence,embedding_type=self.sequence_encoding_type)
        whole_protein_embedding = convert_sequence_to_embeddings(self.entire_protein_sequence,embedding_type = self.sequence_encoding_type)
        self.state = np.concatenate([helix_embedding,whole_protein_embedding])        
        self.dummy_state_for_mutator = protein_to_indices(self.initial_helix_sequence)
        self.number_of_mutations = 0

        info = {}
        return self.state, info