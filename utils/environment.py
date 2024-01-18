
import numpy as np
import biovec
import glob
# from utils functions
from utils.encoder_decoder import *
from utils.sequence import *
from utils.reward import *
# for deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# for envronment creation
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import MultiDiscrete
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box

#for reading PDB files and processing them
from biopandas.pdb import PandasPdb
import pandas as pd
from utils.sequence import *

# for generating structures through esm instead of modeller
import esm
import biotite.structure as struc
import biotite.structure.io as strucio

# for general utility
import random
import os
import subprocess
import time
import matplotlib.pyplot as plt
from datetime import datetime



class PeptideEvolution(Env):
    '''
    This is to modify the class to make it compatible with the biovec embedding. 
    '''
    def __init__(self,
                 folder_containing_pdb_files,
                 structure_generator,
                 folder_to_save_validation_files,
                 reward_cutoff,
                 unique_path_to_give_for_file,
                 maximum_number_of_allowed_mutations_per_episode=15,
                 validation=False):
        # selecting the initial pdb file
        list_of_initial_pdb_files = os.listdir(folder_containing_pdb_files)
        self.path_of_template_pdb_file = folder_containing_pdb_files+ '/' +random.choice(list_of_initial_pdb_files)
        self.unique_path_to_give_for_file = unique_path_to_give_for_file
        initial_pdb_path = self.path_of_template_pdb_file
        initial_pdb_structure_state = protein_to_indices(read_pdb_file(initial_pdb_path))
        length = len(initial_pdb_structure_state)
        self.folder_to_save_validation_files = folder_to_save_validation_files
        self.cutoff_for_the_reward = reward_cutoff
        self.maximum_number_of_allowed_mutations_per_episode = maximum_number_of_allowed_mutations_per_episode
        # global - "ish" variables
        self.tool_to_generate_structures = structure_generator
        self.folder_of_initial_pdb_structures = folder_containing_pdb_files
        self.length = length
        self.no_of_amino_acid = 20
        self.folder_for_training = folder_containing_pdb_files
        self.action_space = Discrete(self.length*self.no_of_amino_acid)
        self.observation_space =Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32)
        self.dummy_state_for_mutator = protein_to_indices(read_pdb_file(initial_pdb_path))
        self.state = convert_sequence_to_embeddings(read_pdb_file(initial_pdb_path))
        self.number_of_mutations = 0
        self.use_environment_for_validation = validation

    def step(self,action):
        
        self.number_of_mutations += 1

        amino_acid_position = action // self.no_of_amino_acid

        amino_acid_new = (action % self.no_of_amino_acid) + 1
        
        # this part is for mutating and obtaining the new state
        self.dummy_state_for_mutator[amino_acid_position] = amino_acid_new

        protein_sequence = indices_to_protein(self.dummy_state_for_mutator) # here you have gotten new protein state
        # print(amino_acid_position,protein_sequence[amino_acid_position],' is the new mutation')
        # New State
        self.state = convert_sequence_to_embeddings(protein_sequence)
        # Reward
        if self.use_environment_for_validation == False:
            reward = reward_function(self.path_of_template_pdb_file,
                                 protein_sequence,
                                 self.folder_for_training,
                                 self.tool_to_generate_structures,
                                 mode_of_rmsd='structural',
                                 unique_name_to_give = self.unique_path_to_give_for_file,
                                 reward_cutoff= self.cutoff_for_the_reward,
                                 to_modify_rmsd=True)
        if self.use_environment_for_validation == True:
            reward = reward_function_for_validation(self.path_of_template_pdb_file,
                        protein_sequence,
                        self.folder_for_training,
                        self.tool_to_generate_structures,
                        mode_of_rmsd='structural',
                        reward_cutoff= self.cutoff_for_the_reward,
                        folder_to_save_validation_files= self.folder_to_save_validation_files,
                        to_modify_rmsd=True)

        
        if reward != 10 and self.number_of_mutations <self.maximum_number_of_allowed_mutations_per_episode:
            actual_reward = reward
            terminated = False
        if reward == 10 and self.number_of_mutations>1:
            terminated= True
            actual_reward=reward
        if reward == 10 and self.number_of_mutations==1:
            terminated= False
            actual_reward=reward
    
        if self.number_of_mutations>=self.maximum_number_of_allowed_mutations_per_episode and reward!=10:
            terminated=True
            actual_reward = -25
            file_base_name_without_extension = os.path.basename(self.path_of_template_pdb_file).split('.')[0]

        info = [amino_acid_position,return_amino_acid_of_index(amino_acid_new)]


        return self.state, actual_reward, terminated, False, info
    
    def render(self):
        None

    def reset(self):
        try:
            os.remove(f'NEW_{self.unique_path_to_give_for_file}.pdb')
        except:
            None
        list_of_initial_pdb_files = os.listdir(self.folder_of_initial_pdb_structures)
        self.path_of_template_pdb_file =self.folder_of_initial_pdb_structures+'/'+random.choice(list_of_initial_pdb_files)
        initial_pdb_path = self.path_of_template_pdb_file
        # initial_pdb_path = self.folder_of_initial_pdb_structures+'/'+random.choice(list_of_initial_pdb_files)
        initial_pdb_structure_state = protein_to_indices(read_pdb_file(initial_pdb_path))
        self.dummy_state_for_mutator = protein_to_indices(read_pdb_file(initial_pdb_path))
        self.state = convert_sequence_to_embeddings(read_pdb_file(initial_pdb_path))
        self.number_of_mutations = 0
        info = {}
        return self.state, info, self.dummy_state_for_mutator