import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from datetime import datetime
import os
from Helix_only.sequence import *
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np

# DEFINING THE MODEL FOR PROTEIN MODELLING
torch.backends.cuda.matmul.allow_tf32 = True
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


# In[4]:


def generate_structure_from_sequence(sequence,name=None):
    '''
    This function takes in the sequence of a protein and gives back the structure - this is using the ESM Model
    '''
    
    tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids']
    tokenized_input = tokenized_input.cuda()

    import torch

    with torch.no_grad():
        output = model(tokenized_input)


    pdb = convert_outputs_to_pdb(output)
    with open(f"{name}.pdb", "w") as f:
        f.write("".join(pdb))
    

def get_structural_annotations(file_path):
    array = strucio.load_structure(file_path)
    sse = struc.annotate_sse(array, chain_id="A")
    return sse

def percentage_of_secondary_structure(arr,secondary_structure_type):
        
    '''
    this function is used to find the percentage secondary stuctural charecter in a peptide.
    '''
    
    if len(arr) == 0:
        raise ValueError("Input array must not be empty")
    # this will count the number of amino acids that are part of a helix.
    count_a = np.count_nonzero(arr == 'a')
    total_elements = len(arr)
    percentage_a = (count_a / total_elements) * 100

    # this will compute the number of amino acids that are part of a sheet. 
    count_b = np.count_nonzero(arr == 'b')
    total_elements = len(arr)
    percentage_b = (count_b / total_elements) * 100

    if secondary_structure_type == 'helix':
        return percentage_a

    if secondary_structure_type == 'sheet':
        return percentage_b

    if secondary_structure_type == 'both':
        return percentage_a, percentage_b


# In[6]:


def give_time_as_string():
    current_time = datetime.now()
    # Format the current time as a string without spaces
    time_str = current_time.strftime("%Y-%m-%d_%H:%M:%S")
    return time_str
    
def get_reward_from_result(result_got, cutoff):
    '''
    The reason I have made this separate is cause I can make it modular. It takes in a reward cutoff and gives back the reward, 
    after looking at the result, which is the percentage content of the secondary structure. 
    '''
    if result_got < cutoff:
        return 10
    else:
        return -0.01
    
def reward_function(template_protein_structure_path,
                    protein_sequence,
                    folder_containing_pdb_files,
                    reward_cutoff,
                    unique_name_to_give,
                    secondary_structure_type_from_env ='helix',
                    validation=False,
                    folder_to_save_validation_files=None
                    ):
    if validation==False:
        generate_structure_from_sequence(protein_sequence,
                                         name=f'NEW_{unique_name_to_give}')
        path_of_the_newly_created_file = f'NEW_{unique_name_to_give}.pdb'
        result = percentage_of_secondary_structure(get_structural_annotations(path_of_the_newly_created_file),
                                                   secondary_structure_type=secondary_structure_type_from_env)
        reward = get_reward_from_result(result_got=result,cutoff=reward_cutoff)
        os.remove(path_of_the_newly_created_file)
        return reward

        
    if validation == True:
        
        template_file_base_name_without_extension = os.path.basename(template_protein_structure_path).split('.')[0]
        base_path_to_give_for_file = f'{folder_to_save_validation_files}/{template_file_base_name_without_extension}_{give_time_as_string()}'
        generate_structure_from_sequence(protein_sequence,
                                         name=base_path_to_give_for_file)
        path_of_the_newly_created_file = f'{base_path_to_give_for_file}.pdb'
        result = percentage_of_secondary_structure(get_structural_annotations(path_of_the_newly_created_file),
                                                   secondary_structure_type=secondary_structure_type_from_env)
        reward = get_reward_from_result(result_got=result,cutoff=reward_cutoff)
        if int(reward)<10:
            os.remove(path_of_the_newly_created_file)
        return reward




################################old################
# def modification_of_rmsd(input_value, threshold = 4):
#     if input_value > threshold:
#         return input_value
#     else:
#         return -1

# def get_structural_annotations(file_path):
#     array = strucio.load_structure(file_path)
#     sse = struc.annotate_sse(array, chain_id="A")
#     return sse

# def get_frequency_of_secondary_structures(list_of_possible_structure_types,array_containing_structure_allocations):
#     frequency_array = np.zeros(len(list_of_possible_structure_types))
#     for index in range(len(list_of_possible_structure_types)):
#         frequency_array[index] = len(array_containing_structure_allocations[array_containing_structure_allocations == list_of_possible_structure_types[index]])
#     return frequency_array

# def get_structural_rmsd(reference_secondary_structure_frequency,target_secondary_structure_frequency):
#     diff_array = target_secondary_structure_frequency - reference_secondary_structure_frequency
#     squared_array = diff_array ** 2
#     return squared_array

# def percentage_of_a(arr):
#     '''
#     this function is used to find the percentage alpha helical charecter in a peptide.
#     '''
#     if len(arr) == 0:
#         raise ValueError("Input array must not be empty")

#     count_a = np.count_nonzero(arr == 'a')
#     total_elements = len(arr)
#     percentage_a = (count_a / total_elements) * 100

#     return percentage_a

# def percentage_of_b(arr):
#     '''
#     this function is used to find the percentage beta sheet charecter in a peptide.
#     '''
#     if len(arr) == 0:
#         raise ValueError("Input array must not be empty")

#     count_b = np.count_nonzero(arr == 'b')
#     total_elements = len(arr)
#     percentage_b = (count_b / total_elements) * 100

#     return percentage_b

# def generate_structure_from_sequence(sequence,counter,name=None):
#     '''
#     This function takes in the sequence of a protein and gives back the structure - this is using the ESM Model
#     '''
    
#     with torch.no_grad():
#         output = model.infer_pdb(sequence)
        
#     if name == None:
#         name_of_the_structure_file =  str(counter) + '.pdb'
        
#     else:
#         name_of_the_structure_file = name +'.pdb'

#     with open(name_of_the_structure_file, "w") as f:
#         f.write(output)

# def find_rmsd(path_of_initial_file,path_of_final_file,
#               mode = None):
    
#     # to read the two pdb files
#     if mode == None:
#         pdb_reader_initial_structure = PandasPdb()
#         pdb_reader_initial_structure.read_pdb(path_of_initial_file)
#         df_of_initial_structure = pdb_reader_initial_structure.df['ATOM']
#         pdb_reader_final_structure = PandasPdb()
#         pdb_reader_final_structure.read_pdb(path_of_final_file)
#         df_of_final_structure = pdb_reader_final_structure.df['ATOM']
        
#         # to get the initial and final arrays
#         initial_numpy_array = np.array(df_of_initial_structure[df_of_initial_structure['atom_name']== 'CA'][['x_coord','y_coord','z_coord']])
#         final_numpy_array = np.array(df_of_final_structure[df_of_final_structure['atom_name']== 'CA'][['x_coord','y_coord','z_coord']])
        
#         # to perform Single Va0lue Decomposition
#         sup = SVDSuperimposer()
#         sup.set(initial_numpy_array,final_numpy_array)
#         sup.run()
#         # initial_rmsd = sup.get_init_rms()
#         rmsd = sup.get_rms()
        
#         return rmsd
#     if mode == 'structural':
#         list_of_possible_structure_types = ['a','b','c']
#         template_frequency = get_frequency_of_secondary_structures(list_of_possible_structure_types,
#                                                                    get_structural_annotations(path_of_initial_file))
#         final_frequency = get_frequency_of_secondary_structures(list_of_possible_structure_types,
#                                                                 get_structural_annotations(path_of_final_file))
#         structural_rmsd = get_structural_rmsd(template_frequency, final_frequency)

#         return structural_rmsd

# def reward_function(template_protein_structure_path, 
#                     protein_sequence,
#                     folder_containing_pdb_files,
#                     tool_to_generate_structures,
#                     mode_of_rmsd,
#                     reward_cutoff,
#                     unique_name_to_give,
#                     to_modify_rmsd = False,
#                     ):
#     if tool_to_generate_structures == 'esm_sse':
#         generate_structure_from_sequence(protein_sequence,
#                                          counter = 1,
#                                          name=f'NEW_{unique_name_to_give}')
#         path_of_the_newly_created_file = f'NEW_{unique_name_to_give}.pdb'
#         result = percentage_of_a(get_structural_annotations(path_of_the_newly_created_file))
#         os.remove(path_of_the_newly_created_file)
#         if result < reward_cutoff:
#             reward = 10
#         else:
#             reward = -0.01
            
#         return reward

# def give_time_as_string():
#     current_time = datetime.now()
#     # Format the current time as a string
#     time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
#     return time_str        
        
        
# def reward_function_for_validation(template_protein_structure_path, 
#                     protein_sequence,
#                     folder_containing_pdb_files,
#                     tool_to_generate_structures,
#                     mode_of_rmsd,
#                     reward_cutoff,
#                     folder_to_save_validation_files,
#                     to_modify_rmsd = False,
#                     ):
#     file_base_name_without_extension = os.path.basename(template_protein_structure_path).split('.')[0]
#     path_to_give_for_file = f'{folder_to_save_validation_files}/{file_base_name_without_extension}_{give_time_as_string()}'
    
#     if tool_to_generate_structures == 'esm_sse':
#         generate_structure_from_sequence(protein_sequence,
#                                          counter = 1,
#                                          name=path_to_give_for_file)
        
#         result = percentage_of_a(get_structural_annotations(path_to_give_for_file+'.pdb'))

#         if result < reward_cutoff:
#             reward = 10
#         else:
#             reward = -0.01

#         if int(reward) != 10:
#             os.remove(path_to_give_for_file+'.pdb')
#         else:
#             None

#         return reward