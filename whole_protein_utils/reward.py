import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from datetime import datetime
import os
from whole_protein_utils.sequence import *
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np

# DEFINING THE MODEL FOR PROTEIN MODELLING
torch.backends.cuda.matmul.allow_tf32 = True
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1",local_files_only=True)
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",local_files_only=True)
model = model.cuda()
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

def percentage_of_secondary_structure(arr,secondary_structure_type,starting_residue,ending_residue):
        
    '''
    MODIFIED.
    this function is used to find the percentage secondary stuctural charecter in a peptide within a protein
    '''
    
    if len(arr) == 0:
        raise ValueError("Input array must not be empty")
    # this will count the number of amino acids that are part of a helix.
    # print(starting_residue,ending_residue)
    # print(len(arr))
    arr = arr[starting_residue:ending_residue+1] 
    # print(arr)
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
                    reward_cutoff,
                    unique_name_to_give,
                    starting_residue_id,
                    ending_residue_id,
                    secondary_structure_type_from_env ='helix',
                    validation=False,
                    folder_to_save_validation_files=None
                    ):
    '''
    MODIFIED
    '''
    if validation==False:
        generate_structure_from_sequence(protein_sequence,
                                         name=f'NEW_{unique_name_to_give}')
        path_of_the_newly_created_file = f'NEW_{unique_name_to_give}.pdb'
        result = percentage_of_secondary_structure(get_structural_annotations(path_of_the_newly_created_file),
                                                   secondary_structure_type=secondary_structure_type_from_env,
                                                  starting_residue = starting_residue_id,
                                                  ending_residue = ending_residue_id) # here the starting and ending residues are also taken into account. 
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
                                                   secondary_structure_type=secondary_structure_type_from_env,
                                                   starting_residue = starting_residue_id,
                                                   ending_residue = ending_residue_id)
        reward = get_reward_from_result(result_got=result,cutoff=reward_cutoff)
        if int(reward)<10:
            os.remove(path_of_the_newly_created_file)
        return reward
