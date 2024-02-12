#  README 
This code is meant to run the code for breaking alpha helices in proteins using reinforcement learning. 

##  Packages Required

It is advised to install all of the below packages in a conda environment. Please install the conda environment from the ESM Fold Repository. This will install pytorch 1.x version as well. 

- [Farama Gymnasium](https://github.com/Farama-Foundation/Gymnasium) - this is to construct the RL environment
- [BioVec](https://github.com/kyu999/biovec/tree/master) - this is to embedd the states
- [Biotite](https://www.biotite-python.org/) - this is to get protein structural embeddings to obtain the reward
- [ESMFold](https://github.com/facebookresearch/esm) - this is to obtain the peptide structure from sequence
- [biopandas](https://biopandas.github.io/biopandas/tutorials/Working_with_PDB_Structures_in_DataFrames/) - this is to read the initial pdb files. 

The reinforcement learning environment is described below:

## Reinforcement Learning Environment
### States
The states are described as protein sequences that are embedded in a 100 dimensional space using a pretrained model called ProtVec[^1]. This is implemented in the file `utils/encoder_decoder.py`. The module that I use for this is biovec, implemented in this  [GitHub Repo](https://github.com/kyu999/biovec/tree/master). Please make sure to pay attention to [this issue](https://github.com/kyu999/biovec/issues/15#issuecomment-1543044407). 

### Actions 
Actions are single point mutations. Based on the kinds of mutations possible, I classify my model into two - **Proline** and **No Proline** Model. I do this since Proline is a known helix breaker and the **Proline** model has shown Mode collapse. Each of these models can be trained separately. While the **Proline** model can be trained using `Training.ipynb`, use `Training_No_Proline.ipynb` to train the **No Proline** model. Per episode, upto 15 mutations are allowed. 

### Rewards
Post mutation, we generate secondary structure of the protein using a tool called ESMFold[^4]. Download it through [this GitHub Repository](https://github.com/facebookresearch/esm). ESMFold (atleast from this implementation) requires cuda 10.2. An alternative implementation is as part of the HuggingFace  Transformers library, although that is not used in my code. 
The main determinant for the reward is the alpha helical percentage, which I define as the percentage of amino acids that are part of a alpha helix in the peptide. If the model creates peptides that have this percent < threshold, it is awarded a positive reward of +10 and the episode ends.If not, the agent is penalised with a negative reward of -0.01 with each mutations. If the number of mutations = 15 and the threshold is still not met it gets a large negative reward of -25. We use the P-SEA algorithm[^2] to get the alpha helical percentage. To use this, please download [Biotite documentation — Biotite 0.39.0 documentation (biotite-python.org)](https://www.biotite-python.org/).  


## RL algorithm
I use the REINFORCE algorithm[^3] with Entropy Regularisation. The algorithm is given as follows:
$$
\nabla_\theta = \log \nabla_\theta G(\tau) \mathbb{E}(\tau) - \sum p_i \log p_i
$$



# References

[^1]: [Continuous Distributed Representation of Biological Sequences for Deep Proteomics and Genomics | PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0141287)
[^2]: [P-SEA: a new efficient assignment of secondary structure from C alpha trace of proteins - PubMed (nih.gov)](https://pubmed.ncbi.nlm.nih.gov/9183534/) 
[^3]: [Simple statistical gradient-following algorithms for connectionist reinforcement learning (springer.com)](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf)  
[^4]: [Evolutionary-scale prediction of atomic-level protein structure with a language model | Science](https://www.science.org/doi/10.1126/science.ade2574) 