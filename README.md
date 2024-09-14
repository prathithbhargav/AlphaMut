# AlphaMut

This is the GitHub Repository accompanying the paper: "AlphaMut: a deep reinforcement learning model to
suggest helix disrupting mutations" 

#  README 

## Running the Inference Model

To run the trained model(Helix-in-protein), you can run the colab notebook - `3_inference_of_Helix-in-protein_trained_model.ipynb`. Instructions are provided in the colab notebook along with an illustrative example. 


## Training the Model

This code is meant to run the code for learning how to break helices using Reinforcement Learning. 
There are two models --- one that disrupts helices, and another that disrupts helices within a protein environment. 

Information on training is provided in the Jupyter Notebooks - `1_training_and_validation_only_helix.ipynb` and `2_training_and_validation_with_protein.ipynb`

###  Packages Required

It is advised to install all of the below packages in a conda environment(>= python 3.8). It is advised to use StableBaselines3 since it has standard ready-to-use implementations of RL Algorithms.  StableBaselines3 also downloads Gymnasium, which is necessary for the 

The following packages are required:

- [BioVec](https://github.com/kyu999/biovec/tree/master) - this is to embed the states. The states are described as protein sequences that are embedded in a 100 dimensional space using a pretrained model called ProtVec[^1]. The other way to get the state is through the ESM-2 model, that gives us a 320 dimensional space. This is implemented in the file `utils/encoder_decoder.py`. The module that I use for this is biovec, implemented in this  [GitHub Repo](https://github.com/kyu999/biovec/tree/master). Please make sure to pay attention to [this issue](https://github.com/kyu999/biovec/issues/15#issuecomment-1543044407). If you're using the esm model, there should be no issues related to installation since esm is implemented in transformers. This package is required only if you're plannning to train the Helix-only model. 
- [Biotite](https://www.biotite-python.org/) - this is to get protein structural embeddings (from P-SEA)[^2] to obtain the reward
- [Transformers](https://huggingface.co/transformers/v3.5.1/installation.html) - this is to get the ESMFold[^3] Model and the ESM embedding model. 
- [biopandas](https://biopandas.github.io/biopandas/tutorials/Working_with_PDB_Structures_in_DataFrames/) - this is to read the initial pdb files. 
- [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/) for the RL algorithms.  




# References

[^1]: [Continuous Distributed Representation of Biological Sequences for Deep Proteomics and Genomics | PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0141287)
[^2]: [P-SEA: a new efficient assignment of secondary structure from C alpha trace of proteins - PubMed (nih.gov)](https://pubmed.ncbi.nlm.nih.gov/9183534/) 
[^3]: [Evolutionary-scale prediction of atomic-level protein structure with a language model | Science](https://www.science.org/doi/10.1126/science.ade2574) 
