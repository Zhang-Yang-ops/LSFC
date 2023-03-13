# Drug-Drug Interaction Prediction Based on Local Substructure Features and Their Complements


## Requirements  

numpy==1.21.0 \
tqdm==4.64.1 \
pandas==1.5.2 \
rdkit==2019.09.3 \
scikit_learn==1.2.0 \
torch==1.8.0 \
torch-cluster==1.5.9 \
torch_geometric==2.2.0 \
torch_scatter==2.0.6 \
torch-sparse==0.6.10 \
matplotlib==3.6.2 \
networkx==2.8.8 \
tensorboardx==2.5.1 \




## Step-by-step running:  
### 1. DrugBank
- First, cd LSFC/drugbank, and run data_preprocessing.py using  
  `python data_preprocessing.py -d drugbank -o all`  
  Running data_preprocessing.py convert the raw data into graph format. \
   Create a directory using \
  `mkdir save`  
- Second, run train.py using 
  `python train.py --fold 0 --save_model` 

  to train LSFC. The training record can be found in save/ folder.

  Explanation of parameters

### 2. TWOSIDES
- First, cd LSFC/drugbank, and run data_preprocessing.py using  
  `python data_preprocessing.py -d twosides -o all`   
  Running data_preprocessing.py convert the raw data into graph format.
  Create a directory using \
  `mkdir save`
- Second, run train.py using 
  `python train.py --fold 0 --save_model` 

  to train LSFC. The training record can be found in save/ folder.
