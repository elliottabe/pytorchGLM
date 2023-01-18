# Code base for "Joint coding of visual input and eye/head position in V1 of freely moving mice"


## Setup for installing conda environment and dependencies
To install the repo there is a conda environment that will install the necessary packages. Make sure you are in the pytorchGLM Github directory.  
Use command:  
`conda env create -f environment.yaml`

After installing activate the conda environment:  
`conda activate pytorchGLM`

Once in the environment go to this site to install the appropriate pytorch version:  
https://pytorch.org/get-started/locally/

After pytorch is correctly installed run this command to install pip reqruirements:  
`pip install -r requirements.txt`

To install pytorchGLM, in the repo folder use:  
`pip install -e .`

## Assumed data file structure 
The base part of this code assumes a specific file structure convention to load data.
- Base_Folder
  - Date
    - Animal_Name
        Experiment_Condition

For example in the Niell Lab convention: 
- FreelyMovingEphys
    - 070921
        - J553RT
            - fm1
            - hf1_wn

This code will then create directories following this convension. 

## Parameters
To view and edit parameters see pytorchGLM/parameters.py file 