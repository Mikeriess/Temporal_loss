# P1_temporal_loss
 Framework used for training remaining time models in paper 1 of the dissertation

- This framework cover:
	- Data preparation for training data
	- Data preparation for inference (production/online) data
	- Model training (seqential, static)
	- Model evaluation


# Installation/requirements

1. Install all required packages from conda via (PPM-RT-MODEL-env.txt)
2. Install TF via pip install tensorflow=2.7.1


# Options

- Model types
	1. Prefix-log:
		- LSTM (DA-LSTM by Navarini et al., 2018)


# Steps

- General usage
	1. Create results/ directory for storing models, inference tables and training history
	2. Modify generate_experiments.py to generate a design table
	3. Run run_experiments.py to train models given the desired settings
	4. Analyze the results using notebooks in /analysis directory

- Approach used in the paper
	1. Clone the repo to two separate directories one for gridsearch, and one for main experiments
	2. Generate design table in gridsearch repo via generate_experiments.py
	3. Run gridsearch via run_experiments.py
	4. Create conditional hyper-parameter code generated in analysis/GS_results.ipynb based on GS results
	5. Copy generated code into generate_experiments.py in main experiments directory
	6. Run run_experiments.py to train models
	3. Analyze the results using notebooks in /analysis directory