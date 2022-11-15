# Introduction 
## Setup
1. Clone this repository
2. Navigate inside the cloned folder
3. Create the conda environment by running `conda env create -f env.yaml`
4. Validate the installation by activating the environment (`conda activate pbpm-experiment`)

## Tests
1. Activate the conda environment (conda activate netzsch-machine-learning)
2. Execute test suite by running pytest in the root directory 
3. To generate test coverage reports use `pytest --cov ml`
4. To get a detailed report about missing lines use `pytest --cov ml --cov-report html`

## Change environment
`conda env update --name pbpm-experiment --file env.yaml --prune`

## Commands
Evaluate architecture: 
`python -m ml.main evaluate-architecture "D:\runs_8\compStudy"  "base" "1_mixed_True_1.2" "2_mixed_True_1.4" "3_mixed_True_1.6" "3_mixed_True_2" -aggregate_on "fold" -num_precision 
3`

Evaluate gain:
`python -m ml.main evaluate-gain "D:\runs_8\compStudy"  "base" "1_mixed_True_1.2" "2_mixed_True_1.4" "3_mixed_True_1.6" "3_mixed_True_2" -d "Helpdesk" -d "BPIC12" -a "Buksh" -a "Camargo" -m "Accuracy"
`
