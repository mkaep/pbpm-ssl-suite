# Introduction 
## Setup
1. Clone this repository
2. Navigate inside the cloned folder
3. Create the conda environment by running `conda env create -f env.yaml`
4. Validate the installation by activating the environment (`conda activate pbpm-experiment`)


## Change environment
`conda env update --name pbpm-experiment --file env.yaml --prune`


## Results
All results of our experiments can be found in plotted version in the folder evaluation. Since the experiment requires a lot of data and space, the used XES logs for training and testing can be accessed by the following link:
https://my.hidrive.com/share/5uhqb6r5d5 

This link contains also the outputs of the models and the time measurements that were recorded during the training process.
The used hyperparameters are recorded in ml.augmentation_experiment_mixed.py

## Approaches
Since the code of approaches is mainly not our own, we can not provide them in this repository. The original code of the approaches can be 
found in the respective repositories of the authors:
* Khan: https://github.com/thaihungle/MAED/tree/deep-process
* Camargo: https://github.com/AdaptiveBProcess/GenerativeLSTM
* Tax: https://github.com/verenich/ProcessSequencePrediction
* Mauro: https://github.com/nicoladimauro/nnpm/tree/e252ded0e1f35c94aad2f3a6f7bc786304b9fcda
* Buksh: https://github.com/Zaharah/processtransformer
* Pasquadibisceglie: https://github.com/vinspdb/ImagePPMiner
* Theis: https://github.com/Julian-Theis/PyDREAM

To run this approaches within the proposed pipeline it is necessary to modify this approaches to work with XES format and output a file in the following format.
If you require help for adapation or insights in die modification please contact us (martin.kaeppel@uni-bayreuth.de).

## Structure of this project
* The module analysis provides basic functionality to extract descriptive statistics from the event logs.
* The module ml.augmentation contains the core of our approach. The transformations can be found in easy_augmentors.py. The supported strategies how the transformations are applied are implemented in the file augmentation_strategy.py.
* The module ml.preprocess contains preprocessing steps that should be applied for all event logs. Currently we have implemented two steps: a step that removes empty traces from an event log and a step that allows to remove traces that do not have the required attributes. Nevertheless, this steps are not used in our augmentation pipeline experiment.
* The ml.visualize module contains all plotting functionality to visualize the results of the experiments
* The ml.prepare module contains different splitting methods to split an event log into training and test data. Currently we support the following split procedures:
  * TimeSplitter that splits an event log based on the temporal order of the traces (either based on the first timestamp or last timestamp of a trace)
  * RandomSplitter splits an event log randomly into training and test data
  * KFoldSpitter splits an event log into k folds. Note, that this functionality would also enable repeated cross validation. 

## Basic structure
To enable a flexible use and support different experiments, we provide a complex hierarchy. The lowest level is a fold, several folds means a repetition. And to support repeated cross validation a run can contains several repetitions.
The evaluation module than allows to aggregate the results on different levels: fold level, repetition level, run level. This structure is defined in the ml.evaluate module in the file model.py. The same structure exists for the evaluation to create an evaluation model that collects all results. It is defined in the same place.

## Evaluation Types:
Currently we provide four different evaluation types:
* Architecture Evaluation: Evaluates the architecture of the trained model (trainable parameters, training duration, etc.)
* Total Evaluation: Evaluates the total performance of the trained models
* PerPrefixLengthEvaluation: Evaluates a model in dependency of the prefix length
* PerActivityEvaluation: Evaluates the model performance for the different activities


## Commands
Evaluate architecture: 
`python -m ml.main evaluate-architecture "D:\runs_8\compStudy"  "base" "1_mixed_True_1.2" "2_mixed_True_1.4" "3_mixed_True_1.6" "3_mixed_True_2" -aggregate_on "fold" -num_precision 
3`

Evaluate gain:
`python -m ml.main evaluate-gain "D:\runs_8\compStudy"  "base" "1_mixed_True_1.2" "2_mixed_True_1.4" "3_mixed_True_1.6" "3_mixed_True_2" -d "Helpdesk" -d "BPIC12" -a "Buksh" -a "Camargo" -m "Accuracy"
`

## Tests
1. Activate the conda environment (conda activate pbpm-experiment)
2. Execute test suite by running pytest in the root directory 
3. To generate test coverage reports use `pytest --cov ml`
4. To get a detailed report about missing lines use `pytest --cov ml --cov-report html`