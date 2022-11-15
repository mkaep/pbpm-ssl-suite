from ml.core import model
from ml.persistence import json

if __name__ == '__main__':
    datasets = [
        model.Dataset('Helpdesk', r'D:\PBPM_Approaches\experiment\data\Helpdesk.xes'),
        model.Dataset('Sepsis', r'D:\PBPM_Approaches\experiment\data\Sepsis.xes'),
        model.Dataset('BPIC13', r'D:\PBPM_Approaches\experiment\data\BPI_Challenge_2013_closed_problems.xes'),
        #model.Dataset('BPIC12', r'D:\PBPM_Approaches\experiment\data\BPI_Challenge_2012.xes'),
        model.Dataset('BPIC15_1', r'D:\PBPM_Approaches\experiment\data\BPIC15_1.xes')
    ]

    approaches = [
        model.Approach('Buksh', 'process-transformer', 'D:\\PBPM_Approaches\\processtransformer\\main.py', {
            'task': 'next_activity',
            'epochs': 10,
            'batch_size': 12,
            'learning_rate': 0.001,
            'gpu': 0,
        }),
        model.Approach('Pasquadibisceglie', 'pasquadibisceglie', 'D:\\PBPM_Approaches\\pasquadibisceglie\\main.py', {
            'batch_size': 128,
            'epochs': 500,
            'n_layers': 2,
            'reg': 0.0001,
            'validation_split': 0.2,
            'patience': 6
        }),
        model.Approach('Camargo', 'camargo', 'D:/PBPM_Approaches/camargo/main.py', {
            'n_gram_size': 5,
            'model_type': 'shared_categorical',
            'l_size': 100,
            'imp': 2,
            'lstm_act': 'tanh',
            'optim': 'Nadam',
            'epochs': 200,
            'emb_epochs': 1,
            'batch_size': 32,
            'dense_act': None,
            'rp_sim': 0.85
        }),
        model.Approach('Theis', 'theis-pydream', 'D:/PBPM_Approaches/theis-pydream/main.py', {
            'seed': 1,
            'epochs': 100,
            'batch_size': 64,
            'dropout_rate': 0.2,
            'eval_size': 0.1,
            'activation_function': 'relu'
        }),
    ]

    splitter_configuration = model.SplitterConfiguration(name='time',
                                                         training_size=0.7,
                                                         by='first',
                                                         seeds=[42, 567, 789],
                                                         repetitions=3,
                                                         folds=2)

    augmentor_names = [model.AugmentorConfig('RandomInsertion', {}),
                       model.AugmentorConfig('RandomDeletion', {}),
                       model.AugmentorConfig('ParallelSwap', {}),
                       model.AugmentorConfig('FragmentAugmentation', {}),
                       model.AugmentorConfig('ReworkActivity', {}),
                       model.AugmentorConfig('DeleteReworkActivity', {}),
                       model.AugmentorConfig('RandomReplacement', {}),
                       model.AugmentorConfig('RandomSwap', {}),
                       model.AugmentorConfig('LoopAugmentation', {'max_additional_repetitions': 3,
                                                                  'duration_tolerance': 0.2})
                       ]

    # todo abbruchkriterium fuer single und parallel swap, wenn es nicht möglich ist, macht es keinen Sinn es immer weiter zu probieren
    augmentation_strategies_config = [
        model.AugmentationStrategyConfig(1, 'mixed', 42, augmentor_names, 1.2, True),
        model.AugmentationStrategyConfig(2, 'mixed', 42, augmentor_names, 1.4, True),
        model.AugmentationStrategyConfig(3, 'mixed', 42, augmentor_names, 1.6, True),
        model.AugmentationStrategyConfig(3, 'mixed', 42, augmentor_names, 2, True)
    ]

    experiment = model.AugmentationExperiment('compStudy', 'data', 'runs',
                                              'D:\\PBPM_Approaches\\experiment\\evaluation_10', datasets, approaches,
                                              splitter_configuration, 2, augmentation_strategies_config)

    exporter = json.JsonExperimentExporter(r'/experiments/exp_small.json')
    exporter.save(experiment)

    importer = json.JsonExperimentImporter(r'/experiments/exp_small.json')
    loaded_experiment = importer.load()
    print(loaded_experiment)
    print(isinstance(loaded_experiment, model.AugmentationExperiment))
    print(isinstance(loaded_experiment, model.Experiment))





