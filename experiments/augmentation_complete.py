from ml.core import model
from ml.augmentation import augmentation_strategy


if __name__ == '__main__':
    datasets = [
        model.Dataset('Helpdesk', r'D:\PBPM_Approaches\experiment\data\Helpdesk.xes'),
        model.Dataset('BPIC12', r'D:\PBPM_Approaches\experiment\data\BPI_Challenge_2012.xes'),
        model.Dataset('Sepsis', r'D:\PBPM_Approaches\experiment\data\Sepsis.xes'),
        model.Dataset('Permit', '../data/Permit.xes'),
        model.Dataset('BPIC13', r'D:\PBPM_Approaches\experiment\data\BPI_Challenge_2013_closed_problems.xes'),
        model.Dataset('BPIC17', '../data/BPI_Challenge_2017.xes.gz'),
        model.Dataset('BPIC18', '../data/BPI_Challenge_2018.xes.gz'),
        model.Dataset('BPIC15_1', r'D:\PBPM_Approaches\experiment\data\BPIC15_1.xes'),
        model.Dataset('BPIC15_2', '../data/BPIC15_2.xes'),
        model.Dataset('BPIC15_3', '../data/BPIC15_3.xes'),
        model.Dataset('BPIC15_4', '../data/BPIC15_4.xes'),
        model.Dataset('BPIC15_5', '../data/BPIC15_5.xes'),
        model.Dataset('Hospital', '../data/Hospital.xes'),
    ]

    approaches = [
        model.Approach('Khan', 'khan', 'D:/PBPM_Approaches/khan/main.py', {
            'iterations': 200,  # 20000
            'hidden_controller_dim': 100,
            'use_emb': False,
            'use_mem': True,
            'decoder_mode': True,
            'dual_controller': True,
            'write_protect': True,
            'attend_dim': 0,
            'batch_size': 1,
            'sequence_max_length': 100,
            'words_count': 5,
            'word_size': 20,
            'read_heads': 1
        }),
        model.Approach('Mauro', 'mauro', 'D:/PBPM_Approaches/mauro/main.py', {
            'iterations': 1,  # 20
            'model_type': 'ACT',
            'epochs': 2,  # 200
            'validation_split': 0.2,
            'patience': 20
        }),
        model.Approach('Theis', 'theis-pydream', 'D:/PBPM_Approaches/theis-pydream/main.py', {
            'seed': 1,
            'epochs': 1,  # 100
            'batch_size': 64,
            'dropout_rate': 0.2,
            'eval_size': 0.1,
            'activation_function': 'relu'
        }),
        model.Approach('Tax', 'tax', 'D:/PBPM_Approaches/tax/main.py', {
            'prediction_task': "ACT_TIME",
            'epochs': 1,  # 500
            'validation_split': 0.2,
            'learning_rate': 0.002,
            'schedule_decay': 0.004,
            'clipvalue': 3,
            'patience': 10,
            'prediction_size': 3
        }),
        model.Approach('Pasquadibisceglie', 'pasquadibisceglie', 'D:/PBPM_Approaches/pasquadibisceglie/main.py', {
            'batch_size': 128,
            'epochs': 1,  # 500
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
            'epochs': 1,  # 200
            'emb_epochs': 2,  # 100
            'batch_size': 32,
            'dense_act': None,
            'rp_sim': 0.85
        }),
        model.Approach('Buksh', 'process-transformer', 'D:/PBPM_Approaches/processtransformer/main.py', {
            'task': 'next_activity',
            'epochs': 1,  # 10
            'batch_size': 12,
            'learning_rate': 0.001,
            'gpu': 0,
        })
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
        model.AugmentationStrategyConfig(3, 'single', 42, ['RandomInsertion'], 1.2, True),
        model.AugmentationStrategyConfig(4, 'single', 42, ['RandomDeletion'], 1.2, True),
        model.AugmentationStrategyConfig(5, 'single', 42, ['ParallelSwap'], 1.2, True)
    ]

    experiment = model.AugmentationExperiment('compStudy', 'data', '/runs_10',
                                              'D:\\PBPM_Approaches\\experiment\\evaluation_10', datasets, approaches,
                                              splitter_configuration, 2, augmentation_strategies_config)

