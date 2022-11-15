from ml.core import model


class TestModel:
    def test_job_to_dict(self, job):
        expected_dict = {
            'approach': {
                'name': 'Test Approach',
                'env_name': 'test_env',
                'dir': 'tests/persistence/res',
                'hyperparameter': {
                    'task': 'next_activity',
                    'epochs': 1,  # 10
                    'batch_size': 12,
                    'learning_rate': 0.001,
                    'gpu': 0
                }
            },
            'dataset': 'Test Log',
            'iteration': 1,
            'fold': 1,
            'path_training_file': '',
            'path_test_file': '',
            'path_complete_log': '',
            'job_directory': '',
            'metadata': {}
        }

        assert expected_dict == job().to_dict()

    def test_job_from_dict(self, job):
        data = {
            'approach': {
                'name': 'Test Approach',
                'env_name': 'test_env',
                'dir': 'tests/persistence/res',
                'hyperparameter': {
                    'task': 'next_activity',
                    'epochs': 1,  # 10
                    'batch_size': 12,
                    'learning_rate': 0.001,
                    'gpu': 0
                }
            },
            'dataset': 'Test Log',
            'iteration': 1,
            'fold': 1,
            'path_training_file': '',
            'path_test_file': '',
            'path_complete_log': '',
            'job_directory': '',
            'metadata': {}
        }

        assert model.Job.from_dict(data) == job()

    def test_get_conda_command(self, job, approach):
        path_training_file = 'files/train.xes'
        path_test_file = 'files/test.xes'
        path_complete_log = 'files/complete.xes'
        path_job_directory = 'file/job_dir/'

        job = job(path_training_file=path_training_file,
                  path_test_file=path_test_file,
                  path_complete_log=path_complete_log,
                  job_directory=path_job_directory)

        conda_command = 'conda run --no-capture-output -n test_env python tests/persistence/res ' \
                        '--original_data files/complete.xes ' \
                        '--train_data files/train.xes ' \
                        '--test_data files/test.xes ' \
                        '--result_dir file/job_dir/ ' \
                        '--task next_activity --epochs 1 --batch_size 12 --learning_rate 0.001 --gpu 0 '

        assert conda_command == job.get_conda_command()

    def test_approach_to_dict(self, approach):
        expected_dict = {
            'name': 'Test Approach',
            'env_name': 'test_env',
            'dir': 'tests/persistence/res',
            'hyperparameter': {
                'task': 'next_activity',
                'epochs': 1,  # 10
                'batch_size': 12,
                'learning_rate': 0.001,
                'gpu': 0
            }
        }

        assert expected_dict == approach().to_dict()

    def test_approach_from_dict(self, approach):
        data = {
            'name': 'Test Approach',
            'env_name': 'test_env',
            'dir': 'tests/persistence/res',
            'hyperparameter': {
                'task': 'next_activity',
                'epochs': 1,  # 10
                'batch_size': 12,
                'learning_rate': 0.001,
                'gpu': 0
            }
        }

        assert model.Approach.from_dict(data) == approach()

    def test_dataset_to_dict(self, dataset):
        expected_dict = {
            'name': 'Sample Log',
            'file_path': 'tests/persistence/res/sample_log.xes'
        }

        assert expected_dict == dataset().to_dict()

    def test_dataset_from_dict(self, dataset):
        data = {
            'name': 'Sample Log',
            'file_path': 'tests/persistence/res/sample_log.xes'
        }

        assert model.Dataset.from_dict(data) == dataset()

    def test_splitter_configuration_to_dict(self, splitter_configuration):
        expected_dict = {
            'name': 'time',
            'training_size': 0.7,
            'by': 'first',
            'seeds': [42],
            'repetitions': 1,
            'folds': 1
        }

        assert expected_dict == splitter_configuration().to_dict()

    def test_splitter_configuration_from_dict(self, splitter_configuration):
        data = {
            'name': 'time',
            'training_size': 0.7,
            'by': 'first',
            'seeds': [42],
            'repetitions': 1,
            'folds': 1
        }

        assert model.SplitterConfiguration.from_dict(data) == splitter_configuration()

    def test_experiment_to_dict(self, experiment):
        expected_dict = {
            'name': 'test_experiment',
            'data_dir': '../data/',
            'run_dir': '../run/',
            'evaluation_dir': '../eval/',
            'event_logs': [{
                'name': 'Sample Log',
                'file_path': 'tests/persistence/res/sample_log.xes'
            }],
            'approaches': [{
                'name': 'Test Approach',
                'env_name': 'test_env',
                'dir': 'tests/persistence/res',
                'hyperparameter': {
                    'task': 'next_activity',
                    'epochs': 1,  # 10
                    'batch_size': 12,
                    'learning_rate': 0.001,
                    'gpu': 0
                }
            }],
            'splitter_configuration': {
                'name': 'time',
                'training_size': 0.7,
                'by': 'first',
                'seeds': [42],
                'repetitions': 1,
                'folds': 1
            },
            'min_pref_length': 2
        }

        assert expected_dict == experiment().to_dict()

    def test_experiment_from_dict(self, experiment):
        data = {
            'name': 'test_experiment',
            'data_dir': '../data/',
            'run_dir': '../run/',
            'evaluation_dir': '../eval/',
            'event_logs': [{
                'name': 'Sample Log',
                'file_path': 'tests/persistence/res/sample_log.xes'
            }],
            'approaches': [{
                'name': 'Test Approach',
                'env_name': 'test_env',
                'dir': 'tests/persistence/res',
                'hyperparameter': {
                    'task': 'next_activity',
                    'epochs': 1,  # 10
                    'batch_size': 12,
                    'learning_rate': 0.001,
                    'gpu': 0
                }
            }],
            'splitter_configuration': {
                'name': 'time',
                'training_size': 0.7,
                'by': 'first',
                'seeds': [42],
                'repetitions': 1,
                'folds': 1
            },
            'min_pref_length': 2
        }

        assert model.Experiment.from_dict(data) == experiment()

    def test_experiment_get_dataset_names(self, experiment):
        exp = experiment()
        assert exp.get_dataset_names() == ['Sample Log']

    def test_experiment_get_approaches_names(self, experiment):
        exp = experiment()
        assert exp.get_approach_names() == ['Test Approach']

    def test_augmentation_experiment_to_dict(self, augmentation_experiment):
        expected_dict = {
            'name': 'test_augmentation_experiment',
            'data_dir': '../data/',
            'run_dir': '../run/',
            'evaluation_dir': '../eval/',
            'event_logs': [{
                'name': 'Sample Log',
                'file_path': 'tests/persistence/res/sample_log.xes'
            }],
            'approaches': [{
                'name': 'Test Approach',
                'env_name': 'test_env',
                'dir': 'tests/persistence/res',
                'hyperparameter': {
                    'task': 'next_activity',
                    'epochs': 1,  # 10
                    'batch_size': 12,
                    'learning_rate': 0.001,
                    'gpu': 0
                }
            }],
            'splitter_configuration': {
                'name': 'time',
                'training_size': 0.7,
                'by': 'first',
                'seeds': [42],
                'repetitions': 1,
                'folds': 1
            },
            'min_pref_length': 2,
            'augmentation_strategies': [
                {
                    'id': 0,
                    'name': 'mixed',
                    'seed': 42,
                    'augmentor_names': ['RandomInsertion', 'ParallelSwap'],
                    'augmentation_factor': 1.2,
                    'allow_multiple': True,
                }
            ]
        }

        assert expected_dict == augmentation_experiment().to_dict()

    def test_augmentation_experiment_from_dict(self, augmentation_experiment):
        data = {
            'name': 'test_augmentation_experiment',
            'data_dir': '../data/',
            'run_dir': '../run/',
            'evaluation_dir': '../eval/',
            'event_logs': [{
                'name': 'Sample Log',
                'file_path': 'tests/persistence/res/sample_log.xes'
            }],
            'approaches': [{
                'name': 'Test Approach',
                'env_name': 'test_env',
                'dir': 'tests/persistence/res',
                'hyperparameter': {
                    'task': 'next_activity',
                    'epochs': 1,  # 10
                    'batch_size': 12,
                    'learning_rate': 0.001,
                    'gpu': 0
                }
            }],
            'splitter_configuration': {
                'name': 'time',
                'training_size': 0.7,
                'by': 'first',
                'seeds': [42],
                'repetitions': 1,
                'folds': 1
            },
            'min_pref_length': 2,
            'augmentation_strategies': [
                {
                    'id': 0,
                    'name': 'mixed',
                    'seed': 42,
                    'augmentor_names': ['RandomInsertion', 'ParallelSwap'],
                    'augmentation_factor': 1.2,
                    'allow_multiple': True,
                }
            ]
        }

        assert model.AugmentationExperiment.from_dict(data) == augmentation_experiment()

