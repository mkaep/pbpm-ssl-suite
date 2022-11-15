import pytest
import os
from ml.persistence import json
from ml.core import model


class TestJsonExperimentExport:
    # todo extend tests to work also with augmentation_experiment
    def test_save(self, experiment, tmp_path):
        target_file = tmp_path / 'tests/persistence/res/test_experiment.json'
        exporter = json.JsonExperimentExporter(str(target_file))
        exp = experiment()
        exporter.save(exp)
        assert os.path.isfile(str(target_file)) is True
        assert os.path.getsize(str(target_file)) > 0


class TestJsonExperimentImport:
# todo extend tests to work also with augmentation_experiment
    def test_failure_load_empty_file(self):
        # reason: file is empty
        with pytest.raises(AssertionError):
            importer = json.JsonExperimentImporter('tests/persistence/res/empty.json')
            importer.load()

    def test_load_default(self, experiment):
        importer = json.JsonExperimentImporter('tests/persistence/res/sample_experiment.json')
        loaded_experiment = importer.load()

        assert loaded_experiment == experiment()

    def test_save_and_load(self, experiment, tmp_path):
        exp = experiment()
        target_file = tmp_path / 'test/res/test_experiment_1.json'
        exporter = json.JsonExperimentExporter(str(target_file))
        exporter.save(exp)

        importer = json.JsonExperimentImporter(str(target_file))
        loaded_experiment = importer.load()

        assert exp == loaded_experiment


class TestJsonJobExport:

    def test_save(self, job, tmp_path):
        jobs = [job() for _ in range(5)]
        target_file = tmp_path / 'test/res/test.jsonl'
        exporter = json.JsonJobExporter(str(target_file))
        exporter.save(jobs)

        assert os.path.isfile(str(target_file)) is True
        assert os.path.getsize(str(target_file)) > 0
        with open(str(target_file), 'r') as f:
            assert len(list(f)) == 5


class TestJsonJobImport:
    def test_load(self):
        expected_jobs = [
            model.Job(approach=model.Approach(
                                    name='Test Approach 1',
                                    env_name='test_env_1',
                                    dir='test_1/main.py',
                                    hyperparameter={'task': 'next_activity', 'epochs': 10, 'batch_size': 12,
                                                    'learning_rate': 0.001, 'gpu': 0}
                                ),
                      dataset='Test Dataset 1',
                      iteration=0,
                      fold=0,
                      path_training_file='test_1/train.xes',
                      path_test_file='test_1/test_pref.xes',
                      path_complete_log='test_1/_common_data/preprocessed.xes',
                      job_directory='test_1/base',
                      metadata={'Key 1':  'Test1'}),
            model.Job(approach=model.Approach(
                                    name='Test Approach 2',
                                    env_name='test_env_2',
                                    dir='test_2/main.py',
                                    hyperparameter={'batch_size': 128, 'epochs': 500, 'n_layers': 2, 'reg': 0.0001,
                                                    'validation_split': 0.2, 'patience': None}
                                ),
                      dataset='Test Dataset 2',
                      iteration=1,
                      fold=1,
                      path_training_file='Test_2/train.xes',
                      path_test_file='test_2/test_pref.xes',
                      path_complete_log='test_2/Helpdesk/_common_data/preprocessed.xes',
                      job_directory='test_2/aug_strat_1',
                      metadata={'Key 1':  'Test2', 'Creator':  'Test'})
        ]
        importer = json.JsonJobImporter('tests/persistence/res/sample_jobs.jsonl')
        assert importer.load() == expected_jobs

    def test_load_empty_file(self):
        importer = json.JsonJobImporter('tests/persistence/res/empty_job.jsonl')
        assert importer.load() == []

    def test_save_and_load(self, job, tmp_path):
        jobs = [job() for _ in range(5)]
        target_file = tmp_path / 'test/res/test.jsonl'
        exporter = json.JsonJobExporter(str(target_file))
        exporter.save(jobs)

        importer = json.JsonJobImporter(str(target_file))
        loaded_jobs = importer.load()

        assert jobs == loaded_jobs
