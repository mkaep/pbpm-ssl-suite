import pytest
import pm4py
import os
from ml.pipeline import augmentation_pipeline
from pm4py.objects.log.obj import EventLog
from ml.core import loader
from ml.prepare import splitter
from ml.augmentation import augmentation_strategy, easy_augmentors
from mockito import when
from unittest import mock


class TestAugmentationPipeline:

    def test_get_activities(self):
        event_log = pm4py.read_xes('tests/pipeline/res/sample_log.xes')
        activities = augmentation_pipeline.get_activities(event_log)
        expected_activities = {'Assign seriousness', 'Take in charge ticket', 'Resolve ticket', 'Closed',
                               'Insert ticket', 'Wait', 'Create SW anomaly', 'Require upgrade'}
        assert set(activities) == expected_activities

    def test_create_fold_dir(self, tmp_path):
        common_data_dir = tmp_path / '_common_data'
        repetition = 2
        fold = 1

        fold_dir, train_dir, test_dir = augmentation_pipeline.create_fold_dir(str(common_data_dir), repetition, fold)

        assert fold_dir == os.path.join(common_data_dir, 'rep_2', 'fold_1')
        assert train_dir == os.path.join(common_data_dir, 'rep_2', 'fold_1', 'train')
        assert test_dir == os.path.join(common_data_dir, 'rep_2', 'fold_1', 'test')

        assert os.path.isdir(os.path.join(common_data_dir, 'rep_2'))
        assert os.path.isdir(os.path.join(common_data_dir, 'rep_2', 'fold_1'))
        assert os.path.isdir(os.path.join(common_data_dir, 'rep_2', 'fold_1', 'train'))
        assert os.path.isdir(os.path.join(common_data_dir, 'rep_2', 'fold_1', 'test'))

    def test_split_random_splitter(self, splitter_configuration):
        config = splitter_configuration(name='random', repetitions=2, seeds=[42, 43])
        event_log = pm4py.read_xes('tests/pipeline/res/sample_log.xes')
        split_results = augmentation_pipeline.split(event_log, config, 1)

        assert len(split_results) == 1
        assert [i.type == 'random' for i in split_results][0] is True

    def test_split_random_splitter_failure_case_wrong_config(self, splitter_configuration):
        config = splitter_configuration(name='random', repetitions=3, seeds=[42])
        event_log = pm4py.read_xes('tests/pipeline/res/sample_log.xes')
        # reason: violates seed consistency
        with pytest.raises(AssertionError):
            augmentation_pipeline.split(event_log, config, 1)

    def test_split_time_splitter(self, splitter_configuration):
        config = splitter_configuration(name='time', training_size=0.7, by='first')
        event_log = pm4py.read_xes('tests/pipeline/res/sample_log.xes')
        split_results = augmentation_pipeline.split(event_log, config, 1)

        assert len(split_results) == 1
        assert [i.type == 'time_first' for i in split_results][0] is True

    def test_split_time_splitter_failure_case_wrong_config(self, splitter_configuration):
        config = splitter_configuration(name='time', training_size=-0.5, by='first')
        event_log = pm4py.read_xes('tests/pipeline/res/sample_log.xes')
        # reason: violates training size
        with pytest.raises(AssertionError):
            augmentation_pipeline.split(event_log, config, 1)

    def test_split_k_fold_splitter(self, splitter_configuration):
        config = splitter_configuration(name='k_fold', repetitions=2, folds=3, seeds=[42, 43])
        event_log = pm4py.read_xes('tests/pipeline/res/sample_log.xes')
        split_results = augmentation_pipeline.split(event_log, config, 1)

        assert len(split_results) == 3
        assert [i.type == 'k_fold' for i in split_results] == [True, True, True]

    def test_split_k_fold_splitter_failure_case_wrong_config(self, splitter_configuration):
        config = splitter_configuration(name='k_fold', repetitions=2, folds=-1, seeds=[42, 43])
        event_log = pm4py.read_xes('tests/pipeline/res/sample_log.xes')
        with pytest.raises(AssertionError):
            augmentation_pipeline.split(event_log, config, 1)

    def test_split_failure_unknown_splitter(self, splitter_configuration):
        config = splitter_configuration(name='Not Existent Splitter')
        with pytest.raises(ValueError):
            augmentation_pipeline.split(event_log=EventLog(), splitter_configuration=config, repetition=1)

    def test_build_augmentation_strategies(self, augmentation_strategy_config):
        configs = [augmentation_strategy_config(id=1, name='mixed', augmentor_names=['RandomInsertion',
                                                                                     'ParallelSwap']),
                   augmentation_strategy_config(id=2, name='single', augmentor_names=['DeleteReworkActivity'])]
        activities = ['A', 'B', 'C']

        random_insertion_augmentor = easy_augmentors.RandomInsertion(activities)
        parallel_swap_augmentor = easy_augmentors.ParallelSwap()
        delete_rework_activity_augmentor = easy_augmentors.DeleteReworkActivity()

        exp_augmentation_strategies = [
            augmentation_strategy.MixedAugmentationStrategy(id=1,
                                                            augmentors=[random_insertion_augmentor,
                                                                        parallel_swap_augmentor],
                                                            augmentation_factor=1.2,
                                                            allow_multiple=True),
            augmentation_strategy.SingleAugmentationStrategy(id=2,
                                                             augmentors=[delete_rework_activity_augmentor],
                                                             augmentation_factor=1.2,
                                                             allow_multiple=True)
        ]

        assert augmentation_pipeline.build_augmentation_strategies(configs, activities) == exp_augmentation_strategies

    def test_run_pipeline(self, tmp_path, augmentation_experiment, dataset, approach, splitter_configuration,
                          augmentation_strategy_config):
        # todo evtl die run methode besser faken so dass auch ordner von den ansaetzen angelegt werden
        exp = augmentation_experiment(name='Test-Experiment',
                                      run_dir=str(tmp_path),
                                      event_logs=[
                                          dataset(name='Test_Log_1'),
                                          dataset(name='Test_Log_2')
                                      ],
                                      approaches=[
                                          approach(name='Approach_1'),
                                          approach(name='Approach_2'),
                                          approach(name='Approach_3')
                                      ],
                                      splitter_config=splitter_configuration(name='k_fold', repetitions=3, folds=2,
                                                                             seeds=[42, 128, 476]),
                                      aug_strat=[augmentation_strategy_config(id=1, name='mixed',
                                                                              augmentor_names=['RandomInsertion',
                                                                                               'ParallelSwap'],
                                                                              augmentation_factor=1.2,
                                                                              allow_multiple=True),
                                                 augmentation_strategy_config(id=2, name='single',
                                                                              augmentor_names=['DeleteReworkActivity'],
                                                                              augmentation_factor=1.2,
                                                                              allow_multiple=True
                                                                              )
                                                 ]
                                      )
        event_log = pm4py.read_xes('tests/pipeline/res/sample_log.xes')
        split_result = [splitter.SplitResult('k_fold', event_log, event_log),
                        splitter.SplitResult('k_fold', event_log, event_log)]

        # Return in all cases the same event log
        when(loader.Loader).load_event_log(...).thenReturn(event_log)
        when(augmentation_pipeline).split(...).thenReturn(split_result)

        with mock.patch('ml.augmentation.augmentation_strategy.MixedAugmentationStrategy') as mock_strategy:
            with mock.patch('ml.augmentation.augmentation_strategy.SingleAugmentationStrategy') as mock_single_strategy:
                mock_single_strategy.return_value.name = '2_single_True_1.2'
                mock_single_strategy.return_value.augment.return_value = event_log, {}, {}

                mock_strategy.return_value.name = '1_mixed_True_1.2'
                mock_strategy.return_value.augment.return_value = event_log, {}, {}

                with mock.patch('ml.pipeline.job_executor.JobExecutor') as mock_executor:
                    mock_executor.return_value.run.return_value = None
                    augmentation_pipeline.run_pipeline(exp)

        exp_folders = {'test', 'train', '1_mixed_True_1.2', '2_single_True_1.2'}

        def strategy_folder_has_files(path) -> bool:
            expected_files = {'aug_count.json', 'aug_record.json', 'train.xes'}
            return set(os.listdir(path)) == expected_files

        def train_folder_has_files(path) -> bool:
            expected_files = {'train.xes'}
            return set(os.listdir(path)) == expected_files

        def test_folder_has_files(path) -> bool:
            expected_files = {'test.xes', 'test_pref.xes', 'test_suf.xes', 'true_result.csv'}
            return set(os.listdir(path)) == expected_files

        # Check existence of preprocessed log
        assert os.path.isdir(os.path.join(tmp_path, 'Test-Experiment', '.jobs'))

        for dataset in exp.event_logs:
            common_data_path = os.path.join(tmp_path, 'Test-Experiment', dataset.name, '_common_data')
            assert os.path.exists(os.path.join(tmp_path, 'Test-Experiment', dataset.name, '_common_data',
                                               'preprocessed.xes'))
            assert os.path.isdir(os.path.join(common_data_path, 'rep_0'))
            assert os.path.isdir(os.path.join(common_data_path, 'rep_0', 'fold_0'))
            assert set(os.listdir(os.path.join(common_data_path, 'rep_0', 'fold_0'))) == exp_folders
            assert strategy_folder_has_files(
                os.path.join(common_data_path, 'rep_0', 'fold_0', '1_mixed_True_1.2')) is True
            assert strategy_folder_has_files(
                os.path.join(common_data_path, 'rep_0', 'fold_0', '2_single_True_1.2')) is True
            assert train_folder_has_files(os.path.join(common_data_path, 'rep_0', 'fold_0', 'train')) is True
            assert test_folder_has_files(os.path.join(common_data_path, 'rep_0', 'fold_0', 'test')) is True

            assert os.path.isdir(os.path.join(common_data_path, 'rep_0', 'fold_1'))
            assert set(os.listdir(os.path.join(common_data_path, 'rep_0', 'fold_1'))) == exp_folders
            assert strategy_folder_has_files(
                os.path.join(common_data_path, 'rep_0', 'fold_1', '1_mixed_True_1.2')) is True
            assert strategy_folder_has_files(
                os.path.join(common_data_path, 'rep_0', 'fold_1', '2_single_True_1.2')) is True
            assert train_folder_has_files(os.path.join(common_data_path, 'rep_0', 'fold_1', 'train')) is True
            assert test_folder_has_files(os.path.join(common_data_path, 'rep_0', 'fold_1', 'test')) is True

            assert os.path.isdir(os.path.join(common_data_path, 'rep_1'))
            assert os.path.isdir(os.path.join(common_data_path, 'rep_1', 'fold_0'))
            assert set(os.listdir(os.path.join(common_data_path, 'rep_1', 'fold_0'))) == exp_folders
            assert strategy_folder_has_files(
                os.path.join(common_data_path, 'rep_1', 'fold_0', '1_mixed_True_1.2')) is True
            assert strategy_folder_has_files(
                os.path.join(common_data_path, 'rep_1', 'fold_0', '2_single_True_1.2')) is True
            assert train_folder_has_files(os.path.join(common_data_path, 'rep_1', 'fold_0', 'train')) is True
            assert test_folder_has_files(os.path.join(common_data_path, 'rep_1', 'fold_0', 'test')) is True
            assert os.path.isdir(os.path.join(common_data_path, 'rep_1', 'fold_1'))
            assert set(os.listdir(os.path.join(common_data_path, 'rep_1', 'fold_1'))) == exp_folders
            assert strategy_folder_has_files(
                os.path.join(common_data_path, 'rep_1', 'fold_1', '1_mixed_True_1.2')) is True
            assert strategy_folder_has_files(
                os.path.join(common_data_path, 'rep_1', 'fold_1', '2_single_True_1.2')) is True
            assert train_folder_has_files(os.path.join(common_data_path, 'rep_1', 'fold_1', 'train')) is True
            assert test_folder_has_files(os.path.join(common_data_path, 'rep_1', 'fold_1', 'test')) is True

            assert os.path.isdir(os.path.join(common_data_path, 'rep_2'))
            assert os.path.isdir(os.path.join(common_data_path, 'rep_2', 'fold_0'))
            assert set(os.listdir(os.path.join(common_data_path, 'rep_2', 'fold_0'))) == exp_folders
            assert strategy_folder_has_files(
                os.path.join(common_data_path, 'rep_2', 'fold_0', '1_mixed_True_1.2')) is True
            assert strategy_folder_has_files(
                os.path.join(common_data_path, 'rep_2', 'fold_0', '2_single_True_1.2')) is True
            assert train_folder_has_files(os.path.join(common_data_path, 'rep_2', 'fold_0', 'train')) is True
            assert test_folder_has_files(os.path.join(common_data_path, 'rep_2', 'fold_0', 'test')) is True
            assert os.path.isdir(os.path.join(common_data_path, 'rep_2', 'fold_1'))
            assert set(os.listdir(os.path.join(common_data_path, 'rep_2', 'fold_1'))) == exp_folders
            assert strategy_folder_has_files(
                os.path.join(common_data_path, 'rep_2', 'fold_1', '1_mixed_True_1.2')) is True
            assert strategy_folder_has_files(
                os.path.join(common_data_path, 'rep_2', 'fold_1', '2_single_True_1.2')) is True
            assert train_folder_has_files(os.path.join(common_data_path, 'rep_2', 'fold_1', 'train')) is True
            assert test_folder_has_files(os.path.join(common_data_path, 'rep_2', 'fold_1', 'test')) is True

            for approach in {'Approach_1', 'Approach_2', 'Approach_3'}:
                assert os.path.isdir(os.path.join(tmp_path, 'Test-Experiment', dataset.name, approach))
