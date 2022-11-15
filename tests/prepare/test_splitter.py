import pandas as pd
import pm4py
import pytest
from datetime import datetime
from pm4py.objects.log.obj import EventLog
from ml.prepare import splitter
from ml.core import model
import os


class TestSplitter:

    def test_get_partition(self):
        case_ids = ['Case_1', 'Case_2', 'Case_3', 'Case_4', 'Case_5']
        training_size = 0.8

        training_cases, test_cases = splitter.get_partition(training_size, case_ids)

        assert len(training_cases) == 4
        assert len(test_cases) == 1
        assert ['Case_1', 'Case_2', 'Case_3', 'Case_4'] == training_cases
        assert ['Case_5'] == test_cases
        assert set(training_cases).isdisjoint(set(test_cases))

        case_empty = []
        training_size = 0.8
        training_cases, test_cases = splitter.get_partition(training_size, case_empty)
        assert len(training_cases) == 0
        assert len(test_cases) == 0
        assert [] == training_cases
        assert [] == test_cases

    def test_store(self, tmp_path):
        train_log = EventLog()
        test_log = EventLog()

        train_dir = tmp_path
        test_dir = tmp_path

        split_result = splitter.SplitResult('test', train_log, test_log)
        train_file, test_file = split_result.store(str(train_dir), str(test_dir))

        assert train_file == os.path.join(train_dir, 'train.xes')
        assert test_file == os.path.join(test_dir, 'test.xes')
        assert os.path.isfile(os.path.join(train_dir, 'train.xes'))
        assert os.path.isfile(os.path.join(train_dir, 'test.xes'))


class TestRandomSplitter:
    def test_is_repeatable(self):
        assert splitter.RandomSplitter.is_repeatable() is True

    def test_format_id(self):
        assert splitter.RandomSplitter.format_id() == 'random'

    def test_split(self):
        event_log = pm4py.read_xes('tests/prepare/res/sample_log_1.xes')
        expected_training_cases = {'Case2727', 'Case345', 'Case1', 'Case346'}
        expected_test_cases = {'Case2726'}

        for i in range(3):
            split_result = splitter.RandomSplitter().split(event_log, training_size=0.8, seed=436)
            assert len(split_result) == 1

            train_cases = [trace.attributes['concept:name'] for trace in split_result[0].train_log]
            test_cases = [trace.attributes['concept:name'] for trace in split_result[0].test_log]

            assert expected_training_cases == set(train_cases)
            assert expected_test_cases == set(test_cases)
            assert set(train_cases).isdisjoint(set(test_cases))

    def test_split_empty_log(self):
        split_result = splitter.RandomSplitter().split(EventLog(), training_size=0.8)
        assert len(split_result) == 1
        assert len(split_result[0].train_log) == 0
        assert len(split_result[0].test_log) == 0

    def test_split_invalid_ratio(self):
        # reason: invalid ratio (> 1)
        with pytest.raises(AssertionError):
            event_log = pm4py.read_xes('tests/prepare/res/sample_log_1.xes')
            splitter.RandomSplitter().split(event_log, training_size=1.8)

    def test_check_configuration_positive(self, splitter_configuration):
        config = splitter_configuration()
        assert splitter.RandomSplitter.check_configuration(config)

    def test_check_configuration_negative(self, splitter_configuration):
        config = splitter_configuration(repetitions=2)
        assert splitter.RandomSplitter.check_configuration(config) is False

        config = model.SplitterConfiguration('random', repetitions=None)
        assert splitter.RandomSplitter.check_configuration(config) is False

        config = model.SplitterConfiguration('random', training_size=None)
        assert splitter.RandomSplitter.check_configuration(config) is False

        config = model.SplitterConfiguration('random', seeds=None)
        assert splitter.RandomSplitter.check_configuration(config) is False

        config = model.SplitterConfiguration('random', training_size=1.2)
        assert splitter.RandomSplitter.check_configuration(config) is False

        config = model.SplitterConfiguration('random', training_size=0.0)
        assert splitter.RandomSplitter.check_configuration(config) is False


class TestKFoldSplitter:
    def test_is_repeatable(self):
        assert splitter.KFoldSplitter.is_repeatable() is True

    def test_format_id(self):
        assert splitter.KFoldSplitter.format_id() == 'k_fold'

    def test_split(self):
        event_log = pm4py.read_xes('tests/prepare/res/sample_log_1.xes')
        expected_train_cases_fold_1 = {'Case345', 'Case2726', 'Case1', 'Case2727'}
        expected_test_cases_fold_1 = {'Case346'}
        expected_train_cases_fold_2 = {'Case346', 'Case2726', 'Case345', 'Case2727'}
        expected_test_cases_fold_2 = {'Case1'}
        expected_train_cases_fold_3 = {'Case1', 'Case346'}
        expected_test_cases_fold_3 = {'Case2726', 'Case345', 'Case2727'}

        for i in range(0, 3):
            split_result = splitter.KFoldSplitter().split(event_log, folds=3, seed=436)
            assert len(split_result) == 3

            train_cases_fold_1 = [trace.attributes['concept:name'] for trace in split_result[0].train_log]
            test_cases_fold_1 = [trace.attributes['concept:name'] for trace in split_result[0].test_log]

            assert expected_train_cases_fold_1 == set(train_cases_fold_1)
            assert expected_test_cases_fold_1 == set(test_cases_fold_1)
            assert set(train_cases_fold_1).isdisjoint(set(test_cases_fold_1))

            train_cases_fold_2 = [trace.attributes['concept:name'] for trace in split_result[1].train_log]
            test_cases_fold_2 = [trace.attributes['concept:name'] for trace in split_result[1].test_log]

            assert expected_train_cases_fold_2 == set(train_cases_fold_2)
            assert expected_test_cases_fold_2 == set(test_cases_fold_2)
            assert set(train_cases_fold_2).isdisjoint(set(test_cases_fold_2))

            train_cases_fold_3 = [trace.attributes['concept:name'] for trace in split_result[2].train_log]
            test_cases_fold_3 = [trace.attributes['concept:name'] for trace in split_result[2].test_log]

            assert expected_train_cases_fold_3 == set(train_cases_fold_3)
            assert expected_test_cases_fold_3 == set(test_cases_fold_3)
            assert set(train_cases_fold_3).isdisjoint(set(test_cases_fold_3))

    def test_split_invalid_folds(self):
        # reason: invalid number of folds
        with pytest.raises(AssertionError):
            event_log = pm4py.read_xes('tests/prepare/res/sample_log_1.xes')
            splitter.KFoldSplitter().split(event_log, folds=-1)

    def test_split_empty_log(self):
        # reason: empty event log
        with pytest.raises(AssertionError):
            splitter.KFoldSplitter().split(EventLog(), folds=5)

    def test_check_configuration_positive(self, splitter_configuration):
        config = splitter_configuration(folds=2)
        assert splitter.KFoldSplitter.check_configuration(config)

    def test_check_configuration_negative(self, splitter_configuration):
        config = splitter_configuration(repetitions=2)
        assert splitter.KFoldSplitter.check_configuration(config) is False

        config = splitter_configuration(folds=1)
        assert splitter.KFoldSplitter.check_configuration(config) is False

        config = model.SplitterConfiguration('k_fold', folds=None)
        assert splitter.KFoldSplitter.check_configuration(config) is False

        config = model.SplitterConfiguration('k_fold', seeds=None)
        assert splitter.KFoldSplitter.check_configuration(config) is False

        config = model.SplitterConfiguration('k_fold', repetitions=None)
        assert splitter.KFoldSplitter.check_configuration(config) is False


class TestTimeSplitter:
    def test_create_sorted_event_log(self):
        event_log = pm4py.read_xes('tests/prepare/res/sample_log_1.xes')

        df_first = splitter.TimeSplitter.create_sorted_event_log(event_log, 'first')
        expected_first = {
            'Case2726': datetime.fromisoformat('2011-01-11T11:20:28.000+02:00'),
            'Case2727': datetime.fromisoformat('2012-03-20T15:58:26.000+02:00'),
            'Case1': datetime.fromisoformat('2012-10-09T14:50:17.000+03:00'),
            'Case346': datetime.fromisoformat('2013-06-17T14:57:04.000+03:00'),
            'Case345': datetime.fromisoformat('2013-11-22T11:48:57.000+02:00')
        }
        expected_first_df = pd.DataFrame(expected_first.items(), columns=['case_id', 'time'])
        assert expected_first_df.equals(df_first)

        df_last = splitter.TimeSplitter.create_sorted_event_log(event_log, 'last')
        expected_last = {
            'Case2726': datetime.fromisoformat('2011-02-23T08:10:27.000+02:00'),
            'Case2727': datetime.fromisoformat('2012-05-05T14:04:06.000+03:00'),
            'Case1': datetime.fromisoformat('2012-11-09T12:54:39.000+02:00'),
            'Case346': datetime.fromisoformat('2013-08-09T09:02:49.000+03:00'),
            'Case345': datetime.fromisoformat('2013-12-25T08:13:45.000+02:00')
        }
        expected_last_df = pd.DataFrame(expected_last.items(), columns=['case_id', 'time'])
        assert expected_last_df.equals(df_last)

    def test_split(self):
        event_log = pm4py.read_xes('tests/prepare/res/sample_log_1.xes')
        split_result_first = splitter.TimeSplitter().split(event_log, training_size=0.8, by='first')

        assert len(split_result_first) == 1
        assert split_result_first[0].type == 'time_first'

        train_cases = [trace.attributes['concept:name'] for trace in split_result_first[0].train_log]
        test_cases = [trace.attributes['concept:name'] for trace in split_result_first[0].test_log]

        expected_training_cases = {'Case2726', 'Case2727', 'Case1', 'Case346'}
        expected_test_cases = {'Case345'}

        assert expected_training_cases == set(train_cases)
        assert expected_test_cases == set(test_cases)

        split_result_last = splitter.TimeSplitter().split(event_log, training_size=0.8, by='last')
        train_cases = [trace.attributes['concept:name'] for trace in split_result_last[0].train_log]
        test_cases = [trace.attributes['concept:name'] for trace in split_result_last[0].test_log]
        assert len(split_result_last) == 1
        assert split_result_last[0].type == 'time_last'
        assert expected_training_cases == set(train_cases)
        assert expected_test_cases == set(test_cases)

    def test_split_invalid_ratio(self):
        # reason: invalid ratio (> 1)
        with pytest.raises(AssertionError):
            event_log = pm4py.read_xes('tests/prepare/res/sample_log_1.xes')
            splitter.TimeSplitter().split(event_log, training_size=1.8, by='first')

    def test_split_empty_log(self):
        split_result = splitter.TimeSplitter().split(EventLog(), training_size=0.8, by='first')
        assert len(split_result) == 1
        assert len(split_result[0].train_log) == 0
        assert len(split_result[0].test_log) == 0

    def test_is_repeatable(self):
        assert splitter.TimeSplitter.is_repeatable() is False

    def test_format_id(self):
        assert splitter.TimeSplitter.format_id() == 'time'

    def test_check_configuration_positive(self, splitter_configuration):
        config = splitter_configuration()
        assert splitter.TimeSplitter.check_configuration(config)

    def test_check_configuration_negative(self, splitter_configuration):
        config = model.SplitterConfiguration('time', by=None, training_size=0.7)
        assert splitter.TimeSplitter.check_configuration(config) is False

        config = model.SplitterConfiguration('time', by='first', training_size=None)
        assert splitter.TimeSplitter.check_configuration(config) is False

        config = model.SplitterConfiguration('time', by=None, training_size=None)
        assert splitter.TimeSplitter.check_configuration(config) is False

        config = model.SplitterConfiguration('time', by='first', training_size=1.2)
        assert splitter.TimeSplitter.check_configuration(config) is False

        config = model.SplitterConfiguration('time', by='first', training_size=0.0)
        assert splitter.TimeSplitter.check_configuration(config) is False

        config = model.SplitterConfiguration('time', by='something', training_size=0.7)
        assert splitter.TimeSplitter.check_configuration(config) is False
