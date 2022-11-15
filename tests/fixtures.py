import datetime
import random
import string
import typing

import pm4py

from ml.core import model
from ml.augmentation import augmentation_strategy
from ml.evaluate import ev_model
import pytest
from pm4py.objects.log.obj import Trace, Event

# Make tests reproducible
random.seed(42)


def random_string(k: int = 10) -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))


@pytest.fixture
def trace():
    activities = ['Closed', 'RESOLVED', 'Take in charge ticket', 'Create SW anomaly', 'Require upgrade',
                  'Resolve SW anomaly', 'INVALID', 'DUPLICATE', 'Wait', 'Resolve ticket', 'VERIFIED',
                  'Schedule intervention', 'Insert ticket', 'Assign seriousness']

    # todo evtl so umbauen das komplett fiktiv generiert wird
    def build_trace(length: int = None, id: str = None) -> Trace:
        assert length is None or length >= 0
        event_log = pm4py.read_xes('tests/augmentation/res/trace.xes')
        trace = event_log[0].__deepcopy__()
        if length is None or length == len(trace):
            return trace
        if id is not None:
            trace.attributes['concept:name'] = id
        if length > len(trace):
            while length > len(trace):
                random_duration = random.uniform(0, 172_800)
                random.shuffle(activities)
                event = Event()
                event.__setitem__('concept:name', activities[0])
                event.__setitem__('time:timestamp',
                                  trace[-1]['time:timestamp'] + datetime.timedelta(seconds=random_duration))
                event.__setitem__('org:resource', 'Artificial Role')

                trace.append(event)
            return trace
        if length < len(trace):
            return Trace(trace[:length], attributes=trace.attributes)

    return build_trace


@pytest.fixture
def rework_trace():
    activities = ['Closed', 'RESOLVED', 'Take in charge ticket', 'Create SW anomaly', 'Require upgrade',
                  'Resolve SW anomaly', 'INVALID', 'DUPLICATE', 'Wait', 'Resolve ticket', 'VERIFIED',
                  'Schedule intervention', 'Insert ticket', 'Assign seriousness']

    def build_trace(length: int = 2) -> Trace:
        assert length is None or length >= 2
        trace = Trace()
        time = datetime.datetime.now()
        rework_pos = random.randint(1, length-1)
        while length > len(trace):
            random.shuffle(activities)
            random_duration = random.uniform(0, 172_800)
            event = Event()
            event.__setitem__('concept:name', activities[0])
            event.__setitem__('time:timestamp', time)
            trace.append(event)

            if rework_pos == len(trace):
                time = trace[-1]['time:timestamp'] + datetime.timedelta(seconds=random_duration)
                event = Event()
                event.__setitem__('concept:name', activities[0])
                event.__setitem__('time:timestamp', time)
                trace.append(event)

            time = trace[-1]['time:timestamp'] + datetime.timedelta(seconds=random_duration)

        return trace

    return build_trace


@pytest.fixture
def trace_with_parallel():
    activities = ['Closed', 'RESOLVED', 'Take in charge ticket', 'Create SW anomaly', 'Require upgrade',
                  'Resolve SW anomaly', 'INVALID', 'DUPLICATE', 'Wait', 'Resolve ticket', 'VERIFIED',
                  'Schedule intervention', 'Insert ticket', 'Assign seriousness']

    def build_trace(length: int = 2) -> Trace:
        assert length is None or length >= 2
        trace = Trace()
        time = datetime.datetime.now()
        first = True
        while length > len(trace):
            random.shuffle(activities)

            event = Event()
            event.__setitem__('concept:name', activities[0])
            event.__setitem__('time:timestamp', time)
            trace.append(event)

            if first is False:
                random_duration = random.uniform(0, 172_800)
                time = trace[-1]['time:timestamp'] + datetime.timedelta(seconds=random_duration)

            first = False
        return trace

    return build_trace


def print_trace(trace: Trace) -> str:
    str_repr = '<'
    for i, event in enumerate(trace):
        if i == len(trace) - 1:
            str_repr = str_repr + f'{event["concept:name"]}'
        else:
            str_repr = str_repr + f'{event["concept:name"]} | '
    return str_repr + '>'


@pytest.fixture
def approach():
    def build_approach(name: str = None, env_name: str = None, dir: str = None,
                       hyperparameter: typing.Dict[str, typing.Union[str, int, float]] = None) -> model.Approach:
        if name is None:
            name = 'Test Approach'
        if env_name is None:
            env_name = 'test_env'
        if dir is None:
            dir = 'tests/persistence/res'
        if hyperparameter is None:
            hyperparameter = {
                'task': 'next_activity',
                'epochs': 1,  # 10
                'batch_size': 12,
                'learning_rate': 0.001,
                'gpu': 0,
            }
        return model.Approach(name, env_name, dir, hyperparameter)

    return build_approach


@pytest.fixture
def dataset():
    def build_dataset(name: str = None, file_path: str = None) -> model.Dataset:
        if name is None:
            name = 'Sample Log'
        if file_path is None:
            file_path = 'tests/persistence/res/sample_log.xes'
        return model.Dataset(name, file_path)

    return build_dataset


@pytest.fixture
def evaluation_model():
    def build_evaluation_model(num_of_runs: int = 1, num_repetitions: int = 1,
                               num_folds: int = 1) -> ev_model.EvaluationModel:
        runs: typing.List[ev_model.Run] = []
        for i in range(num_of_runs):
            repetitions = []
            for j in range(num_repetitions):
                folds = []
                for k in range(num_folds):
                    folds.append(ev_model.Fold(f'fold_{k}', random_string(10), random_string(7), random_string(8),
                                               random_string(4)))
                repetitions.append(ev_model.Repetition(f'rep_{j}', folds))
            runs.append(ev_model.Run(random_string(10), random_string(15), random_string(12), repetitions))

        return ev_model.EvaluationModel(runs)

    return build_evaluation_model


@pytest.fixture
def splitter_configuration():
    def build_splitter_configuration(name: str = None, training_size: float = None, by: str = None,
                                     seeds: typing.List[int] = None, repetitions: int = None,
                                     folds: int = None) -> model.SplitterConfiguration:
        if name is None:
            name = 'time'
        if training_size is None:
            training_size = 0.7
        if by is None:
            by = 'first'
        if seeds is None:
            seeds = [42]
        if repetitions is None:
            repetitions = 1
        if folds is None:
            folds = 1
        return model.SplitterConfiguration(name, training_size, by, seeds, repetitions, folds)

    return build_splitter_configuration


@pytest.fixture
def experiment(dataset, approach, splitter_configuration):
    def build_experiment(name: str = None, data_dir: str = None, run_dir: str = None, evaluation_dir: str = None,
                         event_logs: typing.List[model.Dataset] = None, approaches: typing.List[model.Approach] = None,
                         splitter_config: model.SplitterConfiguration = None,
                         min_pref_length: int = None) -> model.Experiment:
        if name is None:
            name = 'test_experiment'
        if data_dir is None:
            data_dir = '../data/'
        if run_dir is None:
            run_dir = '../run/'
        if evaluation_dir is None:
            evaluation_dir = '../eval/'
        if event_logs is None:
            event_logs = [dataset()]
        if approaches is None:
            approaches = [approach()]
        if splitter_config is None:
            splitter_config = splitter_configuration()
        if min_pref_length is None:
            min_pref_length = 2
        return model.Experiment(name, data_dir, run_dir, evaluation_dir, event_logs, approaches, splitter_config,
                                min_pref_length)

    return build_experiment


@pytest.fixture
def augmentation_strategy_config():
    def build_augmentation_strategy_config(id: int = None, name: str = None, seed: int = None,
                                           augmentor_names: typing.List[str] = None, augmentation_factor: float = None,
                                           allow_multiple: bool = None
                                           ) -> augmentation_strategy.AugmentationStrategyConfig:
        if id is None:
            id = 0
        if name is None:
            name = 'mixed'
        if seed is None:
            seed = 42
        if augmentor_names is None:
            augmentor_names = ['RandomInsertion', 'ParallelSwap']
        if augmentation_factor is None:
            augmentation_factor = 1.2
        if allow_multiple is None:
            allow_multiple = True
        return augmentation_strategy.AugmentationStrategyConfig(id, name, seed, augmentor_names, augmentation_factor,
                                                                allow_multiple)

    return build_augmentation_strategy_config


@pytest.fixture
def augmentation_experiment(dataset, approach, splitter_configuration, augmentation_strategy_config):
    def build_augmentation_experiment(name: str = None, data_dir: str = None, run_dir: str = None,
                                      evaluation_dir: str = None, event_logs: typing.List[model.Dataset] = None,
                                      approaches: typing.List[model.Approach] = None,
                                      splitter_config: model.SplitterConfiguration = None, min_pref_length: int = None,
                                      aug_strat: typing.List[augmentation_strategy.AugmentationStrategyConfig] = None
                                      ) -> model.AugmentationExperiment:
        if name is None:
            name = 'test_augmentation_experiment'
        if data_dir is None:
            data_dir = '../data/'
        if run_dir is None:
            run_dir = '../run/'
        if evaluation_dir is None:
            evaluation_dir = '../eval/'
        if event_logs is None:
            event_logs = [dataset()]
        if approaches is None:
            approaches = [approach()]
        if splitter_config is None:
            splitter_config = splitter_configuration()
        if min_pref_length is None:
            min_pref_length = 2
        if aug_strat is None:
            aug_strat = [augmentation_strategy_config()]

        return model.AugmentationExperiment(name, data_dir, run_dir, evaluation_dir, event_logs, approaches,
                                            splitter_config, min_pref_length, aug_strat)

    return build_augmentation_experiment


@pytest.fixture
def job(approach):
    def build_job(appr: model.Approach = None, dataset_name: str = None, iteration: int = None, fold: int = None,
                  path_training_file: str = None, path_test_file: str = None, path_complete_log: str = None,
                  job_directory: str = None, metadata: typing.Dict[str, typing.Any] = None) -> model.Job:
        if appr is None:
            appr = approach()
        if dataset_name is None:
            dataset_name = 'Test Log'
        if iteration is None:
            iteration = 1
        if fold is None:
            fold = 1
        if path_training_file is None:
            path_training_file = ''
        if path_test_file is None:
            path_test_file = ''
        if path_complete_log is None:
            path_complete_log = ''
        if job_directory is None:
            job_directory = ''
        if metadata is None:
            metadata = {}
        return model.Job(appr, dataset_name, iteration, fold, path_training_file, path_test_file, path_complete_log,
                         job_directory, metadata)

    return build_job


@pytest.fixture
def sample_result():
    def build_sample_result(id: str = None, pred_next_act: typing.Optional[any] = None,
                            pred_next_role: typing.Optional[any] = None,
                            pred_next_time: typing.Optional[any] = None,
                            pred_suffix: typing.Optional[typing.List[any]] = None,
                            pred_remaining_time: typing.Optional[any] = None) -> model.SampleResult:
        if id is None:
            id = 1
        if pred_next_act is None:
            pred_next_act = 'Activity A'
        if pred_next_role is None:
            pred_next_role = 'Role 1'
        if pred_next_time is None:
            pred_next_time = 23.43
        if pred_suffix is None:
            pred_suffix = ['Activity A', 'Activity B']
        if pred_remaining_time is None:
            pred_remaining_time = 3456.21
        return model.SampleResult(id, pred_next_act, pred_next_role, pred_next_time, pred_suffix, pred_remaining_time)

    return build_sample_result
