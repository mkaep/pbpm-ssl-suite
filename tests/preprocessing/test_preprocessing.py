import random

from pm4py.objects.log.obj import EventLog, Trace
from ml.preprocess import preprocessing


class TestRemoveEmptyTraces:

    def test_run_without_empty_traces(self, trace):
        event_log = EventLog()
        for _ in range(5):
            event_log.append(trace())
        step = preprocessing.RemoveEmptyTraces()
        preprocessed_log = step.run(event_log)
        assert preprocessed_log == event_log
        assert len(preprocessed_log) == len(event_log)

    def test_run_with_empty_event_log(self):
        empty_log = EventLog()
        step = preprocessing.RemoveEmptyTraces()
        assert empty_log == step.run(empty_log)

    def test_run(self, trace):
        event_log = EventLog()
        for _ in range(5):
            event_log.append(trace())
        event_log_with_empty_trace = event_log.__deepcopy__()
        event_log_with_empty_trace.append(Trace())

        step = preprocessing.RemoveEmptyTraces()
        assert len(step.run(event_log_with_empty_trace)) == len(event_log)
        assert step.run(event_log_with_empty_trace) == event_log


class TestRemoveTracesWithMissingAttributes:

    def test_run_without_missing_attributes(self, trace):
        event_log = EventLog()
        for _ in range(5):
            event_log.append(trace(length=random.randint(2, 10)))
        step = preprocessing.RemoveTracesWithMissingAttributes()
        assert step.run(event_log) == event_log

    def test_run_with_empty_event_log(self):
        event_log = EventLog()
        step = preprocessing.RemoveTracesWithMissingAttributes()
        assert step.run(event_log) == event_log

    def test_run(self, trace):
        event_log = EventLog()
        for i in range(5):
            event_log.append(trace(length=random.randint(2, 10), id=f'Case_{i}'))
        for event in event_log[0]:
            del event['org:resource']
        for event in event_log[-1]:
            del event['org:resource']

        step = preprocessing.RemoveTracesWithMissingAttributes()
        preprocessed_log = step.run(event_log)
        expected_event_log = EventLog()
        expected_event_log.append(event_log[1])
        expected_event_log.append(event_log[2])
        expected_event_log.append(event_log[3])

        assert len(preprocessed_log) == len(event_log) - 2
        assert expected_event_log == preprocessed_log
