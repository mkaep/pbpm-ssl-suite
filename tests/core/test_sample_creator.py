from ml.core import sample_creator
from pm4py.objects.log.obj import EventLog, Trace
import pm4py
from datetime import datetime


class TestSampleCreator:
    def test_get_name(self):
        assert 'PrefixCreator' == sample_creator.PrefixCreator.get_name()

    def test_prefix_creator(self):
        mock_prefix_creator = sample_creator.PrefixCreator(2)
        event_log = pm4py.read_xes('D:/PBPM_Approaches/experiment/tests/core/res/sample_log_1.xes')
        expected_suffix_log = EventLog()
        expected_prefix_log = EventLog()

        prefix_log, suffix_log = mock_prefix_creator.create_samples(event_log)

        expected_prefix_log.append(Trace([
            {
                'concept:name': 'Assign seriousness',
                'lifecycle:transition': 'complete',
                'org:resource': 'Value 1',
                'time:timestamp': datetime.fromisoformat('2012-10-09T14:50:17.000+03:00'),
                'Activity': 'Assign seriousness',
                'Resource': 'Value 1'
            },
            {
                'concept:name': 'Take in charge ticket',
                'lifecycle:transition': 'complete',
                'org:resource': 'Value 1',
                'time:timestamp': datetime.fromisoformat('2012-10-09T14:51:01.000+03:00'),
                'Activity': 'Take in charge ticket',
                'Resource': 'Value 1'
            }],
            attributes={
                'concept:name': 'Case_0_2',
                'creator': 'PrefixCreator'
        }))
        expected_prefix_log.append(Trace([
            {
                'concept:name': 'Assign seriousness',
                'lifecycle:transition': 'complete',
                'org:resource': 'Value 1',
                'time:timestamp': datetime.fromisoformat('2012-10-09T14:50:17.000+03:00'),
                'Activity': 'Assign seriousness',
                'Resource': 'Value 1'
            },
            {
                'concept:name': 'Take in charge ticket',
                'lifecycle:transition': 'complete',
                'org:resource': 'Value 1',
                'time:timestamp': datetime.fromisoformat('2012-10-09T14:51:01.000+03:00'),
                'Activity': 'Take in charge ticket',
                'Resource': 'Value 1'
            },
            {
                'concept:name': 'Take in charge ticket',
                'lifecycle:transition': 'complete',
                'org:resource': 'Value 2',
                'time:timestamp': datetime.fromisoformat('2012-10-12T15:02:56.000+03:00'),
                'Activity': 'Take in charge ticket',
                'Resource': 'Value 2'
            }], attributes={
            'concept:name': 'Case_0_3',
            'creator': 'PrefixCreator'
        }))
        expected_prefix_log.append(Trace([
            {
                'concept:name': 'Assign seriousness',
                'lifecycle:transition': 'complete',
                'org:resource': 'Value 1',
                'time:timestamp': datetime.fromisoformat('2012-10-09T14:50:17.000+03:00'),
                'Activity': 'Assign seriousness',
                'Resource': 'Value 1'
            },
            {
                'concept:name': 'Take in charge ticket',
                'lifecycle:transition': 'complete',
                'org:resource': 'Value 1',
                'time:timestamp': datetime.fromisoformat('2012-10-09T14:51:01.000+03:00'),
                'Activity': 'Take in charge ticket',
                'Resource': 'Value 1'
            },
            {
                'concept:name': 'Take in charge ticket',
                'lifecycle:transition': 'complete',
                'org:resource': 'Value 2',
                'time:timestamp': datetime.fromisoformat('2012-10-12T15:02:56.000+03:00'),
                'Activity': 'Take in charge ticket',
                'Resource': 'Value 2'
            },
            {
                'concept:name': 'Resolve ticket',
                'lifecycle:transition': 'complete',
                'org:resource': 'Value 1',
                'time:timestamp': datetime.fromisoformat('2012-10-25T11:54:26.000+03:00'),
                'Activity': 'Resolve ticket',
                'Resource': 'Value 1'
            }], attributes={
            'concept:name': 'Case_0_4',
            'creator': 'PrefixCreator'
        }))

        expected_suffix_log.append(Trace([
            {
                'concept:name': 'Take in charge ticket',
                'lifecycle:transition': 'complete',
                'org:resource': 'Value 2',
                'time:timestamp': datetime.fromisoformat('2012-10-12T15:02:56.000+03:00'),
                'Activity': 'Take in charge ticket',
                'Resource': 'Value 2'
            },
            {
                'concept:name': 'Resolve ticket',
                'lifecycle:transition': 'complete',
                'org:resource': 'Value 1',
                'time:timestamp': datetime.fromisoformat('2012-10-25T11:54:26.000+03:00'),
                'Activity': 'Resolve ticket',
                'Resource': 'Value 1'
            },
            {
                'concept:name': 'Closed',
                'lifecycle:transition': 'complete',
                'org:resource': 'Value 3',
                'time:timestamp': datetime.fromisoformat('2012-11-09T12:54:39.000+02:00'),
                'Activity': 'Closed',
                'Resource': 'Value 3'
            }], attributes={
            'concept:name': 'Case_0_2',
            'creator': 'PrefixCreator'
        }))
        expected_suffix_log.append(Trace([
            {
                'concept:name': 'Resolve ticket',
                'lifecycle:transition': 'complete',
                'org:resource': 'Value 1',
                'time:timestamp': datetime.fromisoformat('2012-10-25T11:54:26.000+03:00'),
                'Activity': 'Resolve ticket',
                'Resource': 'Value 1'
            },
            {
                'concept:name': 'Closed',
                'lifecycle:transition': 'complete',
                'org:resource': 'Value 3',
                'time:timestamp': datetime.fromisoformat('2012-11-09T12:54:39.000+02:00'),
                'Activity': 'Closed',
                'Resource': 'Value 3'
            }], attributes={
            'concept:name': 'Case_0_3',
            'creator': 'PrefixCreator'
        }))
        expected_suffix_log.append(Trace([
            {
                'concept:name': 'Closed',
                'lifecycle:transition': 'complete',
                'org:resource': 'Value 3',
                'time:timestamp': datetime.fromisoformat('2012-11-09T12:54:39.000+02:00'),
                'Activity': 'Closed',
                'Resource': 'Value 3'
            }], attributes={
            'concept:name': 'Case_0_4',
            'creator': 'PrefixCreator'
        }))

        assert expected_prefix_log == prefix_log
        assert expected_suffix_log == suffix_log
