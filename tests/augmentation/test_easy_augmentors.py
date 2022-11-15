from ml.augmentation import easy_augmentors


def check_order_of_trace(trace) -> bool:
    previous_timestamp = trace[0]['time:timestamp']
    for event in trace[1:]:
        if event['time:timestamp'] < previous_timestamp:
            return False
        previous_timestamp = event['time:timestamp']
    return True


def check_activities(trace, activities) -> bool:
    for event in trace:
        if event['concept:name'] not in activities:
            return False
    return True


class TestRandomInsertion:
    activities = ['Closed', 'RESOLVED', 'Take in charge ticket', 'Create SW anomaly', 'Require upgrade',
                  'Resolve SW anomaly', 'INVALID', 'DUPLICATE', 'Wait', 'Resolve ticket', 'VERIFIED',
                  'Schedule intervention', 'Insert ticket', 'Assign seriousness']

    def test_augment(self, trace):
        trace = trace(length=2)
        augmentor = easy_augmentors.RandomInsertion(self.activities)
        augmented_trace = augmentor.augment(trace)

        assert len(augmented_trace) == len(trace) + 1
        assert augmented_trace[0] == trace[0]
        assert augmented_trace[-1] == trace[-1]
        assert check_order_of_trace(augmented_trace) is True
        assert check_activities(trace, self.activities) is True

    def test_is_applicable(self, trace):
        augmentor = easy_augmentors.RandomInsertion(self.activities)
        assert augmentor.is_applicable('', trace()) is True

    def test_is_applicable_failure_case(self, trace):
        augmentor = easy_augmentors.RandomInsertion(self.activities)
        assert augmentor.is_applicable('', trace(length=1)) is False

    def test_get_name(self):
        assert easy_augmentors.RandomInsertion.get_name() == 'RandomInsertion'


class TestRandomDeletion:
    def test_augment(self, trace):
        trace = trace(length=4)
        augmentor = easy_augmentors.RandomDeletion()
        augmented_trace = augmentor.augment(trace)

        assert len(augmented_trace) == 3
        assert check_order_of_trace(augmented_trace) is True

    def test_is_applicable(self, trace):
        assert easy_augmentors.RandomDeletion().is_applicable('', trace()) is True
        assert easy_augmentors.RandomDeletion().is_applicable('', trace(length=1)) is False

    def test_get_name(self):
        assert easy_augmentors.RandomDeletion.get_name() == 'RandomDeletion'


class TestParallelSwap:

    def test_augment(self, trace_with_parallel):
        trace_1 = trace_with_parallel(length=2)
        trace_2 = trace_with_parallel(length=3)

        augmentor = easy_augmentors.ParallelSwap()
        augmented_trace_1 = augmentor.augment(trace_1)
        augmented_trace_2 = augmentor.augment(trace_2)

        assert trace_1[0]['concept:name'] == augmented_trace_1[1]['concept:name'] and \
               trace_1[1]['concept:name'] == augmented_trace_1[0]['concept:name']
        assert len(trace_1) == len(augmented_trace_1)
        assert check_order_of_trace(augmented_trace_1) is True

        assert trace_2[0]['concept:name'] == augmented_trace_2[1]['concept:name'] and \
               trace_2[1]['concept:name'] == augmented_trace_2[0]['concept:name']
        assert len(trace_2) == len(augmented_trace_2)
        assert trace_2[2] == augmented_trace_2[2]
        assert check_order_of_trace(augmented_trace_2) is True

    def test_extract_parallel_activities(self, trace, trace_with_parallel):
        trace = trace(length=10)
        trace_with_parallel = trace_with_parallel(length=5)
        assert easy_augmentors.ParallelSwap().extract_parallel_activities(trace) == []
        assert easy_augmentors.ParallelSwap().extract_parallel_activities(trace_with_parallel) == [(1,0)]

    def test_is_applicable(self, trace, trace_with_parallel):
        assert easy_augmentors.ParallelSwap().is_applicable('', trace()) is False
        assert easy_augmentors.ParallelSwap().is_applicable('', trace_with_parallel()) is True

    def test_get_name(self):
        assert easy_augmentors.ParallelSwap.get_name() == 'ParallelSwap'


class TestFragmentAugmentation:

    def test_augment(self, trace):
        trace = trace(length=4)
        augmented_trace = easy_augmentors.FragmentAugmentation().augment(trace)

        assert check_order_of_trace(augmented_trace) is True
        assert len(augmented_trace) < len(trace)

    def test_is_applicable(self, trace):
        assert easy_augmentors.FragmentAugmentation().is_applicable('', trace(length=4)) is True
        assert easy_augmentors.FragmentAugmentation().is_applicable('', trace(length=1)) is False

    def test_get_name(self):
        assert easy_augmentors.FragmentAugmentation.get_name() == 'FragmentAugmentation'


class TestRandomSwapAugmentation:
    def test_augment(self, trace):
        trace = trace(length=4)
        augmented_trace = easy_augmentors.RandomSwap().augment(trace)
        assert check_order_of_trace(augmented_trace) is True
        assert len(augmented_trace) == len(trace)
        assert ['Assign seriousness', 'Take in charge ticket', 'Closed', 'Resolve ticket'] == \
               [k['concept:name'] for k in augmented_trace]

    def test_is_applicable(self, trace):
        assert easy_augmentors.RandomSwap().is_applicable('', trace(length=3)) is False
        assert easy_augmentors.RandomSwap().is_applicable('', trace(length=4)) is True

    def test_get_name(self):
        assert easy_augmentors.RandomSwap().get_name() == 'RandomSwap'


class TestReworkActivity:
    def test_augment(self, trace):
        trace = trace(length=4)
        augmented_trace = easy_augmentors.ReworkActivity().augment(trace)

        assert check_order_of_trace(augmented_trace) is True
        assert len(augmented_trace) == len(trace) + 1
        assert ['Assign seriousness', 'Assign seriousness', 'Take in charge ticket', 'Resolve ticket', 'Closed'] == \
               [k['concept:name'] for k in augmented_trace]

    def test_is_applicable(self, trace):
        assert easy_augmentors.ReworkActivity().is_applicable('', trace(length=2)) is True
        assert easy_augmentors.ReworkActivity().is_applicable('', trace(length=0)) is False

    def test_get_name(self):
        assert easy_augmentors.ReworkActivity.get_name() == 'ReworkActivity'


class TestDeleteReworkActivity:
    def test_augment(self, rework_trace):
        trace = rework_trace(length=4)
        augmented_trace = easy_augmentors.DeleteReworkActivity().augment(trace)

        assert len(augmented_trace) == len(trace)-1
        assert check_order_of_trace(augmented_trace) is True
        assert ['Create SW anomaly', 'Resolve SW anomaly', 'Insert ticket'] == \
               [k['concept:name'] for k in augmented_trace]

    def test_is_applicable(self, trace, rework_trace):
        assert easy_augmentors.DeleteReworkActivity().is_applicable('', trace()) is False
        assert easy_augmentors.DeleteReworkActivity().is_applicable('', rework_trace()) is True

    def test_get_name(self):
        assert easy_augmentors.DeleteReworkActivity.get_name() == 'DeleteReworkActivity'


class TestRandomReplacement:
    activities = ['Closed', 'RESOLVED', 'Take in charge ticket', 'Create SW anomaly', 'Require upgrade',
                  'Resolve SW anomaly', 'INVALID', 'DUPLICATE', 'Wait', 'Resolve ticket', 'VERIFIED',
                  'Schedule intervention', 'Insert ticket', 'Assign seriousness']

    def test_augment(self, trace):
        trace = trace(length=4)
        augmented_trace = easy_augmentors.RandomReplacement(self.activities).augment(trace)

        assert len(trace) == len(augmented_trace)
        assert check_order_of_trace(augmented_trace)
        assert ['Assign seriousness', 'DUPLICATE', 'Resolve ticket', 'Closed'] == \
               [k['concept:name'] for k in augmented_trace]

    def test_is_applicable(self, trace):
        assert easy_augmentors.RandomReplacement(self.activities).is_applicable('', trace(length=3)) is True
        assert easy_augmentors.RandomReplacement(self.activities).is_applicable('', trace(length=0)) is False

    def test_get_name(self):
        assert easy_augmentors.RandomReplacement.get_name() == 'RandomReplacement'
