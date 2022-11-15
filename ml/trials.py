import random
import typing

import pm4py
from pm4py.objects.log.obj import Trace, Event
from ml.augmentation import easy_augmentors
from ml.analysis import statistical_analysis


def trace_to_string(trace: Trace) -> str:
    str_repr = '<'
    for i, event in enumerate(trace):
        if i == len(trace) - 1:
            str_repr = str_repr + f'{event["concept:name"]}, {event["time:timestamp"]}'
        else:
            str_repr = str_repr + f'{event["concept:name"]}, {event["time:timestamp"]} | '
    return str_repr + '>'


def check_order_of_trace(trace) -> bool:
    previous_timestamp = trace[0]['time:timestamp']
    for event in trace[1:]:
        if event['time:timestamp'] < previous_timestamp:
            return False
        previous_timestamp = event['time:timestamp']
    return True

log = pm4py.read_xes('../data/BPI_Challenge_2012.xes')
print(statistical_analysis.Statistic.get_activity_resources(log))

exit(-1)
i = 0
j = 0
while i < 10:
    try:
        print(i)
        print("hello")
        raise KeyError()
        j = j+1
        print("Hello after key error")
    except KeyError:
        print("Excpetion occurcs")
        continue

print("after ", i, j)
exit(-1)
log = pm4py.read_xes('../data/BPI_Challenge_2012.xes')
#for trace in log:
#    print(trace.attributes['concept:name'], check_order_of_trace(trace))

trace = log[0]
dur = trace[1]['time:timestamp'] - trace[0]['time:timestamp']

print(dur.microseconds)
print(dur.total_seconds())
print(dur.seconds)

print(dur.time())
print(random.uniform(0, dur))

exit(-1)

log = pm4py.read_xes('../data/trace.xes')
trace = log[0]
par_trace = log[1]
rework_trace = log[2]
activitities = ['MÜLL', 'NEU', 'Assign seriousness', 'Close', 'Resolve Ticket', 'Take in charge ticket']
insert_augmentor = easy_augmentors.RandomInsertion(activitities)
random_deletion_augmentor = easy_augmentors.RandomDeletion()
parallel_swap_augmentor = easy_augmentors.ParallelSwap()
fragment_augmentor = easy_augmentors.FragmentAugmentation()
rework_augmentor = easy_augmentors.ReworkActivity()
delete_rework_augmentors = easy_augmentors.DeleteReworkActivity()
random_replacement_augmentor = easy_augmentors.RandomReplacement(activitities)
random_swap_augmentor = easy_augmentors.RandomSwap()
loop_augmentor = easy_augmentors.LoopAugmentation(3, 0.2)

insert_aug_trace = insert_augmentor.augment(trace)
random_deletion_aug_trace = random_deletion_augmentor.augment(trace)
parallel_swap_aug_trace = parallel_swap_augmentor.augment(par_trace)
fragment_aug_trace = fragment_augmentor.augment(trace)
rework_aug_trace = rework_augmentor.augment(trace)
delete_rework_aug_trace = delete_rework_augmentors.augment(rework_trace)
random_replacement_aug_trace = random_replacement_augmentor.augment(trace)
random_swap_aug_trace = random_swap_augmentor.augment(trace)

for i in range(1):
    print("hgshshshshsh")
loop_trace = log[5]
print(trace_to_string(loop_trace))

loop_aug_trace = loop_augmentor.augment(loop_trace)

print(trace_to_string(loop_aug_trace))
print(check_order_of_trace(loop_aug_trace))

exit(-1)

import math
import dataclasses


@dataclasses.dataclass
class Loop:
    start: int
    end: int
    body: str

    @property
    def repetitions(self) -> int:
        return int((self.end + 1 - self.start) / len(self.body))

    @property
    def body_length(self) -> int:
        return len(self.body)


def get_all_substrings_of_max_length(trace: str, max_substring_length: int) -> typing.Set[str]:
    substrings = []
    for i in range(len(trace)):
        for k in range(0, max_substring_length):
            substrings.append(trace[i: i + k + 1])

    return set(substrings)


def detect_loops(trace: str) -> typing.List[Loop]:
    max_substring_length = math.floor(len(trace) / 2)
    substrings = get_all_substrings_of_max_length(trace, max_substring_length)

    loops = []
    for substr in substrings:
        for i in range(0, len(trace)):
            repetitions = 0
            if trace[i:i+len(substr)] == substr:
                # Possible loop starts
                start = i
                end = i + len(substr)
                repetitions = repetitions + 1

                j = i + len(substr)
                while True:
                    if trace[j : j + len(substr)] == substr:
                        repetitions = repetitions + 1
                        j = j + len(substr)
                        end = j
                    else:
                        if repetitions > 1:
                            loops.append(Loop(start, end - 1, substr))
                        break
    return loops


def augment(trace: str, max_additional_repetitions: int) -> str:
    loops = detect_loops(trace)
    assert len(loops) > 0, f'Could not apply augmentor. There are no loops within the trace.'

    # Select loop for augmentation
    random.shuffle(loops)
    loop_to_augment = loops[0]
    additional_repetitions = random.randint(1, max_additional_repetitions)

    print(additional_repetitions)
    print(loop_to_augment)

    insertion = ''
    for i in range(0, additional_repetitions):
        insertion = insertion + loop_to_augment.body

    return trace[:loop_to_augment.end + 1] + " " + insertion + " " + trace[loop_to_augment.end + 1:]



exit(-1)
trace = "abababbc"
trace1 = "abcd"
print(trace)
#print(detect_loops(trace))

print(augment(trace, 3))


random_duration_repetition = 10.4
points = 4

lower_bound = 0
upper_bound = random_duration_repetition
elapsed_time = []
for i in range(points):
    value = random.uniform(0, random_duration_repetition)
    elapsed_time.append(value)

elapsed_time.sort()
print(elapsed_time)
durations = [elapsed_time[i+1]-elapsed_time[i] for i in range(len(elapsed_time)-1)]
print(durations)

assert sum(durations) <= random_duration_repetition, f'Longer as allowed'




