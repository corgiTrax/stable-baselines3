import os
from collections import defaultdict

import numpy as np
from responses import target
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath, target_len):

    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]
    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = defaultdict(list)
    print(tags)
    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            if tag == 'rollout/ep_rew_mean':
                print(np.mean([e.step for e in events]))
                print([e.step for e in events])
                if np.mean([e.step for e in events]) < target_len:
                    out[tag].append([e.value for e in events])
                    steps[tag].append(np.mean([e.step for e in events]))

    return out, steps


def write_combined_events(dpath, d_combined, steps, dname='combined'):

    fpath = os.path.join(dpath, dname)
    writer = tf.summary.create_file_writer(fpath)

    tags, values = zip(*d_combined.items())

    timestep_mean = np.array(values).mean(axis=-1)

    with writer.as_default():
        for tag, means in zip(tags, timestep_mean):
            for i, mean in enumerate(means):
                tf.summary.scalar(tag, mean, step=steps[tag][i])

dpath = 'final_results/TamerRL30Random'

data, steps = tabulate_events(dpath, 150000)
write_combined_events(dpath, data, steps, dname='averaged')