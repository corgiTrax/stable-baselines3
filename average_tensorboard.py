import os
from collections import defaultdict
from matplotlib.pyplot import close

import numpy as np
from responses import target
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import long

import sys

def average_tensor_events(dpath, target_len):

    summary_iterators = [EventAccumulator('ballbasket_cumulative/RPEActiveTamer/ActiveTamerRLSACOptim_3').Reload()]
    print(summary_iterators[0].Tags())
    tags = summary_iterators[0].Tags()['tensors']

    for it in summary_iterators:
        assert it.Tags()['tensors'] == tags

    out = defaultdict(list)
    steps = defaultdict(list)
    data = defaultdict(list)
    longest_iterator = [-1, 0]
    
    print(tags)

    for tag in tags:
        data[tag] = []
        for index, curr_iterator in enumerate(summary_iterators):
            # if tag == 'rollout/ep_rew_mean':
            #     data[tag].append([(i.step, 0.1 * i.value, iter_index) for iter_index, i in enumerate(curr_iterator.Scalars(tag)) if i.step < target_len])
            # else:
            # data[tag].append([(i.step, i.value, iter_index) for iter_index, i in enumerate(curr_iterator.Tensors(tag)) if i.step < target_len])
            print(curr_iterator.Tensors(tag))
            for iter_index, i in enumerate(curr_iterator.Tensors(tag)):
                # print(i)
                curr_step = i.step
                curr_data = tf.make_ndarray(i.tensor_proto)
                print(curr_data)
            
            sys.exit(0)

            if longest_iterator[1] < len(data[tag][-1]):
                longest_iterator = [index, len(data[tag][-1])]
    
    for tag in tags:
        for index, i in enumerate(summary_iterators[longest_iterator[0]].Tensors(tag)):
            if i.step < target_len:
                closest_indices = [min(data[tag][iterator],key=lambda x:abs(x[0] - i.step))[2] for iterator in range(len(summary_iterators))]
                curr_values = [data[tag][iterator][closest_indices[iterator]][1] for iterator in range(len(summary_iterators))]
                out[tag].append(curr_values)
                steps[tag].append(i.step)

    return out, steps

def tabulate_events(dpath, target_len):

    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]
    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = defaultdict(list)
    data = defaultdict(list)
    longest_iterator = [-1, 0]

    for tag in tags:
        data[tag] = []
        for index, curr_iterator in enumerate(summary_iterators):
            # if tag == 'rollout/ep_rew_mean':
            #     data[tag].append([(i.step, 0.1 * i.value, iter_index) for iter_index, i in enumerate(curr_iterator.Scalars(tag)) if i.step < target_len])
            # else:
            data[tag].append([(i.step, i.value, iter_index) for iter_index, i in enumerate(curr_iterator.Scalars(tag)) if i.step < target_len])
            if longest_iterator[1] < len(data[tag][-1]):
                longest_iterator = [index, len(data[tag][-1])]
    
    for tag in tags:
        for index, i in enumerate(summary_iterators[longest_iterator[0]].Scalars(tag)):
            if i.step < target_len:
                closest_indices = [min(data[tag][iterator],key=lambda x:abs(x[0] - i.step))[2] for iterator in range(len(summary_iterators))]
                curr_values = [data[tag][iterator][closest_indices[iterator]][1] for iterator in range(len(summary_iterators))]
                out[tag].append(curr_values)
                steps[tag].append(i.step)

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

dpath = 'log_dir/sawyer_sac'

data, steps = tabulate_events(dpath, 100000)

write_combined_events(dpath, data, steps, dname='averaged')