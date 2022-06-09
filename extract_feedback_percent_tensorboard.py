import os
from collections import defaultdict
from matplotlib.pyplot import close

import numpy as np
from responses import target
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import long
import csv

def average_from_per_timestep_rewards(dpath, target_len):

    summary_iterators = [EventAccumulator(os.path.join(dpath)).Reload()]
    tags = summary_iterators[0].Tags()['scalars']
    
    # tensor_data = tf.make_ndarray(summary_iterators[0].Tensors('train/training_rewards_cumulative').summary.value[0].tensor)
    # print(tensor_data)

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = defaultdict(list)
    data = defaultdict(list)
    tag = 'train/training_rewards'
    target_tag = 'train/training_rewards_cumulative'
    
    data[target_tag] = [(0, 0, 0)]
    for index, curr_iterator in enumerate(summary_iterators):        
        for iter_index, i in enumerate(curr_iterator.Scalars(tag)):
            if i.step < target_len:
                data[target_tag].append((i.step, i.value + data[target_tag][-1][1], iter_index + 1))
    
    for index, i in enumerate(summary_iterators[0].Scalars(tag)):
        if i.step < target_len:
            out[target_tag].append(data[target_tag][index])
            steps[target_tag].append(i.step)
    
    return out, steps

def average_from_per_episode_rewards(dpath, target_len):

    summary_iterators = [EventAccumulator(os.path.join(dpath)).Reload()]
    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = defaultdict(list)
    data = defaultdict(list)
    tag = 'rollout/ep_rew_mean'
    target_tag = 'rollout/training_rewards_cumulative'
    
    data[target_tag] = [(0, 0, 0)]
    rewards_buffer = [0]
    steps_buffer = [0]
    for index, curr_iterator in enumerate(summary_iterators):        
        for iter_index, i in enumerate(curr_iterator.Scalars(tag)):
            if i.step < target_len:
                curr_rewards = i.value * len(rewards_buffer)
                curr_steps = i.step * len(steps_buffer)
                
                if len(rewards_buffer) == 25:
                    rewards_buffer.pop(0)
                    steps_buffer.pop(0)
                
                curr_rewards -= sum(rewards_buffer)
                curr_steps -= sum(steps_buffer)
                data[target_tag].append((curr_steps + data[target_tag][-1][0], curr_rewards + data[target_tag][-1][1], iter_index + 1))

                rewards_buffer.append(curr_rewards)
                steps_buffer.append(curr_steps)
    
    for index, i in enumerate(summary_iterators[0].Scalars(tag)):
        if i.step < target_len:
            out[target_tag].append(data[target_tag][index])
            steps[target_tag].append(i.step)
    
    return out, steps

def get_feedback_percent(dpath, target_len):

    summary_iterators = [EventAccumulator(os.path.join(dpath)).Reload()]
    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = defaultdict(list)
    data = defaultdict(list)
    tag = 'train/ep_rew_mean'
    target_tag = 'rollout/training_rewards_cumulative'
    
    data[target_tag] = [(0, 0, 0)]
    rewards_buffer = [0]
    steps_buffer = [0]
    for index, curr_iterator in enumerate(summary_iterators):        
        for iter_index, i in enumerate(curr_iterator.Scalars(tag)):
            if i.step < target_len:
                curr_rewards = i.value * len(rewards_buffer)
                curr_steps = i.step * len(steps_buffer)
                
                if len(rewards_buffer) == 25:
                    rewards_buffer.pop(0)
                    steps_buffer.pop(0)
                
                curr_rewards -= sum(rewards_buffer)
                curr_steps -= sum(steps_buffer)
                data[target_tag].append((curr_steps + data[target_tag][-1][0], curr_rewards + data[target_tag][-1][1], iter_index + 1))

                rewards_buffer.append(curr_rewards)
                steps_buffer.append(curr_steps)
    
    for index, i in enumerate(summary_iterators[0].Scalars(tag)):
        if i.step < target_len:
            out[target_tag].append(data[target_tag][index])
            steps[target_tag].append(i.step)
    
    return out, steps

def update_existing_tensorboard(dpath, tag, steps, data):
    writer = tf.summary.create_file_writer(os.path.join(dpath))
    with writer.as_default():
        for i, curr_val in enumerate(data[tag]):
            tf.summary.scalar(tag, curr_val[1], step=steps[tag][i])

def write_to_csv(data, steps, csv_name):
    fieldnames = ["Step", "Value"]
    rows = [{"Step": steps[i], "Value": curr_val[1]} for i, curr_val in enumerate(data)]
    with open(csv_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        

dpath = 'ballbasket_cumulative/RPEActiveTamer'

for dname in os.listdir(dpath):
    # data, steps = average_from_per_timestep_rewards(os.path.join(dpath, dname), 150000)
    # update_existing_tensorboard(os.path.join(dpath, dname), 'train/training_rewards_cumulative', steps, data)

    data, steps = average_from_per_episode_rewards(os.path.join(dpath, dname), 100000)
    write_to_csv(data['rollout/training_rewards_cumulative'], steps['rollout/training_rewards_cumulative'], os.path.join(dpath, dname+".csv"))
    # update_existing_tensorboard(os.path.join(dpath, dname), 'rollout/training_rewards_cumulative', steps, data)
