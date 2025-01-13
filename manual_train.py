from __future__ import print_function

import gzip
import os
import pickle
import time
import argparse
import pygame
from pyglet.window import key

import gymnasium as gym
import numpy as np

DATA_DIR = 'data'
DATA_FILE = 'data.gzip'

def register_input():
    global restart_train, agent_action, exit_train, pause_train, acceleration
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                agent_action[0] = -1.0
                agent_action[1] = 0.0   # no acceleration while turning
            if event.key == pygame.K_RIGHT:
                agent_action[0] = +1.0
                agent_action[1] = 0.0   # no acceleration when turning
            if event.key == pygame.K_UP:
                agent_action[1] = 1.0
                agent_action[2] = 0
            if event.key == pygame.K_DOWN:
                agent_action[2] = 1
            if event.key == pygame.K_RETURN:
                restart_train = True
            if event.key == pygame.K_ESCAPE:
                exit_train = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                agent_action[0] = 0
                agent_action[1] = acceleration  # restore acceleration
            if event.key == pygame.K_RIGHT:
                agent_action[0] = 0
                agent_action[1] = acceleration  # restore acceleration
            if event.key == pygame.K_UP:
                acceleration = False
                agent_action[1] = 0.0
            if event.key == pygame.K_DOWN:
                agent_action[2] = 0.0

        if event.type == pygame.QUIT:
            exit_train = True

def rollout(env:gym.Env):
    global restart_train, agent_action, exit_train, pause_train
    ACTIONS = env.action_space.shape[0]
    agent_action = np.zeros(ACTIONS, dtype=np.float32)
    exit_train = False
    pause_train = False
    restart_train = False

    # if the file exists, append
    if os.path.exists(os.path.join(DATA_DIR, DATA_FILE)):
        with gzip.open(os.path.join(DATA_DIR, DATA_FILE), 'rb') as f:
            observations = pickle.load(f)
    else:
        observations = list()

    state = env.reset()[0]
    total_reward = 0
    episode = 1
    while 1:
        env.render()
        register_input()
        a = np.copy(agent_action)
        old_state = state
        if agent_action[2] != 0:
            agent_action[2] = 0.1

        state, reward, done, _, info = env.step(agent_action)

        observations.append((old_state, a, state, reward, done))

        total_reward += reward

        if exit_train:
            env.close()
            return

        if restart_train:
            restart_train = False
            state = env.reset()[0]
            continue

        if done:
                
            if episode == 5:
                # store generated data
                data_file_path = os.path.join(DATA_DIR, DATA_FILE)
                print("Saving observations to " + data_file_path)

                if not os.path.exists(DATA_DIR):
                    os.mkdir(DATA_DIR)

                with gzip.open(data_file_path, 'wb') as f:
                    pickle.dump(observations, f)
                
                env.close()
                return

            print("Episodes %i reward %0.2f" % (episode, total_reward))

            episode += 1

            state = env.reset()[0]

        while pause_train:
            env.render()
            time.sleep(0.1)


if __name__ == '__main__':
    env = gym.make('CarRacing-v3', render_mode='human')
    env.reset()
    env.render()
    # env.unwrapped.viewer.window.on_key_press = key_press
    # env.unwrapped.viewer.window.on_key_release = key_release

    print("ACTIONS={}".format(env.action_space.shape[0]))

    rollout(env)
