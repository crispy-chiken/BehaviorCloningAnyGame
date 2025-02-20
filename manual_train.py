from __future__ import print_function

import gzip
import os
import pickle
import time
import inputs
from inputs import INPUTS
from pyglet.window import key
from environment import Game

import gymnasium as gym
import numpy as np

DATA_DIR = 'data'
DATA_FILE = 'data.gzip'

def rollout(env:gym.Env):
    global restart_train, agent_action, exit_train, pause_train
    agent_action = np.zeros(len(INPUTS))
    exit_train = False
    pause_train = False
    restart_train = False

    # if the file exists, append
    if os.path.exists(os.path.join(DATA_DIR, DATA_FILE)):
        with gzip.open(os.path.join(DATA_DIR, DATA_FILE), 'rb') as f:
            observations = pickle.load(f)
    else:
        observations = list()

    # Run key reading tool
    inputs.main()
    state = env.reset()
    total_reward = 0
    episode = 1
    while 1:
        env.render()

        agent_action = inputs.actions
        exit_train = inputs.stop_action
        
        a = np.copy(agent_action)
        old_state = state

        state, reward, done, info = env.step(agent_action)

        observations.append((old_state, a, state, reward, done))

        total_reward += reward

        # if exit_train:
        #     env.close()
        #     return

        if restart_train:
            restart_train = False
            state = env.reset()
            continue

        if done:
                
            if episode == 1:
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

            state = env.reset()

        while pause_train:
            env.render()
            time.sleep(0.1)


if __name__ == '__main__':
    env = Game()
    env.reset()
    env.render()
    rollout(env)
