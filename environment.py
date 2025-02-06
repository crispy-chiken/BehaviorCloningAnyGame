from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from matplotlib import pyplot as plt
import inputs
from inputs import INPUTS
import grab_current_screen as grab
import pynput
import time
import cv2
import numpy as np


class Game(Env):
    def __init__(self, training = False):
        super().__init__()
        # Setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1,90,160), dtype=np.uint8)
        self.action_space = Discrete(len(INPUTS))
        self.training = training
        
    def step(self, action):
        # action[action >= 0.9] = 1
        # action[action < 0.9] = 0
        action = np.round(action, 2)

        if inputs.toggle:
            if not self.training:
                for i in range(len(action)):
                    if action[i] > 0.5:
                        inputs.input_board.press(INPUTS[i])
                    else:
                        inputs.input_board.release(INPUTS[i])
        print(str(action) + " " + str(inputs.toggle))
        done = inputs.stop_action
        #print(done)
        observation = self.get_observation()
        reward = 1 
        info = {}
        return observation, reward, done, info
        
    
    def reset(self):
        time.sleep(1)
        return self.get_observation()
        
    def render(self):
        obs = self.get_observation()
        #plt.imshow(cv2.cvtColor(obs, cv2.COLOR_GRAY2BGR))
        # plt.imshow(obs)
        # plt.show(block = False)
        # plt.pause(0.001)
        # plt.show()
         
    def close(self):
        pass
    
    def get_observation(self):
        img = grab.grab_screen()
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (160,90))
        #img = np.reshape(img, (3,90,160))
        return img