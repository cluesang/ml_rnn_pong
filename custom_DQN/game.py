import gym
import numpy as np
import os
import sys
import random

class Game():
    def __init__(self, game_name, verbose=False):
        self.name = game_name
        self.env = gym.make(self.name, render_mode='rgb_array')
        self.env.reset()
        self.verbose = verbose
        pass

    def reset(self):
        self.env.reset()

    def step(self,showRender=False,action=None):
        if(action == None):
            action = random.sample([0,2,5],1)[0]
        ## see documentation about what env.step returns 
        ## https://gym.openai.com/docs/#observations
        observation, reward, done, info = self.env.step(action)
        if self.verbose:
            print("action: {}\tobvservation: {}\treward: {}\tdone: {}\tinfo: {}"\
                .format(action, observation, reward, done, info))
            # print("action: {}"\
            #     .format(action))
        if showRender:
            self.env.render()
        
        return reward, done

    def play(self,showRender=False):
        self.reset()
        while True:
            score,done = self.step(showRender)
            if done:
                break
        return score


if __name__ == '__main__':
    try:
       game = Game('PongDeterministic-v4',verbose=True)
    #    game.step(showRender=True)
       game.play(showRender=True)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)