import gym
import numpy as np
import os
import sys
import random
import cv2

def resize_frame(frame):
    frame = frame[30:-12,5:-4]
    frame = np.average(frame,axis = 2)
    frame = cv2.resize(frame,(84,84),interpolation = cv2.INTER_NEAREST)
    frame = np.array(frame,dtype = np.uint8)
    return frame

class Game():
    def __init__(self, game_name, agent, verbose=False, filePath="./video_capture/", ):
        self.name = game_name
        self.env = gym.make(self.name, render_mode='rgb_array')
        self.env.reset()
        self.verbose = verbose
        self.filePath = filePath
        self.filterdEpisodeFrames = []
        self.agent = agent
        pass

    def reset(self):
        self.filterdEpisodeFrames = []
        self.env.reset()

    def step(self,showRender=False):
        
        action = self.agent.get_action()
        ## see documentation about what env.step returns 
        ## https://gym.openai.com/docs/#observations
        observation, reward, done, info = self.env.step(action)
        self.filterdEpisodeFrames.append(resize_frame(observation))
        if self.verbose:
            # print("action: {}\tobvservation: {}\treward: {}\tdone: {}\tinfo: {}"\
            #     .format(action, observation, reward, done, info))
            print("reward: {}"\
                .format(reward))
        if showRender:
            self.env.render()
        return reward, done

    def play(self,showRender=False,recordRender=False,recordFilteredRender=False):
        if recordRender:
            self.env = gym.wrappers.Monitor(self.env,self.filePath,force=True)
        self.reset()
        scoreTotal = 0
        while True:
            score,done = self.step(showRender)
            scoreTotal += score
            if done:
                if recordFilteredRender:
                    out = cv2.VideoWriter(self.filePath+'episode.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, (84,84), False)
                    for frame in self.filterdEpisodeFrames:
                        out.write(frame)
                    out.release()
                break
        return scoreTotal


if __name__ == '__main__':
    try:
       game = Game('PongDeterministic-v4',verbose=True,filePath="./video_capture/")
    #    game.step(showRender=True)
       game.play(showRender=True,recordRender=False, recordFilteredRender=False)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)