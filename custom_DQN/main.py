from game import Game
from agent import Agent
import sys, os
# we need an agent

# we need a model

# we need a memory buffer

# we need a gaming environment

# we need to train
    # initialize a model or grab an exisiting one
    # run a game environments
    # insert actions
    # extract rewards
    # do this for a whole episode
    # store result/experience
    # update weights


# we need to play trained models
    # load environment
    # load trained model
    # play game


def train():
    agent = Agent()
    game = Game('PongDeterministic-v4'
                ,agent
                ,verbose=False
                ,filePath="./video_capture/"
                )
    
    # have the agen interact with the game
    # store game experiences
    # train on experiences
        # A Q table and NN
        # bellmen loss equation
    while True:
        # play the game
        score = game.play(showRender=True)
        print("score: {}"\
                .format(score))

if __name__ == '__main__':
    try:
       train()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)