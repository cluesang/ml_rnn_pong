import the_agent
import environment
import matplotlib.pyplot as plt
import time
from collections import deque
import numpy as np
from loggers import saveDictionaryToCSV, openTrainingConfig, saveTrainingConfig
from datetime import datetime
from tensorflow.keras.models import load_model
import os
import sys

def train(silent=False,sessionId=None):
    # had to install tensorflow 'pip install tensorflow'
    timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M:%S:%p")

    isResumingSession = (sessionId != None)
    if(not isResumingSession):
        sessionId = str(int(time.time()))
        os.makedirs("./session_"+sessionId+"/")
    else:
        if not os.path.exists("./session_"+sessionId):
            print("Error: cannot find session folder for session id: "+sessionId)
            print("try restarting without providing a session id or placing the session"\
                " folder in the same directory as this script.")
            return

    sessionFolderpath = "./session_"+sessionId+"/"
    configFilename = sessionFolderpath+"config_"+sessionId+".json"
    historyFilename = sessionFolderpath+"history_"+sessionId+".json"
    modelFilename = sessionFolderpath+"model_"+sessionId
    videoFolderPath = sessionFolderpath+"videos_"+sessionId+"/"

    saved_model = None
    if(isResumingSession):    
        config = openTrainingConfig(configFilename)
        saved_model = load_model(modelFilename)
    else:
        config = {
            'name': 'PongDeterministic-v4'
        ,   'possible_actions': [0,2,3]
        ,   'starting_mem_len': 50000
        ,   'max_mem_len': 750000
        ,   'starting_epsilon': 1
        ,   'learn_rate': 0.00025
        ,   'episode_number': 0
        ,   'debug': not silent # set true to see game monitor
        ,   'datetime': timestamp
        ,   'sessionId': sessionId
        }

    agent = the_agent.Agent(config['possible_actions']\
                            ,config['starting_mem_len']\
                            ,config['max_mem_len']\
                            ,config['starting_epsilon']\
                            ,config['learn_rate']\
                            ,debug=config['debug']
                            ,resume_model=saved_model)
    
    env = environment.make_env(config['name'],agent)

    last_100_avg = [-21]
    scores = deque(maxlen = 100)
    max_score = -21

    """ If testing:
    agent.model.load_weights('recent_weights.hdf5')
    agent.model_target.load_weights('recent_weights.hdf5')
    agent.epsilon = 0.0
    """

    env.reset()
    i = int(config['episode_number'])
    recordEpisode = False
    while True:
        timesteps = agent.total_timesteps
        time_elapsed = time.time()
        score = environment.play_episode(config['name']\
                                        , env\
                                        , agent\
                                        , config['debug']\
                                        , record=recordEpisode\
                                        , recordPath=videoFolderPath+"/ep_"+str(i)
                                        ) #set debug to true for rendering
        recordEpisode = False # reset record flag
        scores.append(score)
        if score > max_score:
            max_score = score

        episodeData = {
            'episode_number': str(i)
        ,   'steps': str(agent.total_timesteps - timesteps)
        ,   'duration': str(time.time() - time_elapsed)
        ,   'score': str(score)
        ,   'max_score': str(max_score)
        ,   'epsilon': str(agent.epsilon)
        }
        print(episodeData)

        saveDictionaryToCSV(episodeData,historyFilename) # save history
        agent.model.save(modelFilename) # save model

        config['episode_number'] = str(i)
        saveTrainingConfig(config,configFilename) # saveConfig

        if i%5==0:
            recordEpisode = True
        i += 1

if __name__ == '__main__':
    try:
        train(silent=False,sessionId="1637802637")
        # train(silent=False)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)