import json
import csv
import os
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def openOrCreateWeights(defaultWeights, filename):
    weights = defaultWeights
    try:
        jsonFile = open(filename, 'r')
        weightsJSONSerialized = jsonFile.read()
        weights = json.loads(weightsJSONSerialized)
        weights['1'] = np.asarray(weights['1'])
        weights['2'] = np.asarray(weights['2'])
    except:
        saveWeights(weights, filename)
    
    return weights
       
def saveWeights(weights, filename):
    weightsJSONSerialized = json.dumps(weights, cls=NumpyArrayEncoder)
    jsonFile = open(filename,'w')
    jsonFile.write(weightsJSONSerialized)
    jsonFile.close()
    return True

def saveEpisodeHistory(episodeData, filename):
    headers = ['episode_number', 'reward_sum', 'running_reward']
    entryData = [episodeData['episode_number']
                ,episodeData['reward_sum']
                ,episodeData['running_reward']]
    firstRun = False
    if (not os.path.exists(filename)):
            firstRun = True
    file = open(filename, 'a')
    writer = csv.writer(file)
    if (firstRun):
        writer.writerow(headers)
    writer.writerow(entryData)
    file.close()
    return True

def saveTrainingConfig(config,filename):
    JSONSerialized = json.dumps(config, cls=NumpyArrayEncoder)
    jsonFile = open(filename,'w')
    jsonFile.write(JSONSerialized+"\n")
    jsonFile.close()
    return True

def openTrainingConfig(filename):
    jsonFile = open(filename, 'r')
    configJSONSerialized = jsonFile.read()
    config = json.loads(configJSONSerialized)
    jsonFile.close()
    return config

def saveDictionaryToCSV(dictData,filenameAndPath):
    headers = dictData.keys()
    values = dictData.values()
    firstRun = False
    try:
        if (not os.path.exists(filenameAndPath)):
            firstRun = True
        file = open(filenameAndPath, 'a')
        writer = csv.writer(file)
        if (firstRun):
            writer.writerow(headers)
        writer.writerow(values)
        file.close()
    except IOError:
        print("I/O error")

def saveModelJsonSummary(modelJSON, summaryFilenameAndPath):
    with open(summaryFilenameAndPath, "w") as json_file:
        json_file.write(modelJSON)