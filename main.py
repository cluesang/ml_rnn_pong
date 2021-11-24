import gym
import sys
import os
import numpy as np
from json import JSONEncoder
import json
from datetime import datetime

################ Image Preprocessing  ###################

def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    # convert the 210x160x3 uint8 frame into a 6400 float vector 
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_observation = processed_observation.astype(np.float64).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations


def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def downsample(image):
    # We will take only half of the image resolution
    return image[::2, ::2, :]

def remove_color(image):
    # We dont need the image colors
    return image[:, :, 0]

############### Activation functions ####################

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(vector):
    vector[vector < 0] = 0
    return vector
    
    
################# Neural net #######################

def neural_net(observation_matrix, weights):
    # Compute the new hidden layer values and the new output layer values using the observation_matrix and weights 
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values

def Move_up_or_down(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        # up in openai gym
        return 2
    else:
         # down in openai gym
        return 3

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2
    }

def weights_update(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
        weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name]) # reset batch gradient buffer


############### Reinforcement learning ################

def discount_rewards(rewards, gamma):
   # Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago. This implements that logic by discounting the reward on previous actions based on how long ago they were taken
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def discount_plus_rewards(gradient_log_p, episode_rewards, gamma):
    # discount the gradient with the normalized rewards 
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards

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
    JSONSerialized = json.dumps(episodeData, cls=NumpyArrayEncoder)
    jsonFile = open(filename,'a')
    jsonFile.write(JSONSerialized+"\n")
    jsonFile.close()
    return True

def saveTrainingConfig(config,timestamp):
    JSONSerialized = json.dumps(config, cls=NumpyArrayEncoder)
    jsonFile = open("trainingConfig_"+timestamp+".json",'a')
    jsonFile.write(JSONSerialized+"\n")
    jsonFile.close()
    return True

#################### The game  ##########################
def main(silent=False):
    env = gym.make("Pong-v0")
    observation = env.reset() # This gets us the image

    # hyperparameters
    episode_number = 0
    batch_size = 10
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 80 * 80
    learning_rate = 1e-4
    reward_sum = 0
    running_reward = None
    prev_processed_observations = None

    sessionTimestampID = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    trainingConfig = {
        'batch_size': batch_size
    ,   'gamma': gamma
    ,   'decay_rate': decay_rate
    ,   'num_hidden_layer_neurons': num_hidden_layer_neurons
    ,   'input_dimensions': input_dimensions
    ,   'learning_rate': learning_rate
    ,   'training_datetime': sessionTimestampID
    }
    saveTrainingConfig(trainingConfig, sessionTimestampID)
    historyFilename = "history_"+sessionTimestampID+".json"
    weightsFilename = "weights_"+sessionTimestampID+".json"
    defaultWeights = {
        '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
        '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
    }
    weights = openOrCreateWeights(defaultWeights,weightsFilename)
    

    # To be used with rmsprop algorithm 
    expectation_g_squared = {}
    g_dict = {}
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []


    while True:
        if(silent):
            env.render(mode='rgb_array')
        else:
            env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
        hidden_layer_values, up_probability = neural_net(processed_observations, weights)
    
        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = Move_up_or_down(up_probability)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)

        # see here: http://cs231n.github.io/neural-networks-2/#losses
        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        episode_gradient_log_ps.append(loss_function_gradient)


        if done: # an episode finished
            episode_number += 1

            # Combine the following values for the episode
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            # Tweak the gradient of the log_ps based on the discounted rewards
            episode_gradient_log_ps_discounted = discount_plus_rewards(episode_gradient_log_ps, episode_rewards, gamma)

            gradient = compute_gradient(
              episode_gradient_log_ps_discounted,
              episode_hidden_layer_values,
              episode_observations,
              weights
            )

            # Sum the gradient for use when we hit the batch size
            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]

            if episode_number % batch_size == 0:
                weights_update(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)

            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], [] # reset values
            observation = env.reset() # reset env
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            saveEpisodeHistory({'episode_number': episode_number,'reward_sum': reward_sum, 'running_reward': running_reward}, historyFilename)
            saveWeights(weights,weightsFilename)
            reward_sum = 0
            prev_processed_observations = None
           


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)