import numpy as np
import enum
import random
class PongActions(enum.Enum):
    do_nothing = 0
    up = 2
    down = 5

class Agent():
    def __init__(self) -> None:
        pass

    def build_model(self):
        pass

    def load_model(self):
        pass

    def get_action(self,game_state=None):             
        action = random.sample([act.value for act in PongActions],1)[0]
        # use game_state to find the next
        # appropriate action
        return action

    def add_experience(action, observation, reward, done):
        # we take an env's step results and the action that preceded
        # the step and store that as an experience.
        
        pass

    def learn(self):
        pass

if __name__ == '__main__':
    try:
      pass
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)