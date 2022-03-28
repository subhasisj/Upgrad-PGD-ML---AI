# Import routines

import numpy as np
import math
import random


# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

"""
Location λ(of Poisson Distribution)
    0       2
    1       12
    2       4
    3       7
    4       8
"""
# below variable contains the poisson mean mapping with location in key-value pair
poisson_mean = { '0' : 2, '1' : 12, '2' : 4, '3' : 7, '4' : 8 }

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        # action_space variable contains all the allowed/possible action an agent can take.
        # This excludes actions space like [(1,1), (2,2), (3,3), (4,4)]
        self.action_space = tuple([(0, 0)]) + tuple(((x, y) for x in range(m) for y in range(m) if x != y))

        # state_space variable contains all the possible state in which agent can be.
        self.state_space = tuple(((x, y, z) for x in range(m) for y in range(t) for z in range(d)))

        # randomly choosing a state from state_space
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input
    def state_encod_arch1(self, state):
        """
        convert the state into a vector so that it can be fed to the NN. 
        This method converts a given state into a vector format. 
        Hint: The vector is of size m + t + d.
        """
        # based on current location, hour and day, creates a vector representation
        (curr_loc, curr_hour, curr_day) = state
        # location vector
        loc_vector = tuple((0 if x != curr_loc else 1 for x in range(m)))
        # hour vector
        hour_vector = tuple((0 if x != curr_hour else 1 for x in range(t)))
        # week day vector
        week_day_vector = tuple((0 if x != curr_day else 1 for x in range(d)))

        # combining all the vector in order of location, hour and week day
        state_encod = loc_vector + hour_vector + week_day_vector

        return state_encod



    ## Getting number of requests

    def requests(self, state):
        """
        Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations
        """
        # get location from state
        (location, _, _) = state
        # using possion_mean to get the mean value for the location.
        # This was done for faster processing rather than using if-else check
        requests = np.random.poisson(poisson_mean.get(str(location)))

        # make sure request is not 0 so agent always gets a request
        if requests == 0:
            requests = poisson_mean.get(str(location))

        # make sure agents doesn't get request more than 15 
        if requests > 15:
            requests = 15

        # get a random drive request based on request count
        possible_actions_index = random.sample(range(1, (m-1) * m + 1), requests)
        actions = [self.action_space[i] for i in possible_actions_index]

        # Note: appending offline action only for the possible index, not for possible index.
        # Idea here is get to offline action only based on the max q-value. 
        actions.append((0,0))

        return (tuple(possible_actions_index), tuple(actions))


    def get_updated_hour_and_day(self, present_hour, present_day, working_hours):
        """
        Purpose of this method to update agent present working hours and day to new working hours and days
        """
        # updated hours after accounting for agent droping customer to new location
        updated_hour = present_hour + working_hours
        # updating week day to present day. 
        # This will incremented by 1 in case agent drop time has passed next week day
        # This will be updated 0 when agent drop time has passed next week day and week day is 6
        updated_day = present_day

        # if hours are more than 23, update the update_hours to next day hour and also increment the udpate_day
        if updated_hour >= t :
            updated_hour = updated_hour % t
            updated_day = updated_day + 1
            # incase present_day week was 6 and agent drop time has passed present day then make week_day to 0
            if updated_day == d:
                updated_day = 0
        
        return (int(updated_hour), updated_day)


    def reward_func(self, isActionOffline, curr_to_pickup_loc_time, pickup_to_drop_loc_time):
        """Takes in state, action and Time-matrix and returns the reward"""
        if isActionOffline :
            # incase agent choose to go offline by choosing action (0,0)
            # reward would -C
            reward = -C
        else:
            # calculating the reward
            reward = (R * pickup_to_drop_loc_time) - (C * (curr_to_pickup_loc_time + pickup_to_drop_loc_time))

        return reward


    def next_state_func(self, state, action, Time_matrix):
        """
        Takes state and action as input and 
        returns next_state, pickup time and drop time
        """
        # initialise the variables
        (curr_loc, curr_hour, curr_day) = state
        (pickup_loc, drop_loc) = action
        (curr_to_pickup_loc_time, pickup_to_drop_loc_time) = (None, None)

        if action in [(0, 0)]:
            # incase agent choose to go offline by choosing action (0,0)
            # update hour and week day by accounting for 1 hour no work. location would remain same
            updated_hour, updated_day = self.get_updated_hour_and_day(curr_hour, curr_day, 1)
            
            # location remain same, just updating hour and week day
            next_state = (curr_loc, updated_hour, updated_day)
        else:
            # getting time required for agent to reach to customer pickup location
            # in case agent and customer is on same location then this would be zero. Time_matrix already takes cares.
            curr_to_pickup_loc_time = Time_matrix[curr_loc][pickup_loc][curr_hour][curr_day]

            # considering agent location to pickup location and get updated hours and week day
            ride_start_hour, ride_day = self.get_updated_hour_and_day(curr_hour, curr_day, curr_to_pickup_loc_time)

            # getting time required agent to reach to customer drop location
            pickup_to_drop_loc_time = Time_matrix[pickup_loc][drop_loc][ride_start_hour][ride_day]

            # considering agent location to drop location and get updated hours and week day
            updated_hour, updated_day = self.get_updated_hour_and_day(ride_start_hour, ride_day, pickup_to_drop_loc_time)      
            
            # location changes to customer drop location and also updating hour and weeks
            next_state = (drop_loc, updated_hour, updated_day)
        
        return (next_state, curr_to_pickup_loc_time, pickup_to_drop_loc_time)

    
    def step(self, state, action, Time_matrix):
        """ function to take an action from a given state"""
        # return new state, time from current car location to pickup location and time from pickup to drop location
        (next_state, curr_to_pickup_loc_time, pickup_to_drop_loc_time) = self.next_state_func(state, action, Time_matrix)

        # initialise 
        isActionOffline = False

        # if action is offline.
        if action in [(0, 0)]:
            isActionOffline = True
            # total worked hours would be 1 as driver will get request only next hour. And agent has opted not to take a ride.
            total_worked_hours = 1
        else:
            # calculate total worked hours
            total_worked_hours = curr_to_pickup_loc_time + pickup_to_drop_loc_time

        # calculate rewards.
        reward = self.reward_func(isActionOffline, curr_to_pickup_loc_time, pickup_to_drop_loc_time)

        # return next state, rewards, total time to server a ride
        return (next_state, reward, total_worked_hours)



    def reset(self):
        return self.action_space, self.state_space, self.state_init



####################################
# Unit Testing 
####################################
if __name__ == "__main__":
    env = CabDriver()
    Time_matrix = np.load("TM.npy")

    action_space, state_space, initial_state = env.reset()
    # print('Cab Driver Action Space:[{}]'.format(action_space))
    # print('Cab Driver State Space:[{}]'.format(state_space))
    print('Cab Driver random initial Space:[{}] \n'.format(initial_state))


    print('Cab Driver Request API:')
    for x in range(10):
        print(env.requests(initial_state))
    print()

    print('Cab Driver State Vector: [{}]\n'.format(env.state_encod_arch1(initial_state)))
    print('Cab Driver State Vector Length: [{}]\n'.format(len(env.state_encod_arch1(initial_state))))


    print('Cab Driver action and new state:')
    state = initial_state
    action = random.choice(env.requests(initial_state)[1])
    for x in range(30):
        (_, actions) = env.requests(initial_state)
        action = random.choice(actions)
        (state, reward, time) = env.step(state, action, Time_matrix)
        print('Action:[{}] Next State:[{}] Reward:[{}] time:[{}]'.format(action, state, reward, time))
    print()
