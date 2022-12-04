# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint
from operator import itemgetter


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        print("My best_act: ", best_actions)
        food_left = len(self.get_food(game_state).as_list())
        #print("The food carry: ",game_state.get_agent_state(self.index).num_carrying)
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
               # print("AAAAAAAAAAAACT")
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    

    def choose_action(self, game_state):
        """
        Our strategy is: Offensive is trying to pick 5 foods and come back. When our score is 10,
        offensive become defensive and trying eat enemies.
        """
        actions = game_state.get_legal_actions(self.index)
        food_left = len(self.get_food(game_state).as_list())
    
        if (self.get_score(game_state)<10) and (self.get_score(game_state)>=0):
                values = [self.evaluate(game_state, a) for a in actions]
                
                food_carry=game_state.get_agent_state(self.index).num_carrying
                max_value = max(values)
                best_actions = [a for a, v in zip(actions, values) if v == max_value]
                if food_carry>=5:
                
                    return self.go_home(game_state,actions)
                elif food_left <= 2:
                    best_dist = 9999
                    best_action = None
                    for action in actions:
                    
                        successor = self.get_successor(game_state, action)
                        pos2 = successor.get_agent_position(self.index)
                        dist = self.get_maze_distance(self.start, pos2)
                        
                        if dist < best_dist:
                            best_action = action
                            best_dist = dist
                    return best_action

                return random.choice(best_actions)
        else:
            values = [self.evaluate_def(game_state,a)for a in actions]
            max_value = max(values)
            best_actions = [a for a, v in zip(actions, values) if v == max_value]
            print("Defend mode!!!")
            return random.choice(best_actions)

    
    
    def get_features_def(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features
    
    def get_weights_def(self, game_state, action): #weight when offensive pacman decides to defend
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
     
    def evaluate_def(self,game_state,action): #evaluation when pacman decides to defend
        features = self.get_features_def(game_state, action)
        weights = self.get_weights_def(game_state, action)
        print("Def evaluate!")  
        return features * weights 
      
    def evaluate(self, game_state, action): #offensive evaluation
        successorGameState = game_state.generate_successor(self.index,action)
        newPosState = successorGameState.get_agent_state(self.index)
        newPos=newPosState.get_position()
        newFood = successorGameState.get_blue_food()
        actions = game_state.get_legal_actions(self.index)
        ghostDistance=[]
        foodDistance=[]
        ghostStates=[]
        ghostDistanceFromStart=[]
        score = 0    
        # if we can observe at least one ghost we can calculate distance between agent and enemy and distance between agent and food
        if ((game_state.get_agent_position(game_state.get_blue_team_indices()[0])!=None)or(game_state.get_agent_position(game_state.get_blue_team_indices()[1])!=None)): 
                for action in actions:

                        #calculate distance between agent and food
                        for food in newFood.as_list():
                            
                            foodDistance.append(self.get_maze_distance(newPos,food))
                       
                        for i in range(len(game_state.get_blue_team_indices())):
                        
                           
                            ghostStates.append(game_state.get_agent_state(game_state.get_blue_team_indices()[0]).get_position())
                            ghostStates.append(game_state.get_agent_state(game_state.get_blue_team_indices()[1]).get_position())
                        
                        for ghost in ghostStates:
                            if ghost != None: # calculate distance if we can
                            
                                ghostDistance.append(self.get_maze_distance(ghost,newPos)) 
                                ghostDistanceFromStart.append(self.get_maze_distance(ghost,self.start))     
                
                
                    
            
                for j in range(len(foodDistance)):
                    for i in range (len(ghostDistance)):
                    
                        if ghostDistance[i]>foodDistance[j]: #if ghost farer than food
                            print("Not scary")
                            score+=1
                        else:
                            if ghostDistanceFromStart[i]>foodDistance[j]: #trying to check: if enemy on our territory we dont scare them
                                print("I am scary")
                                score-=1
                            else:
                                print("ITS MY LAAAND")
                                score+=1

                "* YOUR CODE HERE *"
                return successorGameState.get_score()+score
         #if enemies are  not observable we just choose best actions   
        else:
            features = self.get_features(game_state, action)
            weights = self.get_weights(game_state, action)
            
            return features * weights

    # if we collect enough food we want to carry it on our territory. This function will action that return pacman at home   
    def go_home(self,game_state,actions):
        action_home_dist = []
        print("I am hooome")
        for action in actions:
        
           
            successor_state = game_state.generate_successor(self.index, action)
            print("type of successor: ",type(successor_state))
           
            agent_state = successor_state.get_agent_state(self.index)
            
            new_pos = agent_state.get_position()
            dist_to_initial_pos = self.get_maze_distance(new_pos, self.start)
            action_home_dist.append((action, dist_to_initial_pos))
           
        return min(action_home_dist, key=itemgetter(1))[0]
        
            
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        print(food_list)
        features['successor_score'] = -len(food_list)  # self.getScore(successor)
	  
        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
    
    
