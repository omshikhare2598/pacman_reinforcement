# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        #Initializing Q values and Previous pacman state to None as we do not it Initially  
        self.Q_values = {}
        self.previous_pacman_state = None
        self.counter =0
    
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    
    
    #Called everytime when pacman needs to perform a Action.
    def getAction(self, state):

        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        
        #The below function Decides actions that pacman should take based on the state it is in
        def action_decide():
            #Consider the we Do not know what the pacmans previous state was
            if self.previous_pacman_state ==None:
                #Initializing Previous State of Pacman to the current state
                self.previous_pacman_state= state  
                #Initially to go from state1->state2 we choose a random action (Note: Random action should be there in the legal actions)
                pick = random.choice(legal)
                #Storing the action taken into some_variable (previous_action) for future use.
                self.previous_action = pick

                
            else:
                #Preparing Data for TD formula TD= IR+gamma*(max(q_values[current_state]) - q_values[previous_state,action])
                ir = state.getScore() - self.previous_pacman_state.getScore()
                cur_state= max(self.Q_values[state].values())
                prev_state= self.Q_values[self.previous_pacman_state][self.previous_action]
                # The Formula for TD = ImmidiateReward + discounted_factor * ( U(s') - U(s) ) 
                TD = (ir) + (self.gamma) * ((cur_state) - (prev_state))
                #Updating the Q-Values of states
                self.Q_values[self.previous_pacman_state][self.previous_action]= (self.Q_values[self.previous_pacman_state][self.previous_action]+ self.alpha*TD)
                

                #Calculating Next Action
                #Exploration vs Exploitation
                import numpy as np
                randn= np.random.random()
                if(randn < self.epsilon):
                    #Here , the Pacman is Exploring as it is taking random actions
                    pick =random.choice(legal)
                    #Storing current state as Previous Pacman State and action taken as previous action
                    self.previous_pacman_state= state
                    self.previous_action = pick
                else:
                    #Here the Pacman is Exploiting
                    #Pacman is picking the Action with maximum Q_value
                    pick = max(self.Q_values[state], key= self.Q_values[state].get)
                    #This is a loop to avoid Illegal Action being taken
                    while pick not in legal:
                        temp = self.Q_values[state]
                        del temp[pick]
                        pick = max(temp,key= temp.get)
                    
                    self.previous_pacman_state = state
                    self.previous_action = pick
            return pick

        #  The the pacman is visiting a New State    
        if state not in self.Q_values:
            #Initialize its Q_values to 0
            self.Q_values[state] = {'East':0,'West':0, 'North':0, 'South':0}
            #Decide the Action to take to move to future state from current state
            pick= action_decide()
        # The Pacman is visiting a previously visited state in the Games    
        elif state in self.Q_values:
            #This action_decide() will return a action, Given that it is Exploring or Exploiting
            pick = action_decide()

        return pick
                    

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        print "A game just ended!"
        
        #Calculating the Qvalue when the pacman takes action and dies at a given state s.
        ir = state.getScore() - self.previous_pacman_state.getScore()
        max_qvalue_state= 0
        sec_term= self.Q_values[self.previous_pacman_state][self.previous_action]
        TD = (ir) + (self.gamma) * ((max_qvalue_state) - (sec_term))
        
        #Updating Q_values after the Game is ended
        self.Q_values[self.previous_pacman_state][self.previous_action]= (self.Q_values[self.previous_pacman_state][self.previous_action]+ self.alpha*TD)
        
        #Initializing Previous pacman state and action =None. To Lean the New Pacman Game
        self.previous_pacman_state= None
        self.previous_action=None
        #Over the time we reduct the epsilon, as our end result is to make optimal moves and win the game. 
        self.setEpsilon(self.epsilon - 0.01)
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)


