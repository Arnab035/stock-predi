import quandl
import pandas as pd
import random
from datetime import datetime

class Environment():

	def __init__(self):
		self.holding_stock = False

		self.q_values = dict()
		self.gamma = 0.10

		self.decayed_learning_rate = 0.30

		self.epsilon = 0.00 #random actions do not seem necessary.
	
	def collectData(self, symbol, start_date, end_date):
                quandl.ApiConfig.api_key = "VkbqJssMFzS4q9jpoiaU"

		df = quandl.get("WIKI/{}".format(symbol), start_date = start_date, 
			            end_date = end_date)

		# pre-processing step
		if df.isnull().values.any():
                        df = df.dropna(axis =0, how='any')

		return df

	def calculateDailyReturns(self, df):
		"""Calculate the daily returns for the particular training stock"""

		daily_returns = df.copy()

		daily_returns[1:] = (df[1:] / df[:-1].values) - 1 # calculate daily returns

		daily_returns[0] = 0 # daily returns for first row is zero

		return daily_returns



	def prepareStates(self, df):

		"""This function converts states that contain floating - point values into discrete values for easy representation"""

		df = df['Adj. Close']
		daily_returns = self.calculateDailyReturns(df)

		rm_df = pd.rolling_mean(df, window = 3) ##keep a small window to not make the rolling mean curve too smooth

		rm_df = rm_df.fillna(method = 'bfill')

		ratioAdjCloseRm = df / rm_df
		"""[Ratio of adj close to rolling mean , holding stocks] would be our state"""

		discreteRatio = discretize(ratioAdjCloseRm)

		return discreteRatio, daily_returns 

	def isStatePresent(self, state):
		for states in self.q_values.keys():
			if state == states:
				return True

		return False

        def update(self, df, rewards, date, testingState, trials):
		self.state = (df[0], self.holding_stock)

                # learning rate starts decaying during testing- after a long training session is complete
		if testingState:
                        self.decayed_learning_rate = self.decayed_learning_rate * (1 / trials)

		if not self.isStatePresent(self.state):
			self.q_values[self.state]  = {'buy' : 0.0, 'sell' : 0.0, 'nothing' : 0.0}

			print "Dictionary of values inserted"

			action = random.choice(['buy','sell','nothing'])
			reward = self.act(action, rewards)

			next_state = (df[1] , self.holding_stock)

			self.q_values[self.state][action]  = self.q_values[self.state][action] * (1- self.decayed_learning_rate) \
			                                     + self.decayed_learning_rate * (reward + (self.gamma * self.q_next(self.state)))
			 #q_next computes the maximum q-value for the next state

		else:
			count = 0

			if random.random() > self.epsilon: ##not necessary actually - randomness decreases the performance
				action = max(self.q_values[self.state], key = lambda i: self.q_values[self.state][i])

				actions = []
				value = self.q_values[self.state][action]

				for k,v in self.q_values[self.state].items():
					if value == v:
						count = count + 1
						actions.append(k)

				#if you have more than 1 q-value
				if count > 1:
                                        action = random.choice(actions)

			else:
                                action = random.choice(['buy','sell','nothing'])

                        reward = self.act(action, rewards)

			next_state = (df[1] , self.holding_stock)

			self.q_values[self.state][action] = self.q_values[self.state][action] * (1-self.decayed_learning_rate) \
			                                    + self.decayed_learning_rate * (reward + (self.gamma * self.q_next(self.state)))

			
                if testingState:
                    print "Action {} taken on date : {} giving us reward {}".format(action, date, reward)

                return reward

        def act(self, action, rewards):
                if action == 'buy':
                        if self.holding_stock:
                                reward = -5 * abs(rewards) #if you are already holding stock - you should not BUY again. That is very wrong!!!
                                return reward   #use return statements to save some time :)
                        else:
                                self.holding_stock = True
                                return rewards
                elif action == 'sell':
                        if self.holding_stock:
                                self.holding_stock = False
                                return 0
                        else:
                                reward = -5 * abs(rewards) #if you do not hold stocks yet, why are you selling them. That is very wrong!!!
                                return reward
                else:
                        if self.holding_stock:
                                return rewards  #if you hold stocks and do nothing, the reward is expected to build
                        else:
                                return 0  #if you do not hold stocks and still do nothing, you get no reward.

        def q_next(self, state):
                if not self.isStatePresent(state):
                        return 0
                else:
                        action = max(self.q_values[state], key= lambda i: self.q_values[state][i])
                        return self.q_values[state][action]



def discretize(df):
	"""This function assigns a discrete value to each ratio"""
	df = df.sort(inplace = False, axis = 1) ##sort the ratios

	steps = 10 ##10 steps from 0-9 the ratios will be assigned any distinct number between 0-9

	stepsize = len(df.index)/steps

	extraData = len(df.index) % steps 

	k = 0

	for i in range(0, steps):
		lower = k
		upper = lower + stepsize
		df[lower : upper] = int(i)
		if (int(i) == 9):
                        df[upper : upper + extraData] = int(i)

		k = upper

    ## sort the dataframe - the way it was initially with all indices arranged chronologically
	df = df.sort_index(inplace = False)

	return df



	









		
		













