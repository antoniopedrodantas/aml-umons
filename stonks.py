import pandas as pd
import matplotlib.pyplot as plt
import math

from src.functions import get_stock_rewards, distribution, draw, get_reward, get_max, get_best_action_cumulative_reward

# gets stock rewards for each company
plt.figure()

# TSLA
data = pd.read_excel('stock_info.xlsx', sheet_name='TSLA')
tsla, tsla_cum = get_stock_rewards(data)
plt.plot(tsla_cum, label='tsla')

# GOOG
data = pd.read_excel('stock_info.xlsx', sheet_name='GOOG')
goog, goog_cum = get_stock_rewards(data)
plt.plot(goog_cum, label='goog')

# AAPL
data = pd.read_excel('stock_info.xlsx', sheet_name='AAPL')
aapl, aapl_cum = get_stock_rewards(data)
plt.plot(aapl_cum, label='aapl')

# FB
data = pd.read_excel('stock_info.xlsx', sheet_name='FB')
fb, fb_cum = get_stock_rewards(data)
plt.plot(fb_cum, label='fb')

# NFLX
data = pd.read_excel('stock_info.xlsx', sheet_name='NFLX')
nflx, nflx_cum = get_stock_rewards(data)
plt.plot(nflx_cum, label='nflx')

# AMZN
data = pd.read_excel('stock_info.xlsx', sheet_name='AMZN')
amzn, amzn_cum = get_stock_rewards(data)
plt.plot(amzn_cum, label='amzn')

# decide what gamma is
# in this case we will set it to 0.3
gamma = 0.2

# we initialize the weights for each hand of the bandit
weights = [1, 1, 1, 1, 1, 1]

# this holds the rewards for each day in every company
rewards = [tsla, goog, aapl, fb, nflx, amzn]
options = ["tesla", "google", "apple", "facebook", "netflix", "amazon"]

regret = []
theoretical_regret = []

cumulative_reward = []
best_action_cumulative_reward = []

t = 1000
k = len(rewards)

# lets have them train for 500 moves
for i in range(t):
    # this is the exp3 algorithm
    # creates a probability distribution that takes into account the gamma value
    probability_distribution = distribution(weights, gamma)
    # gets the choice for each round according to the probability distribution
    choice = draw(probability_distribution)
    # gets the choice's reward
    reward = get_reward(choice, i, rewards)
    estimated_reward = reward / probability_distribution[choice]
    # updates weights
    weights[choice] = weights[choice] * math.exp(estimated_reward * gamma / len(rewards))

    # gets best action reward
    best_action_reward = get_best_action_cumulative_reward(i, rewards)

    # updates regret for every move
    if i == 0:
        regret.append(best_action_reward - reward)
        theoretical_regret.append((math.e - 1) * gamma * best_action_reward + (k * math.log(k)) / gamma)
        cumulative_reward.append(reward)
        best_action_cumulative_reward.append(best_action_reward)
    else:
        regret.append((best_action_reward - reward) + regret[i - 1])
        theoretical_regret.append(((math.e - 1) * gamma * best_action_reward + (k * math.log(k)) / gamma) + theoretical_regret[i - 1])
        cumulative_reward.append(reward + cumulative_reward[i - 1])
        best_action_cumulative_reward.append(best_action_reward + best_action_cumulative_reward[i - 1])

    if (i + 1) % 50 == 0:
        print("for iteration number", i + 1, "exp3 decided to invest in", options[choice])

# compares final weights to figure out  which is the best company to invest in
best_option = get_max(weights)
print("The Exp3 algorithm says", options[best_option], "is the best company to invest in")

# shows stock cumulative rewards on a plot
plt.legend(loc=0)
plt.title('Cumulative rewards for FAANG companies')
plt.show()

# shows investment cumulative cumulative on a plot
plt.figure()
plt.plot(cumulative_reward, label='cumulative reward')
# plt.plot(best_action_cumulative_reward, label='best action cumulative reward')
plt.legend(loc=0)
plt.title('Rewards for the Exp3 Algorithm')
plt.show()

# shows investment cumulative regret on a plot
plt.figure()
plt.plot(regret, label='exp3 cumulative regret')
plt.plot(theoretical_regret, label='theoretical cumulative regret')
plt.legend(loc=0)
plt.title('Regret for the Exp3 Algorithm')
plt.show()
