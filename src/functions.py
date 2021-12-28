import random

def get_stock_rewards(data):
    cmp_open = data['Open'].tolist()
    cmp_close = data['Close'].tolist()
    cmp_rewards = []
    cmp_cum = []
    for i in range(len(cmp_open)):
        cmp_rewards.append(cmp_close[i] - cmp_open[i])
        if i == 0:
            cmp_cum.append(cmp_close[i] - cmp_open[i])
        else:
            cmp_cum.append((cmp_close[i] - cmp_open[i]) + cmp_cum[i - 1])
    return cmp_rewards, cmp_cum

def distribution(weights, gamma):
    theSum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)

def draw(probability_distribution):
    choice = random.uniform(0, sum(probability_distribution))
    choiceIndex = 0

    for weight in probability_distribution:
        choice -= weight
        if choice <= 0:
            return choiceIndex

        choiceIndex += 1

def get_reward(choice, index, rewards):
    return rewards[choice][index]

def get_max(weights):
    max = 0
    pick = 0
    for i in range(len(weights)):
        if weights[i] > max:
            max = weights[i]
            pick = i
    return pick

def get_best_action_cumulative_reward(index, rewards):
    max = -999
    for cmp in rewards:
        if cmp[index] > max:
            max = cmp[index]
    return max
