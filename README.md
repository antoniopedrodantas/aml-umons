# Advanced Machine Learning - Adversarial Bandits

## **Exp3 Implementation**

We decided to implement the **Exp3** algorithm on stock market data for the practical project of the Advanced Machine Learning course. For this, we used market data provided by **Google Finance** using **Google Sheets**. We got the information of six top 500 companies for the last 4 years. The companies chosen were **Tesla**, **Google**, **Facebook**, **Apple**, **Netflix** and **Amazon**. This method lets us get the daily open and close price for stocks of each company. These values will be very important for the algorithm's implementation.

![Stock Info](https://i.imgur.com/zol9sLc.png)

After that, we had to export the sheets document to a .xlsx file to be able to parse this information to our python code.

As we know, an adversarial bandit problem is a pair **(*K*, x)** where *K* represents the number of actions (indexed by *i*) and x is the sequence of payoff vectors x = { x*i*(t) } that represents the reward of action *i* on step t. The game is played in rounds indexed by t = {1, 2, ..., n} and the payoffs are fixed for each action and time before the game starts. The player only knows the value of *K* and the reward for the time step he is in.

With this in mind, we can assume that for our case, *K* represents the six companies from which we extracted the market information. Each time step t represents a day from which our algorithm will have to choose a company to invest in, in our case, buy one stock from that company. At the end of the day he will sell the stock he invested in, making our reward vector x the **difference between the closing and opening price of stock** for each company every day. We then applied the Exp3 algorithm to find out what's the cumulative reward we could expect by applying it to the stock market.

**Exp3** stands for Exponential-weight algorithm for Exploration and Exploitation. It works by maintaining a list of weights to decide which action to take, updating them at each time step.The egalitarianism facotor gamma tunes the desire for the algorithm to choose an action at random (Exploration) or to choose the action with the highest weight (Exploitation). If gamma is set to 1, the weights have no effect on the choices of any step. The pseudocode for the algorithm is as follows.

![Exp3 pseudocode](https://i.imgur.com/pAEYP8W.png)

We now have everything to get our solution and analyse our results. Our Exp3 implementation adapted to the stock market problem is as follows.

```python
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
```

## **Result Analysis**

To better understand our results and tell whether or not the algorithm would make good decisions we start by showing the cumulative rewards of each company for which we have the option to invest in.

![Stock Cumulative Reward](https://i.imgur.com/d4Bem0u.png)

The graph shows us the daily cumulative profit, so at the last time step we can see how much reward we would get if we invested on each company on every day.

We also output the company that ends up with the highest weight (that should be the best company to invest in) and percentage of times the algorithm chooses to invest in the company that has the best daily reward (this means that it took the best action possible).

![Important results information](https://i.imgur.com/MdXDhXx.png)

Twicking the gamma value is a way of getting different types of results since lowering it will grant us a bigger exploitation of the weights of each action. We found out that having a value of 0.2 to 0.3 is a good measure to obtain the best cumulative reward over a long period of time (approximately 1000 time steps). This means that the Exp3 algorithm will favor the company that has the best weight at the time while still leaving some room to try different ones.

![Cumulative Rewards 1](https://i.imgur.com/5EkOxZk.png)

However, it is important to note that the weights of each company will heavily depend not only on the first choices the algorithm take but also on the random ones he makes throughout his run. There are times where the algorithm gets results that are not optimal and sometimes not even desirable. We can justify that on the nature of data we are analysing. The stock market is a fleeding and unstable environment and there isn't a company (or in this case, an action) that is superior to the others as a whole. There are times when it makes sense to invest in a company, but after a while, the same company can start to be unprofitable for us and that is something we need to take into account. The **Exp3** algorithm can understand these phenomenons in the long run, but when it is time to take its decision on the spot it is prone to errors and gets mislead by its current weights. It is likely that it will find out what the best company is at the end of the run, but to do that it will probably make some bad investments on the way.

![Cumulative Rewards 2](https://i.imgur.com/8M2hRCm.png)
![Cumulative Rewards 3](https://i.imgur.com/b7OyqzP.png)

Nonetheless, we found this algorithm to be very powerful and could be a very important tool in the resolution of other problems or used in conjunction with other criteria to solve this one. We can justify that by keeping track of the regret we should expect to get based on the **Upper Bound** of weak regret of Exp3 **Theorem** and the regret we actually got.

![Upper Bound Theorem](https://i.imgur.com/cosntUY.png)
![Regret Comparison](https://i.imgur.com/xcuF8HA.png)

## **References**

- Bandit algorithms. Lattimore, Tor, and Csaba Szepesva ́ri. Cambridge University Press, (2020).

- [Adversarial Bandits and The Exp3 Algorithm, jeremykun, (2013).](https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm)

- [Bandits and Stocks, jeremykun, (2013).](https://jeremykun.com/2013/12/09/bandits-and-stocks)
