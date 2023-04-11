# Implement vanilla Thompson sampling
import random as rand
import heapq
import numpy as np
import matplotlib.pyplot as plt

# Create Random bandit machine with unknown mean and variance
class BernoulliBandit:
    def __init__(self, arms_no, id_no, estimation_threshold):
        '''
        arms: A list of each arm and it's mean at index: arm number
        id_no: ID number for the bandit
        Arms_no: number of arms this bandit has
        '''
        self.arms = []
        # Give each arm of the bandit a mean
        for arm in range(arms_no):
            self.arms.append(rand.random())

        self.id_no = id_no
        self.threshold = estimation_threshold
        self.arms_no = arms_no

        self.best_arm = np.argmax(self.arms)
    

    def sample(self, arm):
        '''
        Returns: Bernoulli reward with probability of the mean of the Bandit
        '''
        # Compares with mean of arm
        if rand.random() < self.arms[arm]:
            return 1
        
        else:
            return 0
    
    def guess(self, guess, arm):
        '''
        Guess: A float of the estimated mean for the bandit
        
        Returns:
            True if value is within threshold, False otherwise
        '''
        isTrue = abs(self.arms[arm] - guess)

        if isTrue <= self.threshold:
            return True
        
        return False
    
    def get_best_arm(self):
        return self.best_arm

class GaussianBandit:
    def __init__(self, num_arms, id_num,  sigma=1): # add mu
        self.num_arms = num_arms
        self.id_num = id_num
        #self.mean = mu
        self.deviation = sigma
        # Initialize the mean reward for each arm
        self.arms = []
        # Give each arm of the bandit a mean
        for i in range(num_arms):
            x = rand.random() *10
            #print(x)
            self.arms.append(x)

        self.best_arm = np.argmax(self.arms)
        self.visit_count = 0
        self.sum_satisfaction = 0
        
    def sample(self, arm):
        # Sample a reward for the given arm based on the Gaussian distribution
        self.visit_count += 1

        s = np.random.normal(self.arms[arm],self.deviation)

        self.sum_satisfaction += s
        return s
    
    def get_best_arm(self):
        return self.best_arm

# This Agent plays the game and makes estimates on the 
class TSBernoulli_Agent:
    def __init__(self, epsilon, decay_rate, epsilon_end):
        self.bandit_interactions = {} #maybe make this a heap in the future
        self.epsilon = epsilon
        self.decay = decay_rate
        self.epsilon_end = epsilon_end
        self.action_count = 0
        self.cum_reward = []

    def choose_arm(self, sample_list):
        '''
        Returns List of best arm number for each known bandit
        Sample_list: A list of the ID numbers of Bandits to choose
        '''
        best_arm_idx_list = []

        for bandit in sample_list:
            best_idx = 0 
            best_reward = 0
            idx = 0

            # For every bandit in our sample list, sample the best arm
            for alpha, beta in self.bandit_interactions[bandit][0]:
 
                x = np.random.beta(alpha, beta)
                if x > best_reward:
                    best_reward = x
                    best_idx = idx
                idx += 1      

            best_arm_idx_list.append(best_idx)
        
        return best_arm_idx_list
    
    def choose_bandit(self, sample_no):
        '''
        Returns a number of bandits to test
        Sample_no: The number of bandits to test
        '''     
        sample = np.random.choice(list(self.bandit_interactions.keys()), size=sample_no)

        return sample

    def update_estimates(self, reward, id, arm_no):
        '''
        Reward: The reward received from a bandit
        ID: The id number of the bandit
        Arm_no: The arm of bandit ID that was pulled
        '''
        #Update List, reward count and visit count
        distribution, c, rc = self.bandit_interactions[id]
        #Update Alpha Beta count for list at arm arm_no
        if reward == 1:
          distribution[arm_no][0] += 1
        else:
          distribution[arm_no][1] += 1
        
        self.bandit_interactions[id] = (distribution, c + 1, rc+reward)         
        
        # Record cumulative reward
        if len(self.cum_reward) > 0:
          self.cum_reward.append(self.cum_reward[-1] +reward)
        else:
          self.cum_reward.append(reward)
        return

    def add_bandit(self, id, num_of_arms, arm_alpha = 1, arm_beta = 1 ):
        '''
        ID: ID number of the bandit to be added to self history
        Bandit_alpah/beta: The estimated reward from the bandit, initialized to 1 for uniform beta distribution
        '''
        # Bandit item has tuple (Beta distribution for each arm in bandit, visit_count, cumulative_reward)
        arm_distribution = []
        for i in range(num_of_arms):
            arm_distribution.append([arm_alpha, arm_beta])

        self.bandit_interactions[id] = (arm_distribution, 0, 0)
        
        return False
    
    def show_estimates(self):
        '''
        Return a list of each bandit and assumed win probability
        '''
        estimates = list()

        for key in self.bandit_interactions:
          if self.bandit_interactions[key][2] != 0:
            estimates.append((key,self.bandit_interactions[key][0]/self.bandit_interactions[key][2]))
          else:
            estimates.append((key, 0))
        return estimates
    
    def get_cum_reward(self):
      '''
      Return the cumulative reward for the agent
      '''
      return self.cum_reward[-1]
####################################################################################

####################################################################################
class TSGaussian_Agent:
    def __init__(self,bandit_count, arm_co, mu=0, sigma=100, ):
        self.prior_mu_of_mu = np.zeros((bandit_count,arm_co))
        
        self.prior_sigma_of_mu = np.full((bandit_count, arm_co), 100.0)
        self.bandit_interactions = {} #maybe make this a heap in the future
        
        self.post_mu_of_mu = self.prior_mu_of_mu
        self.post_sigma_of_mu = self.prior_sigma_of_mu

        self.action_count = 0
        self.cum_reward = []
        
        self.n = 0
        self.sum_satisfaction = 0
            
    def choose_arm(self, sample_list):
        '''
        Returns List of best arm number for each known bandit
        Sample_list: A list of the ID numbers of Bandits to choose
        '''
        best_arm_idx_list = []
        
        for bandit in sample_list:
            best_idx = 0 
            best_reward = float('-inf')
            idx = 0

            # For every bandit in our sample list, sample the best arm
            for mean, sig in zip(self.prior_mu_of_mu[bandit], self.prior_sigma_of_mu[bandit]):              
              x = np.random.normal(mean, sig)
            
              # Record Best index
              if x > best_reward:
                best_reward = x
                best_idx = idx
              
              idx += 1      

            best_arm_idx_list.append(best_idx)

        return best_arm_idx_list
    
    def choose_bandit(self, sample_no):
        '''
        Returns a number of bandits to test
        Sample_no: The number of bandits to test
        '''     
        sample = np.random.choice(list(self.bandit_interactions.keys()), size=sample_no)

        return sample

    def update_estimates(self, reward, id, arm_no):
        '''
        Reward: The reward received from a bandit
        ID: The id number of the bandit
        Arm_no: The arm of bandit ID that was pulled
        '''
        # Visit Count and Cumulative Reward
        c, rc = self.bandit_interactions[id]
        # Since updating every experience, count is always equal to 1
        self.bandit_interactions[id] = (1, rc+reward)

        #print("Count and Cum_Reward and reward: ", c, rc, reward)
        # Posterior Update 
        y = self.prior_sigma_of_mu[id][arm_no]**2
        s = np.sqrt((1 /y + c/ 1)**-1)
        self.post_sigma_of_mu[id][arm_no] = s
        self.post_mu_of_mu[id][arm_no] = (s**2) * ((self.prior_mu_of_mu[id][arm_no] /y) + (reward / 1))
  
        # Update Prior
        self.prior_mu_of_mu[id][arm_no] = self.post_mu_of_mu[id][arm_no]
        self.prior_sigma_of_mu[id][arm_no] = self.post_sigma_of_mu[id][arm_no]

        # Record cumulative reward
        if len(self.cum_reward) > 0:
          self.cum_reward.append(self.cum_reward[-1] +reward)
        else:
          self.cum_reward.append(reward)
        return

    def add_bandit(self, id, num_of_arms, arm_mean = 0, arm_variance = 100 ):
        '''
        ID: ID number of the bandit to be added to self history
        Bandit_mean/variance: The estimated reward from the bandit, initialized to 0 for uniform Gaussian distribution
        '''
        # Bandit item has tuple (arm visit_count, cumulative_reward)

        self.bandit_interactions[id] = (0, 0)
        
        return False
    
    def show_estimates(self):
        '''
        Return a list of each bandit and assumed win probability
        NOT HELPFUL YET
        '''
        estimates = list()

        for key in self.bandit_interactions:
          if self.bandit_interactions[key][2] != 0:
            estimates.append((key,self.bandit_interactions[key][0]/self.bandit_interactions[key][2]))
          else:
            estimates.append((key, 0))
        return estimates
    
    def get_cum_reward(self):
      '''
      Return the cumulative reward for the agent
      '''
      return self.cum_reward[-1]

if __name__ == '__main__':
    threshold = 0.05
    num_of_arms = 20
    epsilon = 0
    decay = 0.995
    epsilon_end = 0.0
    bandits_count = 10
    episodes = 100
    sample_no = 5

    bandits = {}
    # Change to Bernoulli if desired
    agent = TSGaussian_Agent(bandits_count, num_of_arms)

    best_bandit = None
    cum_best_reward = []
    best_reward = []
    reward = []
    regret = []
    bayesian_regret = 0

    # Initialize the bandits and find which has the best mean
    for i in range(bandits_count):
        # Change to Bernoulli if desired
        x = GaussianBandit(num_of_arms, i, threshold)
        bandits[i] = x
        agent.add_bandit(i, num_of_arms)
        
    for e in range(episodes):
        choice = agent.choose_bandit(sample_no)
        best_arm_list = agent.choose_arm(choice)
        #print(best_arm_list)
        cum_regret = 0
        cum_rew = 0
        rew = 0

        for idx, bandit in enumerate(choice):
            #print(idx, bandit)
            breward = bandits[bandit].sample(best_arm_list[idx])
            agent.update_estimates(breward, bandit, best_arm_list[idx])
            rew += breward

            # Calculate the Regret
            # Sample the best arm - reward and current arm 
            best_arm_reward = bandits[bandit].sample(bandits[bandit].get_best_arm())
            cum_regret += (best_arm_reward - breward)
            cum_rew += best_arm_reward

        regret.append(cum_regret)
        best_reward.append(cum_rew)
        reward.append(rew)
    cum_sum = np.cumsum(regret)
    cum_best_reward = np.cumsum(best_reward)

    # Check our work
    count_correct=0
    cum_reward = np.cumsum(reward)

    # Does not get approximation for every bandit
    # When it gets a correct approximation, it is because agent
    # noticed it gave a higher reward distribution and played it again

    
    # Plot 
    fig = plt.figure()
    
    ax = fig.add_subplot(2,1,1)    
    plt.title("Vanilla Thompson Sampling \n Perfomance of Gaussian Bandits  ")
    plt.plot(range(len(cum_best_reward)), cum_best_reward, label="Optimal Cum Reward") 
    plt.plot(range(len(cum_reward)), cum_reward, label="Agent Cum Reward")
    plt.ylabel('Cumulative Score')
    ax.legend()
    
    bx = fig.add_subplot(2,1,2)
    plt.plot(range(len(cum_sum)), cum_sum, label="Regret")
    plt.ylabel('Regret')
    plt.xlabel('Pull #')
     
    plt.legend(loc='best')
    plt.show()