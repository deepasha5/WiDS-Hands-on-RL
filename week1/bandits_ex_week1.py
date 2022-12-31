import numpy as np 
import math 
import matplotlib.pyplot as plt 
from bandits import bandit
from bandits import bandit_arm
from multiprocessing import Pool
class Algorithm:
    def __init__(self, num_arms):
        self.num_arms = num_arms
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError


    # Implement the epsilon greedy algorithm 
# as a child class of the Algorithm class
class eps_greedy(Algorithm):
    def __init__(self,num_arms):
        super().__init__(num_arms)
        self.eps = 0.3
        self.countarr = np.zeros(num_arms)
        self.rewarr = np.zeros(num_arms)
        self.num_arms = num_arms
        # write the necessary data structures 

    def give_pull(self):
        rd= np.random.random()
        if  rd< self.eps:
            return np.random.choice(self.num_arms, 1)[0]

        else:
            div = np.zeros(self.num_arms)
            for i in range(self.num_arms):
                if self.countarr[i] !=0 :   
                    div[i] = self.rewarr[i]/self.countarr[i]
                else:
                    div[i]= 0
            
            return np.argmax(div)
            # bandit.arms.index(bandit_arm(bandits.max_mean))
        # write the code to give a pull via the epsilon greedy algo 
    
    def get_reward(self,index,reward):
        self.countarr[index] = self.countarr[index] +1
        self.rewarr[index] = self.rewarr[index] +reward
        
        # update the data structures based on the reward received

class UCB(Algorithm):
    def __init__(self, num_arms):
        super().__init__(num_arms)
        self.countarr = np.zeros(num_arms)
        self.rewarr = np.zeros(num_arms)
        self.num_arms= num_arms
        # write the necessary data structures here
    
    def give_pull(self):
        div = np.zeros(self.num_arms)
        for i in range(self.num_arms):
                if self.countarr[i] !=0 :   
                    div[i] = self.rewarr[i]/self.countarr[i]
                else :
                    div[i] = 0
        avg = div
        extra_term = np.zeros(self.num_arms)
        for i in range(self.num_arms):
            #extra_term.append(np.sqrt(2*np.log(np.sum(countarr))/countarr[i]))
            if self.countarr[i] !=0 :
                extra_term[i]= (np.sqrt(2*np.log(np.sum(self.countarr))/self.countarr[i]))
            else:
                #extra_term[i]= 0
                return i
        return np.argmax(np.add(avg, extra_term))
        # write the code to give a pull via the epsilon greedy algo 
    
    def get_reward(self,index,reward):
        self.countarr[index] = self.countarr[index] +1
        self.rewarr[index] = self.rewarr[index] +reward
        # update the data structures based on the reward received       

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms):
        super().__init__(num_arms)
        #we write the required data structures for this algorithms
        self.successes = np.zeros(num_arms)
        self.failures = np.zeros(num_arms)
    
    def give_pull(self):
        # we give a rule according to the algorithm to sample the bandit instance
        beta = [np.random.beta(self.successes[bandit]+1,self.failures[bandit] +1) for bandit in range(self.num_arms)]
        return np.argmax(beta)
    
    def get_reward(self, arm_index, reward):
        # we update the data structures as we see the reward received
        self.successes[arm_index] += reward
        self.failures[arm_index] += (1-reward)

def plot_avg_reward(algo,horizon,averaging=100):
        average_reward = np.zeros(horizon)
        for j in range(averaging):
            np.random.seed(0)
            bandit_instance = bandit([0.2,0.1,0.6,0.1]) # use this bandit instance only
            thompson_instance = algo(4) # do this for the other two algorithms as well
            rewards = [] 
            for i in range(horizon):
                arm_to_pull = thompson_instance.give_pull() #get the arm to pull _From the ALGORITHM_
                
                reward = bandit_instance.pull(arm_to_pull) #get the (stochastic) reward _from the BANDIT INSTANCE_
                thompson_instance.get_reward(arm_to_pull,reward) #update the internal data structures of the algorithm
                rewards.append(bandit_instance.avg_reward)
            average_reward = average_reward + rewards


        average_reward = average_reward/averaging 

        plt.plot(average_reward)
        plt.title("Average Reward")


def main():
   
        plot_avg_reward(Thompson_Sampling,1000)
        plot_avg_reward(eps_greedy,1000) #the eps-greedy code was removed after plotting this, try to write your own and see what kind of plots you get
        plt.legend(["Thomspon","e-greedy"]) 
        plt.show()

# __name__
if __name__=="__main__":
    main()