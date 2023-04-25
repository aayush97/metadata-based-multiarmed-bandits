# Implement Hierarchical Thompson Sampling

from thompson_sampling import Bandit


import random as rand
import numpy as np
import matplotlib.pyplot as plt


class LinBandit(object):
    """Linear bandit."""

    def __init__(self, X, theta, noise="normal", sigma=0.5):
        self.X = np.copy(X)
        self.K = self.X.shape[0]  # number of arms/actions
        self.d = self.X.shape[1]  # dimension of the context
        self.theta = np.copy(theta)  # true parameter
        self.noise = noise
        if self.noise == "normal":
            self.sigma = sigma
        self.mu = self.X.dot(self.theta)
        self.best_arm = np.argmax(self.mu)
        # ipdb.set_trace()
        self.randomize()

    def randomize(self):
        # generate random rewards
        if self.noise == "normal":
            self.rt = self.mu + self.sigma * np.random.randn(self.K)
        elif self.noise == "bernoulli":
            self.rt = (np.random.rand(self.K) < self.mu).astype(float)
        elif self.noise == "beta":
            self.rt = np.random.beta(4 * self.mu, 4 * (1 - self.mu))

    def reward(self, arm):
        # instantaneous reward of the arm
        return self.rt[arm]

    def regret(self, arm):
        # instantaneous regret of the arm
        return self.rt[self.best_arm] - self.rt[arm]

    def pregret(self, arm):
        # expected regret of the arm
        return self.mu[self.best_arm] - self.mu[arm]

    def print(self):
        if self.noise == "normal":
            return "Linear bandit: %d dimensions, %d arms" % \
                (self.d, self.K)
        elif self.noise == "bernoulli":
            return "Bernoulli linear bandit: %d dimensions, %d arms" % \
                (self.d, self.K)
        elif self.noise == "beta":
            return "Beta linear bandit: %d dimensions, %d arms" % \
                (self.d, self.K)


class MetaHierLinTSAgent(object):
    def __init__(self, num_tasks, K, d, params):
        # meta data is numpy array with dimensions (num_tasks, d)
        self.meta_data = params["metadata"]
        self.num_tasks = num_tasks
        self.K = K
        self.d = d
        self.mu_q = np.zeros(self.d)
        self.mu_bar = np.zeros(self.num_tasks)
        self.mu_hat = np.zeros(self.num_tasks, self.K)
        self.mu_tilde = np.zeros(self.num_tasks)
        self.Sigma_q = np.eye(self.d)
        self.sigma0 = 1.0
        self.sigma = 0.5
        self.sigma_hat = np.zeros(self.num_tasks, self.K, self.K)
        self.sigma_bar = np.zeros(self.num_tasks, self.num_tasks)
        self.sigma_tilde = np.zeros(self.num_tasks)

        self.crs = 1.0  # confidence region scaling
        self.sim_mat = np.zeros(self.num_tasks, self.num_tasks)
        self.task_action_visit = np.zeros(self.num_tasks, self.K)
        self.reward_actions = np.zeros(self.num_tasks, self.K)
        for attr, val in params.items():
            setattr(self, attr, val)

        if not hasattr(self, "Sigma0"):
            self.Sigma0 = np.square(self.sigma0) * np.eye(self.d)

        # hyper-posterior
        self.mu_tildes = np.tile(self.mu_q, (self.num_tasks, 1))
        self.Sigma_tildes = np.tile(self.Sigma_q, (self.num_tasks, 1, 1))
        self.M = np.ones(self.num_tasks, self.K) / self.K
        self.gammastar = np.random.normal(self.mu_q, self.Sigma_q)
        # sufficient statistics used in posterior update
        # outer product of features of taken actions in each task
        self.Grams = (np.zeros((self.num_tasks, self.d, self.d)) +
                      1e-6 * np.eye(self.d)[np.newaxis, ...])
        # sum of features of taken actions in each task weighted by rewards
        self.Bs = np.zeros((self.num_tasks, self.d))
        self.counts = np.zeros(self.num_tasks)

    def create_similarity(self):
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                self.sim_mat[i][j] = np.exp(-0.5 *
                                            np.linalg.norm(
                                                self.meta_data[i] - self.meta_data[j])
                                            )

    def update(self, t, tasks, xs, action, rewards, gamma):
        # TODO: update hyper-posterior and posterior
        '''        #alg_params = {
                    "mu_q": np.copy(mu_q),
                    "Sigma_q": np.copy(Sigma_q),
                    "Sigma0": np.copy(Sigma_0),
                    "sigma": sigma,
                    "metadata": np.array(meta_data_list),
                }
                alg = MetaHierLinTSAgent(
                    num_tasks, K, d, alg_params)
        '''

        # Step : update mu_bar and sigma_bar
        sum_states = 0
        sum_sigma = 0
        for i in range(self.num_tasks):
            for a in range(self.K):
                x = self.reward_actions[i][a] / \
                    (self.task_action_visits[i][a] *
                     self.sigma0**2 + self.sigma**2)
                y = self.task_action_visits[i][a]/(
                    self.task_action_visits[i][a]*self.sigma0**2 + self.sigma**2)
                sum_sigma += y
                sum_states += x

        self.sigma_bar = (self.Sigma_q ** -2 + y)**-1
        self.mu_bar = self.sigma_bar * (self.mu_q/self.Sigma_q**2 + sum_states) # possibly index at some dimension

        sigmaA = 1

        # Step : update M
        for other_tasks in range(self.num_tasks):
            self.M[tasks][other_tasks] = self.sigma_bar *\
                (self.M[tasks][other_tasks]/sigmaA
                 + self.reward_actions[tasks][other_tasks] /
                 self.task_action_visits[tasks][other_tasks]*self.sigma0**2 + self.sigma**2)

        # Update mu_hat and sigma_hat
        sum_sigma_hat = 0
        # Step : Update mu^ and sigma^
        for s in range(self.num_tasks):
            for a in range(self.K):
                sum_sigma_hat += self.task_action_visit[s][a]/(self.task_action_visits[s][a]*self.sigma0**2
                                                               + self.sigma**2 * self.sim_mat[tasks][s])

        self.sigma_hat = sigmaA ** -2 + sum_sigma_hat

        self.mu_hat = np.transpose(self.sim_mat[tasks]) * self.M
        
        # Step : Sample from gamma_st and new mu_st to get theta s*
        # mu_tilde will be a vector of dimension (num_tasks, K) action space
        self.sigma_tilde = (1/self.sigma0**2 + self.task_action_visit[tasks][action] / self.sigma * 2)**-1
        self.mu_tilde = self.sigma_tilde * ((self.lamda * gamma + self.mu[tasks]*(1-self.lamda)) /
                                self.sigma0**2 + self.reward_actions[tasks][action]/sigma**2)

        return

    def get_arm(self, t, tasks, xs):
        # xs is a list of feature vectors of shape (num_tasks_per_round, K, d)
        # xs[s] is a feature vector of shape (K, d) which is th
        # feature vector of the K actions for s-th task

        # if there are 5 tasks and 3 arms per task
        # tasks[0] = 1
        # xs[0] = [[d-dimensional vector], [d-dimensional vector], [d-dimensional vector]]
        # len(tasks) == len(xs)

        arms = []
        # sample gamma from posterior Q
        gamma = np.random.multivariate_normal(self.mu_bar, self.Sigma_bar)
        for s, x in zip(tasks, xs):
            # sample mu_s from posterior P
            mu_s = np.random.multivariate_normal(self.mu_hat, self.Sigma_hat)
            # sample theta from posterior for thetas
            theta_s = np.random.multivariate_normal(
                (1-self.lamda) * gamma + mu_s * self.lamda, self.Sigma_tilde)
            # posterior sampling
            mu = x.dot(theta_s)
            # TODO: Ask @jhanvi and @fatemeh
            # choose the arm with the highest posterior mean?
            # or choose the arm with the highest posterior sample?
            arms.append(np.argmax(mu))
        return arms


if __name__ == "__main__":
    alg_spec = ("HierTS", "green", "-")
    num_runs = 100
    num_tasks = 10
    num_tasks_per_round = 5
    n = 200 * num_tasks // num_tasks_per_round

    step = np.arange(1, n + 1)
    sube = (step.size // 10) * np.arange(1, 11) - 1

    for d in [1]:
        K = 5 * d
        for sigma_q_scale in [0.5, 1]:
            # meta-prior parameters
            mu_q = np.zeros(d)
            # prior parameters
            sigma_0 = 0.1
            # reward noise
            sigma = 1

            Sigma_q = np.square(sigma_q_scale) * np.eye(d)
            Sigma_0 = np.square(sigma_0) * np.eye(d)

            plt.figure(figsize=(10, 6))

            # for alg_spec in alg_specs:
            regret = np.zeros((n, num_runs))

            for run in range(num_runs):
                # true hyper-prior
                # TODO: sample multiple mu_stars for a cluster of tasks

                num_clusters = 3
                mu_stars = np.zeros((num_clusters, d))
                # Get our mu_stars
                for i in range(num_clusters):
                    mu_stars[i] = mu_q + sigma_q_scale * np.random.randn(d)
                envs = []
                meta_data_list = []
                for _ in range(num_tasks):
                    # sample problem instance from N(\mu_*, \sigma_0^2 I_d)
                    # randomly assign a task to a cluster
                    mu_star = np.random.choice(mu_stars)

                    theta = mu_star + sigma_0 * np.random.randn(d)
                    # TODO: generate meta-data = theta + noise, noise is 100 times smaller than the variance of theta
                    meta_data = theta + (sigma_0/100) * np.random.randn(d)
                    meta_data_list.append(meta_data)

                    # sample arms from a unit ball
                    X = np.random.randn(K, d)
                    X /= np.linalg.norm(X, axis=-1)[:, np.newaxis]
                    envs.append(LinBandit(X, theta, sigma=sigma))

                # TODO: see if this is the problem later
                # we took out the two lines below from outside the for loop
                alg_params = {
                    "mu_q": np.copy(mu_q),
                    "Sigma_q": np.copy(Sigma_q),
                    "Sigma0": np.copy(Sigma_0),
                    "sigma": sigma,
                    "metadata": np.array(meta_data_list),
                }
                alg = MetaHierLinTSAgent(
                    num_tasks, K, d, alg_params)

                for t in range(n):
                    tasks = np.random.randint(
                        0, num_tasks, size=num_tasks_per_round)

                    for s in tasks:
                        envs[s].randomize()

                    Xs = [envs[s].X for s in tasks]
                    arms = alg.get_arm(t, tasks, Xs)
                    rs = [envs[s].reward(arm)
                          for s, arm in zip(tasks, arms)]
                    alg.update(t, tasks, Xs, arms, rs)
                    regret[t, run] = np.sum(
                        [envs[s].regret(arm) for s, arm in zip(tasks, arms)])

            cum_regret = regret.cumsum(axis=0)
            plt.plot(step, cum_regret.mean(axis=1),
                     label=alg_spec[0])
            plt.errorbar(step[sube], cum_regret[sube, :].mean(axis=1),
                         cum_regret[sube, :].std(
                axis=1) / np.sqrt(cum_regret.shape[1]),
                fmt="none", ecolor=alg_spec[1])

            print("%s: %.1f +/- %.1f" % (alg_spec[0],
                                         cum_regret[-1, :].mean(),
                                         cum_regret[-1, :].std() / np.sqrt(cum_regret.shape[1])))

    plt.title(r"Linear Bandit (d = %d, $\sigma_q$ = %.3f)" %
              (d, sigma_q_scale))
    plt.xlabel("Round t")
    plt.xticks(np.arange(n + 1, step=100))
    plt.ylabel("Regret")
    plt.ylim(bottom=0)
    plt.legend(loc="upper left", frameon=False, prop={'size': 10})

    plt.tight_layout()
    plt.show()
