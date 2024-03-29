# Implement Hierarchical Thompson Sampling

# from thompson_sampling import Bandit


import random as rand
import numpy as np
import matplotlib.pyplot as plt
import hier_ts


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
        self.mu_bar = np.zeros(self.d)
        self.mu_hat = np.zeros((self.num_tasks, self.d))
        self.mu_tilde = np.zeros((self.num_tasks, self.d))
        self.Sigma_q = np.eye(self.d)
        self.sigma0 = 1.0
        self.sigma = 0.5
        self.Sigma_hat = np.zeros((self.num_tasks, self.d, self.d))
        self.Sigma_bar = np.zeros((self.d, self.d))
        self.crs = 1.0  # confidence region scaling
        self.sim_mat = np.zeros((self.num_tasks, self.num_tasks))
        self.task_action_visit = np.zeros((self.num_tasks, self.K))
        self.reward_actions = np.zeros((self.num_tasks, self.K))
        self.R = np.zeros((self.d, self.num_tasks))
        for attr, val in params.items():
            setattr(self, attr, val)

        if not hasattr(self, "Sigma0"):
            self.Sigma0 = np.square(self.sigma0) * np.eye(self.d)

        # hyper-posterior
        self.Sigma_tildes = np.tile(self.Sigma_q, (self.num_tasks, 1, 1))
        self.M = np.ones((self.d, self.num_tasks))/self.num_tasks
        self.gammastar = np.random.normal(self.mu_q, self.Sigma_q)
        # sufficient statistics used in posterior update
        # outer product of features of taken actions in each task
        self.Grams = (np.zeros((self.num_tasks, self.d, self.d)) +
                      1e-6 * np.eye(self.d)[np.newaxis, ...])

        # sum of features of taken actions in each task weighted by rewards
        self.Bs = np.zeros((self.num_tasks, self.d))
        # sum of features of taken actions in each task weighted by rewards
        self.counts = np.zeros(self.num_tasks)

        # Creates the symmetric positive definite matrix Sigma A
        def create_sym_def_pos_mat(size):
            n = size  # size of the matrix
            A = np.random.rand(n, n)  # create random matrix
            A_symm = 0.5 * (A + A.T)  # make matrix symmetric
            eigenvalues = np.linalg.eigvals(A_symm)  # compute eigenvalues
            while not all(e > 0 for e in eigenvalues):
                # if any eigenvalue is not positive, perturb the matrix until it is
                perturbation = abs(np.min(eigenvalues)) + 1e-6
                A_symm += perturbation * np.eye(n)
                eigenvalues = np.linalg.eigvals(A_symm)

            # print("Random symmetric positive definite matrix:")
            # print(A_symm)
            return A_symm
        self.SigmaA = create_sym_def_pos_mat(self.d)

    def create_similarity(self):
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                self.sim_mat[i][j] = np.exp(-0.5 *
                                            np.linalg.norm(
                                                self.meta_data[i] - self.meta_data[j])
                                            )

    def update(self, t, tasks, xs, arms, rs):
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

        ###### G and B Update ######
        # Copying from HierTS, but only relevant parts
        # TODO: pull each arm from each task at least once before updating
        for s, x, arm, r in zip(tasks, xs, arms, rs):
            x_a = x[arm]
            self.Grams[s] += np.outer(x[arm], x[arm]) / np.square(self.sigma)
            # print('Did it go here yet?')
            self.Bs[s] += x[arm] * r / np.square(self.sigma)
            self.counts[s] += 1

        ###### MU AND SIGMA BAR ######
        sum_sigma_bar = 0
        sum_mu_bar = 0
        for s in range(self.num_tasks):
            z = np.linalg.pinv(self.Sigma0 + np.linalg.inv(self.Grams[s]))
            # print(self.Bs.shape, np.linalg.inv(self.Grams[s]).shape, self.sim_mat.shape)
            y = z @ np.linalg.pinv(self.Grams[s]) @ self.Bs[s]
            sum_sigma_bar += z
            sum_mu_bar += y
        # inverse was missing?
        self.Sigma_bar = np.linalg.pinv(
            np.linalg.pinv(self.Sigma_q) + sum_sigma_bar)
        self.mu_bar = self.Sigma_bar @ (np.linalg.pinv(self.Sigma_q)@self.mu_q
                                        + sum_mu_bar)

        ###### SIGMA HAT ######

        for ss in range(self.num_tasks):
            # print(self.Grams[ss].shape, self.Bs[ss].shape)
            # import ipdb
            # ipdb.set_trace()
            self.R[:, ss] = np.linalg.pinv(self.Sigma0 + np.linalg.pinv(self.Grams[ss])
                                           ) @ np.linalg.pinv(self.Grams[ss]) @ (self.Bs[ss])
        for s in tasks:
            sum_sigma_hat = 0
            for ss in range(self.num_tasks):

                # print( np.linalg.inv(self.Grams[ss]).shape) #(self.sim_mat[s][ss]**-2).shape )
                sum_sigma_hat += np.linalg.pinv(self.Sigma0 +
                                                np.linalg.pinv(self.Grams[ss])).dot(self.sim_mat[s][ss]**-2)
            self.Sigma_hat[s] = np.linalg.inv(
                np.linalg.pinv(self.SigmaA) + sum_sigma_hat)
            # print(self.M[tasks].shape,self.Sigma_hat[tasks].dot(np.linalg.inv(self.Sigma0).dot(self.M) + self.R).shape)
            self.M = self.Sigma_hat[s].dot(
                np.linalg.pinv(self.Sigma0).dot(self.M) + self.R)
            self.mu_hat[s] = self.M.dot(self.sim_mat[s])

        ###### MU AND SIGMA TILDE ######

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
        # import ipdb
        # ipdb.set_trace()
        gamma = np.random.multivariate_normal(self.mu_bar, self.Sigma_bar)
        # import ipdb
        # ipdb.set_trace()
        for s, x in zip(tasks, xs):
            # sample mu_s from posterior P
            mu_s = np.random.multivariate_normal(
                self.mu_hat[s], self.Sigma_hat[s])
            # import ipdb
            # ipdb.set_trace()
            # sample theta from posterior for thetas
            mu_prime = self.lamda * gamma + mu_s*(1-self.lamda)
            self.Sigma_tildes[s] = np.linalg.pinv(
                np.linalg.pinv(self.Sigma0) + self.Grams[s])

            self.mu_tilde[s] = self.Sigma_tildes[s] @ (
                np.linalg.pinv(self.Sigma0)@mu_prime + self.Bs[s])

            theta_s = np.random.multivariate_normal(
                self.mu_tilde[s], self.Sigma_tildes[s])
            # posterior sampling
            mu = x.dot(theta_s)
            # Choose the arm with the highest posterior mean
            arms.append(np.argmax(mu))
        return arms, gamma


if __name__ == "__main__":
    alg_spec = ("MetaHierTS", "green", "-")
    num_runs = 100
    num_tasks = 10
    num_tasks_per_round = 5
    n = 200 * num_tasks // num_tasks_per_round

    step = np.arange(1, n + 1)
    sube = (step.size // 10) * np.arange(1, 11) - 1

    for d in [4]:
        K = 5 * d
        for sigma_q_scale in [1.0]:
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
            regretLin = np.zeros((n, num_runs))

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
                    mu_star = np.random.choice(mu_stars.flatten(), size=d)
                    theta = mu_star + sigma_0 * np.random.randn(d)
                    # theta = mu_star
                    # TODO: generate meta-data = theta + noise, noise is 100 times smaller than the variance of theta
                    meta_data = theta + (sigma_0/100) * np.random.randn(d)
                    # meta_data = theta
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
                    "lamda": 0.1
                }

                alg = MetaHierLinTSAgent(num_tasks, K, d, alg_params)
                alg.create_similarity()

                algLin = hier_ts.HierLinTSAgent(num_tasks, K, d, alg_params)

                for t in range(n):
                    tasks = np.random.randint(
                        0, num_tasks, size=num_tasks_per_round)

                    for s in tasks:
                        envs[s].randomize()

                    Xs = [envs[s].X for s in tasks]

                    # Baseline -- HierLinTS
                    armsLin = algLin.get_arm(t, tasks, Xs)
                    rsLin = [envs[s].reward(arm)
                             for s, arm in zip(tasks, armsLin)]
                    algLin.update(t, tasks, Xs, armsLin, rsLin)
                    regretLin[t, run] = np.sum(
                        [envs[s].regret(arm) for s, arm in zip(tasks, armsLin)])

                    # Our algorithm
                    arms, gamma = alg.get_arm(t, tasks, Xs)
                    rs = [envs[s].reward(arm)
                          for s, arm in zip(tasks, arms)]
                    alg.update(t, tasks, Xs, arms, rs)
                    regret[t, run] = np.sum(
                        [envs[s].regret(arm) for s, arm in zip(tasks, arms)])

            cum_regret = regret.cumsum(axis=0)
            cum_regret_Lin = regretLin.cumsum(axis=0)

            plt.plot(step, cum_regret.mean(axis=1),
                     label=alg_spec[0])
            plt.errorbar(step[sube], cum_regret[sube, :].mean(axis=1),
                         cum_regret[sube, :].std(
                axis=1) / np.sqrt(cum_regret.shape[1]),
                fmt="none", ecolor=alg_spec[1])

            plt.plot(step, cum_regret_Lin.mean(axis=1),
                     label="HierTS")
            plt.errorbar(step[sube], cum_regret_Lin[sube, :].mean(axis=1),
                         cum_regret[sube, :].std(
                axis=1) / np.sqrt(cum_regret_Lin.shape[1]),
                fmt="none", ecolor='red')

            print("%s: %.1f +/- %.1f" % (alg_spec[0],
                                         cum_regret[-1, :].mean(),
                                         cum_regret[-1, :].std() / np.sqrt(cum_regret.shape[1])))

    plt.title(r"Linear Bandit (d = %d, $\sigma_q$ = %.3f, num_tasks = %d, num_arms, = %d)" %
              (d, sigma_q_scale, num_tasks, K))
    plt.xlabel("Round t")
    plt.xticks(np.arange(n + 1, step=100))
    plt.ylabel("Regret")
    plt.ylim(bottom=0)
    plt.legend(loc="upper left", frameon=False, prop={'size': 10})

    plt.tight_layout()
    plt.show()
