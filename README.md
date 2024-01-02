# MetaHierTS: Meta Hierarchical Thompson Sampling

## Introduction
MetaHierTS is a novel recommendation system algorithm aimed at enhancing user experiences in online marketing. Developed by Aayush Gautam, Jhanvi Garg, Fatemeh Doudi, and Andrew Porter, this algorithm focuses on leveraging metadata and similarities between tasks to optimize decision-making in a multi-task Multi-Armed Bandit (MAB) environment.
![Advertisement Motivation (1)](https://github.com/aayush97/bandits-and-attention/assets/24679505/c06bcfdd-0a44-4a6f-adc6-6e044962a358)

## Motivation
- Efficient exploration techniques in recommendation systems.
- Utilizing metadata and task similarities to improve decision-making.
- Enhancing the quality and effectiveness of recommendation systems.

## Key Features
- **Algorithm Design:** MetaHierTS learns appropriate actions for each task, minimizing cumulative regret.
- **Utilization of Metadata:** Incorporates task metadata and similarities for better selection decisions.
- **Learning Process:** Employs information gained from each task, weighting similar tasks to accelerate learning.
- **Experimental Validation:** Demonstrates superior performance over Hierarchical Thompson Sampling Model through extensive experiments with Gaussian bandits.

## Setting
- The algorithm operates in an environment where actions, rewards, and task parameters follow specific distributions.
- Utilizes a similarity matrix to weigh the importance of tasks and their correlations.

## Algorithm
1. **Initialization:** Setting up the prior distributions for the tasks.
2. **Task Selection:** Randomly choosing tasks and sampling from posterior distributions.
3. **Reward Sampling:** Sampling rewards for actions and updating task histories.
4. **Posterior Updates:** Updating the posterior distributions based on task interactions.

## Regret Bound Analysis
- Provides a theoretical regret bound for the sequential setting.
- Shows linear scaling with the number of tasks and sublinear scaling with the number of interactions per task.
![image](https://github.com/aayush97/bandits-and-attention/assets/24679505/43b3fd03-ad23-4583-9d47-7abf0a9d8b3c)

## Experiments
- Simulated environment to test the algorithm.
- Compared MetaHierTS with Hierarchical Thompson Sampling, demonstrating improved performance.

## Conclusion and Future Directions
- MetaHierTS offers a significant improvement in utilizing metadata for multi-task bandits.
- Future work includes improving theoretical regret bounds and testing with real-world datasets.

## Acknowledgements
Special thanks to Nilson Chapagain for his insights and Prof. Dileep for guidance in literature.

## Contributions
- **Problem Formulation:** Jhanvi Garg, Fatemeh Doudi, Aayush Gautam, Andrew Porter.
- **Algorithm Design and Analysis:** Jhanvi Garg, Fatemeh Doudi.
- **Data Generation and Implementation:** Aayush Gautam, Andrew Porter.

---

For more detailed information, refer to the complete paper and slide deck provided by the authors.
