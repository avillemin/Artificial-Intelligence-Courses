# World Model
   
[Course] https://www.udemy.com/artificial-intelligence-masterclass/   
World Model : https://worldmodels.github.io/   
Evolution Strategies : http://blog.otoro.net/2017/10/29/visual-evolution-strategies/   

![Alt Text](https://github.com/avillemin/SuperDataScience-Courses/blob/master/Hybrid%20AI/world_model_overview.png)

![Alt Text](https://storage.googleapis.com/quickdraw-models/sketchRNN/world_models/assets/world_model_overview.svg)

![Alt Text](https://worldmodels.github.io/assets/vae.svg)

MDN-RNN (M) Model : Mixture Density Network combined with a RNN

<p align="center"><img src="https://worldmodels.github.io/assets/mdn_rnn_new.svg"></p>

Controller (C) Model   
The Controller (C) model is responsible for determining the course of actions to take in order to maximize the expected cumulative reward of the agent during a rollout of the environment. In our experiments, we deliberately make C as simple and small as possible, and trained separately from V and M, so that most of our agent’s complexity resides in the world model (V and M).   

C is a simple single layer linear model that maps z(t) and h(t) directly to action a(t) at each time step:

a(t) = Wc [z(t) h(t)] + bc

In this linear model, Wc and bc are the weight matrix and bias vector that maps the concatenated input vector [z(t) h(t)] to the output action vector a(t).

<p align="center"><img src="https://worldmodels.github.io/assets/world_model_schematic.svg"></p>

<p align="center"><img src="https://worldmodels.github.io/assets/conv_vae_label.svg" width="350" height="600"></p>

# Evolution Strategy

OpenAI published a paper called Evolution Strategies as a Scalable Alternative to Reinforcement Learning where they showed that evolution strategies, while being less data efficient than RL, offer many benefits. The ability to abandon gradient calculation allows such algorithms to be evaluated more efficiently. It is also easy to distribute the computation for an ES algorithm to thousands of machines for parallel computation. By running the algorithm from scratch many times, they also showed that policies discovered using ES tend to be more diverse compared to policies discovered by RL algorithms.


<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/8/8b/Rastrigin_function.png">Two-dimensional Rastrigin function has many local optima</p>

The diagrams below are top-down plots of shifted 2D Schaffer and Rastrigin functions, two of several simple toy problems used for testing continuous black-box optimisation algorithms. Lighter regions of the plots represent higher values of F(x,y). As you can see, there are many local optimums in this function. Our job is to find a set of model parameters (x,y), such that F(x,y) is as close as possible to the global maximum.

Although there are many definitions of evolution strategies, we can define an evolution strategy as an algorithm that provides the user a set of candidate solutions to evaluate a problem. The evaluation is based on an objective function that takes a given solution and returns a single fitness value. Based on the fitness results of the current solutions, the algorithm will then produce the next generation of candidate solutions that is more likely to produce even better results than the current generation. The iterative process will stop once the best known solution is satisfactory for the user.

## Simple Evolution Strategy

One of the simplest evolution strategy we can imagine will just sample a set of solutions from a Normal distribution, with a mean μ and a fixed standard deviation σ. Initially, μ is set at the origin. After the fitness results are evaluated, we set μ to the best solution in the population, and sample the next generation of solutions around this new mean. This is how the algorithm behaves over 20 generations on Schaffer-2D Function and Rastrigin-2D Function:

<p align="center"><img src="http://blog.otoro.net/assets/20171031/schaffer/simplees.gif" width="400" height="400"><img src="http://blog.otoro.net/assets/20171031/rastrigin/simplees.gif" width="400" height="400"></p>

Given its greedy nature, it throws away all but the best solution, and can be prone to be stuck at a local optimum for more complicated problems. It would be beneficial to sample the next generation from a probability distribution that represents a more diverse set of ideas, rather than just from the best solution from the current generation.

## Simple Genetic Algorithm

The idea is quite simple: keep only 10% of the best performing solutions in the current generation, and let the rest of the population die. In the next generation, to sample a new solution is to randomly select two solutions from the survivors of the previous generation, and recombine their parameters to form a new solution. This crossover recombination process uses a coin toss to determine which parent to take each parameter from. In the case of our 2D toy function, our new solution might inherit x or y from either parents with 50% chance. Gaussian noise with a fixed standard deviation will also be injected into each new solution after this recombination process.

<p align="center"><img src="http://blog.otoro.net/assets/20171031/schaffer/simplega.gif" width="400" height="400"><img src="http://blog.otoro.net/assets/20171031/rastrigin/simplega.gif" width="400" height="400"></p>

Genetic algorithms help diversity by keeping track of a diverse set of candidate solutions to reproduce the next generation. However, in practice, most of the solutions in the elite surviving population tend to converge to a local optimum over time. There are more sophisticated variations of GA out there, such as CoSyNe, ESP, and NEAT, where the idea is to cluster similar solutions in the population together into different species, to maintain better diversity over time.

## Covariance-Matrix Adaptation Evolution Strategy (CMA-ES)

A shortcoming of both the Simple ES and Simple GA is that our standard deviation noise parameter is fixed. There are times when we want to explore more and increase the standard deviation of our search space, and there are times when we are confident we are close to a good optima and just want to fine tune the solution. We basically want our search process to behave like this:

<p align="center"><img src="http://blog.otoro.net/assets/20171031/schaffer/cmaes.gif" width="400" height="400"><img src="http://blog.otoro.net/assets/20171031/rastrigin/cmaes.gif" width="400" height="400"></p>

https://en.wikipedia.org/wiki/CMA-ES

CMA-ES an algorithm that can take the results of each generation, and adaptively increase or decrease the search space for the next generation. It will not only adapt for the mean μ and sigma σ parameters, but will calculate the entire covariance matrix of the parameter space. At each generation, CMA-ES provides the parameters of a multi-variate normal distribution to sample solutions from. To see how the covariance matrix is calculated, go to : http://blog.otoro.net/2017/10/29/visual-evolution-strategies/

## Natural Evolution Strategies

Imagine if you had built an artificial life simulator, and you sample a different neural network to control the behavior of each ant inside an ant colony. Using the Simple Evolution Strategy for this task will optimise for traits and behaviours that benefit individual ants, and with each successive generation, our population will be full of alpha ants who only care about their own well-being.  
Instead of using a rule that is based on the survival of the fittest ants, what if you take an alternative approach where you take the sum of all fitness values of the entire ant population, and optimise for this sum instead to maximise the well-being of the entire ant population over successive generations? Well, you would end up creating a Marxist utopia.

A perceived weakness of the algorithms mentioned so far is that they discard the majority of the solutions and only keep the best solutions. Weak solutions contain information about what not to do, and this is valuable information to calculate a better estimate for the next generation.
Many people who studied RL are familiar with the REINFORCE paper. In this 1992 paper, Williams outlined an approach to estimate the gradient of the expected rewards with respect to the model parameters of a policy neural network. This paper also proposed using REINFORCE as an Evolution Strategy, in Section 6 of the paper. This special case of REINFORCE-ES was expanded later on in Parameter-Exploring Policy Gradients (PEPG, 2009) and Natural Evolution Strategies (NES, 2014).

In this approach, we want to use all of the information from each member of the population, good or bad, for estimating a gradient signal that can move the entire population to a better direction in the next generation. Since we are estimating a gradient, we can also use this gradient in a standard SGD update rule typically used for deep learning. We can even use this estimated gradient with Momentum SGD, RMSProp, or Adam if we want to.


<p align="center"><img src="http://blog.otoro.net/assets/20171031/schaffer/pepg.gif" width="400" height="400"><img src="http://blog.otoro.net/assets/20171031/rastrigin/pepg.gif" width="400" height="400"></p>

Lots of Mathematics behind it, see the original website for more information.

I like this algorithm because like CMA-ES, the \sigmaσ’s can adapt so our search space can be expanded or narrowed over time. Because the correlation parameter is not used in this implementation, the efficiency of the algorithm is O(N)O(N) so I use PEPG if the performance of CMA-ES becomes an issue. I usually use PEPG when the number of model parameters exceed several thousand.

## OpenAI Evolution Strategy

In OpenAI’s paper,https://blog.openai.com/evolution-strategies/, they implement an evolution strategy that is a special case of the REINFORCE-ES algorithm outlined earlier. In particular, σ is fixed to a constant number, and only the μ parameter is updated at each generation. Below is how this strategy looks like, with a constant σ parameter:

<p align="center"><img src="http://blog.otoro.net/assets/20171031/schaffer/openes.gif" width="400" height="400"><img src="http://blog.otoro.net/assets/20171031/rastrigin/oes.gif" width="400" height="400"></p>

In addition to the simplification, this paper also proposed a modification of the update rule that is suitable for parallel computation across different worker machines. In their update rule, a large grid of random numbers have been pre-computed using a fixed seed. By doing this, each worker can reproduce the parameters of every other worker over time, and each worker needs only to communicate a single number, the final fitness result, to all of the other workers. This is important if we want to scale evolution strategies to thousands or even a million workers located on different machines, since while it may not be feasible to transmit an entire solution vector a million times at each generation update, it may be feasible to transmit only the final fitness results. In the paper, they showed that by using 1440 workers on Amazon EC2 they were able to solve the Mujoco Humanoid walking task in ~ 10 minutes.


Although ES might be a way to search for more novel solutions that are difficult for gradient-based methods to find, it still vastly underperforms gradient-based methods on many problems where we can calculate high quality gradients. For instance, only an idiot would attempt to use a genetic algorithm for image classification. But sometimes such people do exist in the world, and sometimes these explorations can be fruitful!

# Variational Auto-Encoder

![Alt text](https://github.com/avillemin/SuperDataScience-Courses/blob/master/Hybrid%20AI/VAR.png)

![Alt text](https://github.com/avillemin/SuperDataScience-Courses/blob/master/Hybrid%20AI/reparameterization_trick.png)

# Mixture Density Networks

Instead of having a neural network, the MDN gives us a distribution of possible values of the output. To do so, the network is going to output the mean and the standard deviation of the normal distribution.

![Alt text](https://github.com/avillemin/SuperDataScience-Courses/blob/master/Hybrid%20AI/MDN.png)

But what if the distribution is not a normal distribution ?
Any general distribution can be broken down into a mixture of normal distributions.

![Alt text](https://github.com/avillemin/SuperDataScience-Courses/blob/master/Hybrid%20AI/MDN2.png)

With alpha the weights of each distribution. The sum of all the alphas should be equal to 1. To do so, we are going to apply a softamx function.
