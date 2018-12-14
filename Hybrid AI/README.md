# World Model
   
[Course] https://www.udemy.com/artificial-intelligence-masterclass/   
World Model : https://worldmodels.github.io/   
Evolution Strategies : http://blog.otoro.net/2017/10/29/visual-evolution-strategies/   

![Alt Text](https://github.com/avillemin/SuperDataScience-Courses/blob/master/Hybrid%20AI/world_model_overview.png)

![Alt Text](https://storage.googleapis.com/quickdraw-models/sketchRNN/world_models/assets/world_model_overview.svg)

![Alt Text](https://worldmodels.github.io/assets/vae.svg)

MDN-RNN (M) Model : Mixture Density Network combined with a RNN

![Alt Text](https://worldmodels.github.io/assets/mdn_rnn_new.svg)

Controller (C) Model   
The Controller (C) model is responsible for determining the course of actions to take in order to maximize the expected cumulative reward of the agent during a rollout of the environment. In our experiments, we deliberately make C as simple and small as possible, and trained separately from V and M, so that most of our agent’s complexity resides in the world model (V and M).   

C is a simple single layer linear model that maps z(t) and h(t) directly to action a(t) at each time step:

a(t) = Wc [z(t) h(t)] + bc

In this linear model, Wc and bc are the weight matrix and bias vector that maps the concatenated input vector [z(t) h(t)] to the output action vector a(t).

![Alt Text](https://worldmodels.github.io/assets/world_model_schematic.svg)

![Alt Text](https://worldmodels.github.io/assets/conv_vae_label.svg)
