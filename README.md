# LyceumAI

![](https://github.com/Lyceum/LyceumAI.jl/workflows/CI/badge.svg)

LyceumAI packages tools and frameworks for algorithm development for continous control problems, such as trajectory optimization and reinforcement learning. By using the [LyceumBase interface](https://github.com/Lyceum/LyceumBase.jl) algorithms can be developed agnostic to underlying physics simulators while leveraging the rich Julia ecosystem of packages.


# Algorithms

LyceumAI comes with an on-policy Policy Gradient method called Natural Policy Gradient, referenced in 
[Towards Generalization and Simplicity in Continuous Control](https://arxiv.org/pdf/1703.02660.pdf), and a indirect shooting method of trajectory optimization ment for model predictive control called [Model predictive path integral](https://www.cc.gatech.edu/~bboots3/files/InformationTheoreticMPC.pdf).

Examples of these to methods' use can be found in the [examples](https://docs.lyceum.ml/dev/examples/NPG/).


# Development

Developing new algorithms can be done outside of the LyceumAI package; one can build on the LyceumBase interface and use any LyceumAI tooling necessary. Pull requests are welcome!
