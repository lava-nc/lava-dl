# Lava-DL Bootstrap

`lava.lib.dl.bootstrap` accelerates rate coded Spiking Neural Network (SNN) training using by dynamically estimating the equivalent ANN transfer function of a spiking layer with a picewise linear model at regular interval and using the ANN equivlent network to train the original SNN. 

**Highlight features**

* Accelerated rate coded SNN training.
* Low latency inference of trained SNN made possible by close modeling of equivalent ANN dynamics.
* Hybrid training with a mix of SNN layers and ANN layers for minimal drop in SNN accuracy.
* Scheduler for seamless switching between different bootstrap modes.

## Bootstrap Training

The underlying principle for ANN-SNN conversion is that the ReLU activation function (or similar form) approximates the firing rate of an LIF spiking neuron. Consequently, an ANN trained with ReLU activation can be mapped to an equivalent SNN with proper scaling of weights and thresholds. However, as the number of time-steps reduces, the alignment between ReLU activation and LIF spiking rate falls apart mainly due to the following two reasons (especially, for discrete-in-time models like Loihi’s CUBA LIF):

![fit](https://user-images.githubusercontent.com/29907126/140595166-336e625d-c269-40d6-af85-caf5d2328139.png)

* With less time steps, the SNN can assume only a few discrete firing rates.
* Limited time steps mean that the spiking neuron activity rate often saturates to maximum allowable firing rate.

In Bootstrap training. An SNN is used to jumpstart an equivalent ANN model which is then used to accelerate SNN training. There is no restriction on the type of spiking neuron or it's reset behavior. It consists of following steps:

<p align="center">
<img src="https://user-images.githubusercontent.com/29907126/140595174-2feb6946-bf64-4188-a6ea-eeb693a3052d.png" alt="Drawing" style="max-height: 300px;"/>
</p>

* Input output data points are first collected from the network running as an SNN: **`bootstrap.mode.SNN`**.
* The data is used to estimate the corresponding ANN activation as a piecewise linear layer, unique to each layer: **``bootstrap.mode.FIT``** mode.
* The training is accelerated using the piecewise linear ANN activation: **``bootstrap.mode.ANN``** mode.
* The network is seamlessly translated to an SNN: **``bootstrap.mode.SNN``** mode.
* SAMPLE mode and FIT mode are repeated for a few iterations every couple of epochs, thus maintaining an accurate ANN estimate.

## Hybridization

With `bootstrap.block` interface, some of the layers in the network can be run in SNN and rest in ANN. We define **crossover** layer which splits layers earlier than it to always SNN and rest to ANN-SNN bootstrap mode.

<p align="center">
<img src="https://user-images.githubusercontent.com/29907126/140595438-142a68a5-83be-4131-a979-5c7b750b1055.png" alt="Drawing" style="max-height: 250px;"/>
</p>

## Tutorials

* [MNIST digit classification](dummy_link) TODO: UPDATE LINK

## Modules
The main modules are 

### `bootstrap.block`
It provides `lava.lib.dl.slayer.block` based network definition interface.

### `bootstrap.ann_sampler`
It provides utilities for sampling SNN data points and pievewise linear ANN fit.

### `bootstrap.routine`
`bootstrap.routine.Scheduler` provides an easy scheduling utility to seamlessly switch between SAMPLING | FIT | ANN | SNN mode. It also provides ANN-SNN bootstrap **hybrid traiing** utility as well determined by crossover point.
