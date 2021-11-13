/*
Copyright (C) 2021 Intel Corporation
SPDX-License-Identifier:  BSD-3-Clause
*/

#include <torch/script.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <iostream>
#include <memory>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DEVICE(x, y) AT_ASSERTM(x.device().index() == y.device().index(), #x " and " #y " must be in same CUDA device")

using namespace torch::indexing;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

template <class T>
__global__ void AThDynamicsFwdKernel(
    T* __restrict__ threshold,
    T* __restrict__ refractory,
    const T* __restrict__ input,
    const T* __restrict__ ref_state,
    const T* __restrict__ ref_decay,
    const T* __restrict__ th_state,
    const T* __restrict__ th_decay,
    const int th_scale,
    const int th0,
    const int w_scale,
    const int num_neurons,
    const int neurons_per_batch,
    const int num_decays, // this determines individual, channelwise or shared decay
    const int decay_block, // if decay_block==1 then its individual neuron
    const int num_steps
) {
    unsigned neuron_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(neuron_id >= num_neurons)    return;

    int th = th_state[neuron_id] * w_scale;
    int th_step = th_scale;
    int th_decay_int;
    int ref_new = ref_state[neuron_id] * w_scale;
    int ref_decay_int;
    int linear_id;  

    if(num_decays > 1) {  
        // individual decays or channelwise decay
        // num_decays * decay_block == num_neurons
        th_decay_int = (1<<12) - th_decay[(neuron_id) % neurons_per_batch / decay_block];
        ref_decay_int = (1<<12) - ref_decay[(neuron_id) % neurons_per_batch / decay_block];
    } else { // shared decays
        th_decay_int = (1<<12) - th_decay[0];
        ref_decay_int = (1<<12) - ref_decay[0];
    }

    // if(neuron_id == 0)  printf("int: %d bytes\n", sizeof(int));

    for(int n=0; n<num_steps; ++n) {
        linear_id = n + neuron_id * num_steps;

        ref_new = (ref_new * ref_decay_int) >> 12;
        th = (((th - th0) * th_decay_int) >> 12) + th0;
        
        threshold[linear_id] = 1.0f * th / w_scale;
        refractory[linear_id] = 1.0f * ref_new / w_scale;
        
        if(int(w_scale * input[linear_id]) >= (th + ref_new)) {
            ref_new += 2*th;
            th += th_step;
        } 
    }
}

variable_list AThDynamicsFwd(
    const Variable input,
    const Variable ref_state,
    const Variable ref_decay,
    const Variable th_state,
    const Variable th_decay,
    float th_scale,
    float th0,
    int w_scale
) {
    // make sure all the inputs are contigious and in same device
    CHECK_INPUT(input);
    CHECK_INPUT(th_state);
    CHECK_INPUT(th_decay);
    CHECK_INPUT(ref_state);
    CHECK_INPUT(ref_decay);
    CHECK_DEVICE(input, th_state);
    CHECK_DEVICE(input, th_decay);
    CHECK_DEVICE(input, ref_state);
    CHECK_DEVICE(input, ref_decay);
    cudaSetDevice(input.device().index());

    auto device = input.device().type();
    auto dtype = input.dtype();
    auto threshold = torch::zeros_like(input);
    auto refractory = torch::zeros_like(input);

    int num_neurons = input.numel() / input.size(-1);
    int thread = 256;
    int block = ceil(1.0f * num_neurons / thread);

    // std::cout << "num_neurons : " << num_neurons << std::endl
    //           << "thread : " << thread << std::endl
    //           << "block : " << block << std::endl;
    AThDynamicsFwdKernel<float><<< block, thread >>>(
        threshold.data<float>(),
        refractory.data<float>(),
        input.data<float>(), 
        ref_state.data<float>(),
        ref_decay.data<float>(),
        th_state.data<float>(),
        th_decay.data<float>(),
        th_scale*w_scale, th0*w_scale, w_scale, num_neurons, num_neurons / input.size(0),
        th_decay.numel(), // num_th_decays 
        num_neurons / th_decay.numel() / input.size(0), // decay_block 
        input.size(-1) // num_steps
    );
    // cudaDeviceSynchronize();

    return {threshold, refractory};
}

class AThDynamics : public torch::autograd::Function<AThDynamics> {
    public:
    static variable_list forward(
        AutogradContext* ctx, 
        const Variable input,
        const Variable ref_state,
        const Variable ref_decay,
        const Variable th_state,
        const Variable th_decay,
        float th_scale,
        float th0,
        int w_scale
    ) {
        auto result = AThDynamicsFwd(input, ref_state, ref_decay, th_state, th_decay, th_scale, th0, w_scale);
        return result;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_output) {
        // No backward gradients from here
        return {
            torch::Tensor(), // Gradients of non-tensor arguments to forward must be `torch::Tensor()`.
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor() 
        };
    }
};

std::vector<torch::Tensor> AThDynamicsFx(
    const torch::Tensor& input,
    const torch::Tensor& ref_state,
    const torch::Tensor& ref_decay,
    const torch::Tensor& th_state,
    const torch::Tensor& th_decay,
    float th_scale,
    float th0,
    int w_scale
) {
    auto result = AThDynamics::apply(
            input, 
            ref_state, ref_decay, 
            th_state, th_decay, th_scale, th0, 
            w_scale
        );
    return {result[0], result[1]};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("dynamics", &AThDynamicsFx, "Dynamics of Loihi Adaptive Threshold.");
    m.def("fwd", &AThDynamicsFwd, "Fwd dynamics of Loihi Adaptive Threshold.");
    // m.def("bwd", &AThDynamicsBwd, "Bwd dynamics of Loihi Adaptive Threshold.");
}