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
__global__ void LIDynamicsFwdKernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const T* __restrict__ decay,
    const T* __restrict__ state,
    const int threshold, // this must be scaled by w_scale
    const int w_scale,
    const int num_neurons,
    const int neurons_per_batch,
    const int num_decays, // this determines individual, channelwise or shared decay
    const int decay_block, // if decay_block==1 then its individual neuron
    const int num_steps
) {
    unsigned neuron_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(neuron_id >= num_neurons)    return;

    int output_old = state[neuron_id] * w_scale;
    int output_new, output_sign, decay_int, decayed_output;
    int linear_id;

    if(num_decays > 1) {  
        // individual decays or channelwise decay
        // num_decays * decay_block == num_neurons
        decay_int = (1<<12) - decay[(neuron_id) % neurons_per_batch / decay_block];
    } else { // shared decays
        decay_int = (1<<12) - decay[0];
    }

    // if(neuron_id == 0)  printf("int: %d bytes\n", sizeof(int));

    for(int n=0; n<num_steps; ++n) {
        linear_id = n + neuron_id * num_steps;

        output_sign = (output_old >= 0) ? 1 : -1;

        // anything larger than 524,287 = 0x7FFFF in magnitude will potentially cause value overflow and change in sign
        // calculate the decay in double and convert it to int
        decayed_output = int(1.0l * output_sign * output_old * decay_int / 4096);

        output_new = output_sign * decayed_output + int(w_scale * input[linear_id]);
        
        output[linear_id] = 1.0f * output_new / w_scale;
        
        if(threshold >= 0 && output_new >= threshold) { // if threshold <0, then there is no spike and reset dynamics
            output_old = 0;
        } else {
            output_old = output_new;
        }
    }
}

template <class T>
__global__ void LIDynamicsBwdKernel(
    T* __restrict__ grad_input_tensor,
    const T* __restrict__ grad_output,
    const T* __restrict__ decay_tensor,
    const int num_neurons,
    const int neurons_per_batch,
    const int num_decays, // this determines individual, channelwise or shared decay
    const int decay_block, // if decay_block==1 then its individual neuron
    const int num_steps
) {
    unsigned neuron_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(neuron_id >= num_neurons)    return;

    float decay;
    float grad_input = 0;

    if(num_decays > 1) {  
        // individual decays or channelwise decay
        // num_decays * decay_block == num_neurons
        decay = 1 - decay_tensor[(neuron_id) % neurons_per_batch / decay_block] / (1<<12);
    } else { // shared decays
        decay = 1 - decay_tensor[0] / (1<<12);
    }

    int linear_id;

    for(int n=num_steps-1; n>=0; --n) {
        linear_id = n + neuron_id * num_steps;
        grad_input = decay * grad_input + grad_output[linear_id];
        grad_input_tensor[linear_id] = grad_input;
    }
}

variable_list LIDynamicsFwd(
    const Variable input,
    const Variable decay,
    const Variable state,
    float threshold,
    int w_scale
) {
    // make sure all the inputs are contigious and in same device
    CHECK_INPUT(input);
    CHECK_INPUT(decay);
    CHECK_INPUT(state);
    CHECK_DEVICE(input, decay);
    CHECK_DEVICE(input, state);
    cudaSetDevice(input.device().index());

    auto device = input.device().type();
    auto dtype = input.dtype();
    auto output = torch::zeros_like(input);

    threshold *= w_scale;

    int num_neurons = input.numel() / input.size(-1);
    int thread = 256;
    int block = ceil(1.0f * num_neurons / thread);

    // std::cout << "num_neurons : " << num_neurons << std::endl
    //           << "thread : " << thread << std::endl
    //           << "block : " << block << std::endl;
    LIDynamicsFwdKernel<float><<< block, thread >>>(
        output.data<float>(),
        input.data<float>(), 
        decay.data<float>(), 
        state.data<float>(),
        threshold, w_scale, num_neurons, num_neurons / input.size(0),
        decay.numel(), // num_decays 
        num_neurons / decay.numel() / input.size(0), // decay_block 
        input.size(-1) // num_steps
    );
    // cudaDeviceSynchronize();

    return {output};
}

variable_list LIDynamicsBwd(
    const Variable grad_output,
    const Variable output,
    const Variable decay
) {
    // make sure all the inputs are contigious
    CHECK_INPUT(grad_output);
    CHECK_INPUT(output);
    CHECK_INPUT(decay);
    // // Since these variables come from autograd context, they are ensured to be on same device
    // CHECK_DEVICE(grad_output, output);
    // CHECK_DEVICE(grad_output, decay);
    cudaSetDevice(grad_output.device().index());

    auto grad_input = torch::zeros_like(grad_output);

    int num_neurons = grad_output.numel() / grad_output.size(-1);
    int thread = 256;
    int block = ceil(1.0f * num_neurons / thread);

    LIDynamicsBwdKernel<float><<< block, thread >>>(
        grad_input.data<float>(),
        grad_output.data<float>(),
        decay.data<float>(),
        num_neurons, num_neurons / grad_output.size(0),
        decay.numel(), // num_decays 
        num_neurons / decay.numel() / grad_output.size(0), // decay_block 
        grad_output.size(-1) // num_steps
    );
    // cudaDeviceSynchronize();
    
    // It makes sense to use torch primitives for the following calculations
    // Custom cuda implementation would need reimplementing tensor primitives
    auto grad_decay = grad_input.index({"...", Slice(1, None)}) * output.index({"...", Slice(None, -1)});

    if(numel(decay) == 1) {
        grad_decay = torch::sum(grad_decay.flatten(), 0, true);
    } else {
        grad_decay = torch::sum(grad_decay, c10::IntArrayRef({0, -1}));
        if(grad_decay.ndimension()!=1) {
            grad_decay = torch::sum(grad_decay.reshape({grad_decay.size(0), -1}), 1);
        }
    }

    return {grad_input, grad_decay};
}


class LIDynamics : public torch::autograd::Function<LIDynamics> {
    public:
    static variable_list forward(
        AutogradContext* ctx, 
        const Variable input,
        const Variable decay,
        const Variable state,
        float threshold,
        int w_scale
    ) {
        auto result = LIDynamicsFwd(input, decay, state, threshold, w_scale);
        ctx->save_for_backward({result[0], decay});
        return result;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_output) {
        // expand saved variables and parameters
        auto saved = ctx->get_saved_variables();
        auto output = saved[0];
        auto decay  = saved[1];

        auto grads = LIDynamicsBwd(grad_output[0], output, decay);

        return {
            grads[0], // grad_input, 
            grads[1], // grad_decay,
            torch::Tensor(), // Gradients of non-tensor arguments to forward must be `torch::Tensor()`.
            torch::Tensor(), 
            torch::Tensor() 
        };
    }
};

torch::Tensor LIDynamicsFx(
    const torch::Tensor& input,
    const torch::Tensor& decay,
    const torch::Tensor& state,
    float threshold,
    int w_scale
) {
    auto result = LIDynamics::apply(input, decay, state, threshold, w_scale);
    return result[0];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("dynamics", &LIDynamicsFx, "Dynamics of Loihi Leaky Integrator.");
    m.def("fwd", &LIDynamicsFwd, "Fwd dynamics of Loihi Leaky Integrator.");
    m.def("bwd", &LIDynamicsBwd, "Bwd dynamics of Loihi Leaky Integrator.");
}