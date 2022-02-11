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
__global__ void AdResDynamicsFwdKernel(
    T* __restrict__ real_tensor,
    T* __restrict__ imag_tensor,
    T* __restrict__ threshold,
    T* __restrict__ refractory,
    const T* __restrict__ real_input,
    const T* __restrict__ imag_input,
    const T* __restrict__ sin_decay,
    const T* __restrict__ cos_decay,
    const T* __restrict__ real_state,
    const T* __restrict__ imag_state,
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

    int real = real_state[neuron_id] * w_scale;
    int imag = imag_state[neuron_id] * w_scale;
    int real_new, imag_new;

    int th = th_state[neuron_id] * w_scale;
    int th_step = th_scale;
    int th_decay_int;

    int ref_new = ref_state[neuron_id] * w_scale;
    int ref_decay_int;

    int real_sign;
    int imag_sign;
    double real_decayed;
    double imag_decayed;

    int sin_decay_int, cos_decay_int;
    int linear_id;

    if(num_decays > 1) {
        // individual decays or channelwise decay
        // num_decays * decay_block == num_neurons
        sin_decay_int = sin_decay[(neuron_id) % neurons_per_batch / decay_block];
        cos_decay_int = cos_decay[(neuron_id) % neurons_per_batch / decay_block];
        th_decay_int = (1<<12) - th_decay[(neuron_id) % neurons_per_batch / decay_block];
        ref_decay_int = (1<<12) - ref_decay[(neuron_id) % neurons_per_batch / decay_block];
    } else {
        sin_decay_int = sin_decay[0];
        cos_decay_int = cos_decay[0];
        th_decay_int = (1<<12) - th_decay[0];
        ref_decay_int = (1<<12) - ref_decay[0];
    }

    for(int n=0; n<num_steps; ++n) {
        linear_id = n + neuron_id * num_steps;

        ref_new = (ref_new * ref_decay_int) >> 12;
        th = (((th - th0) * th_decay_int) >> 12) + th0;

        real_sign = (real >= 0) ? 1 : -1;
        imag_sign = (imag >= 0) ? 1 : -1;
        
        // calculate this portion in double precision
        real_decayed = 1.0l * cos_decay_int * real;
        imag_decayed = 1.0l * sin_decay_int * imag;
        real_new = real_sign * int(real_sign * real_decayed / 4096) 
                 - imag_sign * int(imag_sign * imag_decayed / 4096)
                 + int(w_scale * real_input[linear_id]);
        
        // calculate this portion in double precision
        real_decayed = 1.0l * sin_decay_int * real;
        imag_decayed = 1.0l * cos_decay_int * imag;
        imag_new = real_sign * int(real_sign * real_decayed / 4096) 
                 + imag_sign * int(imag_sign * imag_decayed / 4096)
                 + int(w_scale * imag_input[linear_id]);

        real_tensor[linear_id] = 1.0f * real_new / w_scale;
        imag_tensor[linear_id] = 1.0f * imag_new / w_scale;
        threshold[linear_id] = 1.0f * th / w_scale;
        refractory[linear_id] = 1.0f * ref_new / w_scale;
        
        if(imag_new >= (th + ref_new)) {
            // real = 0;
            // imag = (th + ref_new)-1;
            real = real_new;
            imag = imag_new;
            ref_new += 2*th;
            th += th_step;
        } else {
            real = real_new;
            imag = imag_new;
        }
    }
}

template <class T>
__global__ void AdResDynamicsBwdKernel(
    T* __restrict__ grad_real_input_tensor,
    T* __restrict__ grad_imag_input_tensor,
    const T* __restrict__ grad_real,
    const T* __restrict__ grad_imag,
    const T* __restrict__ sin_decay_tensor,
    const T* __restrict__ cos_decay_tensor,
    const int num_neurons,
    const int neurons_per_batch,
    const int num_decays, // this determines individual, channelwise or shared decay
    const int decay_block, // if decay_block==1 then its individual neuron
    const int num_steps
) {
    unsigned neuron_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(neuron_id >= num_neurons)    return;

    float sin_decay, cos_decay;
    float grad_real_input = 0;
    float grad_imag_input = 0;
    float grad_real_input_new;
    float grad_imag_input_new;

    if(num_decays > 1) {  
        // individual decays or channelwise decay
        // num_decays * decay_block == num_neurons
        sin_decay = sin_decay_tensor[(neuron_id) % neurons_per_batch / decay_block] / (1<<12);
        cos_decay = cos_decay_tensor[(neuron_id) % neurons_per_batch / decay_block] / (1<<12);
    } else { // shared decays
        sin_decay = sin_decay_tensor[0] / (1<<12);
        cos_decay = cos_decay_tensor[0] / (1<<12);
    }

    int linear_id;

    for(int n=num_steps-1; n>=0; --n) {
        linear_id = n + neuron_id * num_steps;

        grad_real_input_new =  cos_decay * grad_real_input + sin_decay * grad_imag_input + grad_real[linear_id];
        grad_imag_input_new = -sin_decay * grad_real_input + cos_decay * grad_imag_input + grad_imag[linear_id];

        grad_real_input_tensor[linear_id] = grad_real_input_new;
        grad_imag_input_tensor[linear_id] = grad_imag_input_new;

        grad_real_input = grad_real_input_new;
        grad_imag_input = grad_imag_input_new;
    }
}

variable_list AdResDynamicsFwd(
    const Variable real_input,
    const Variable imag_input,
    const Variable sin_decay,
    const Variable cos_decay,
    const Variable ref_decay,
    const Variable th_decay,
    const Variable real_state,
    const Variable imag_state,
    const Variable ref_state,
    const Variable th_state,
    float th_scale,
    float th0,
    int w_scale
) {
    // make sure all the inputs are contigious and in same device
    CHECK_INPUT(real_input);
    CHECK_INPUT(imag_input);
    CHECK_INPUT(sin_decay);
    CHECK_INPUT(cos_decay);
    CHECK_INPUT(real_state);
    CHECK_INPUT(imag_state);
    CHECK_INPUT(th_state);
    CHECK_INPUT(th_decay);
    CHECK_INPUT(ref_state);
    CHECK_INPUT(ref_decay);
    CHECK_DEVICE(real_input, imag_input);
    CHECK_DEVICE(real_input, sin_decay);
    CHECK_DEVICE(real_input, cos_decay);
    CHECK_DEVICE(real_input, real_state);
    CHECK_DEVICE(real_input, imag_state);
    CHECK_DEVICE(real_input, th_state);
    CHECK_DEVICE(real_input, th_decay);
    CHECK_DEVICE(real_input, ref_state);
    CHECK_DEVICE(real_input, ref_decay);
    cudaSetDevice(real_input.device().index());

    auto device = real_input.device().type();
    auto dtype = real_input.dtype();

    auto real = torch::zeros_like(real_input);
    auto imag = torch::zeros_like(imag_input);
    auto threshold = torch::zeros_like(real_input);
    auto refractory = torch::zeros_like(real_input);

    int num_neurons = real_input.numel() / real_input.size(-1);
    int thread = 256;
    int block = ceil(1.0f * num_neurons / thread);

    AdResDynamicsFwdKernel<float><<< block, thread >>>(
        real.data<float>(), imag.data<float>(),
        threshold.data<float>(), refractory.data<float>(),
        real_input.data<float>(), imag_input.data<float>(),
        sin_decay.data<float>(), cos_decay.data<float>(),
        real_state.data<float>(), imag_state.data<float>(),
        ref_state.data<float>(), ref_decay.data<float>(),
        th_state.data<float>(), th_decay.data<float>(),
        th_scale*w_scale, th0*w_scale, w_scale, num_neurons, num_neurons / real_input.size(0),
        sin_decay.numel(), // num_decays
        num_neurons / sin_decay.numel() / real_input.size(0), // decay_block
        real_input.size(-1) // num_steps
    );
    // cudaDeviceSynchronize();

    return {real, imag, threshold, refractory};
}

variable_list AdResDynamicsBwd(
    const Variable grad_real,
    const Variable grad_imag,
    const Variable real,
    const Variable imag,
    const Variable sin_decay,
    const Variable cos_decay
) {
    // make sure all the inputs are contigious
    CHECK_INPUT(grad_real);
    CHECK_INPUT(grad_imag);
    CHECK_INPUT(real);
    CHECK_INPUT(imag);
    CHECK_INPUT(sin_decay);
    CHECK_INPUT(cos_decay);
    // // Since these variables come from autograd context, they are ensured to be on same device
    // CHECK_DEVICE(grad_real, grad_imag);
    // CHECK_DEVICE(grad_real, real);
    // CHECK_DEVICE(grad_real, imag);
    // CHECK_DEVICE(grad_real, sin_decay);
    // CHECK_DEVICE(grad_real, cos_decay);
    cudaSetDevice(grad_real.device().index());

    auto grad_real_input = torch::zeros_like(grad_real);
    auto grad_imag_input = torch::zeros_like(grad_imag);

    int num_neurons = grad_real.numel() / grad_real.size(-1);
    int thread = 256;
    int block = ceil(1.0f * num_neurons / thread);

    // std::cout << grad_imag.index({0, 0, Slice(-50, None)}) << std::endl;

    AdResDynamicsBwdKernel<float><<< block, thread >>>(
        grad_real_input.data<float>(), grad_imag_input.data<float>(),
        grad_real.data<float>(), grad_imag.data<float>(),
        sin_decay.data<float>(), cos_decay.data<float>(),
        num_neurons, num_neurons / grad_real.size(0),
        sin_decay.numel(), // num_decays
        num_neurons / sin_decay.numel() / grad_real.size(0), // decay_block
        grad_real.size(-1) // num_steps
    );
    // cudaDeviceSynchronize();

    // It makes sense to use torch primitives for the following calculations
    // Custom cuda implementation would need reimplementing tensor primitives
    auto grad_sin_decay = - grad_real_input.index({"...", Slice(1, None)}) * imag.index({"...", Slice(None, -1)})
                          + grad_imag_input.index({"...", Slice(1, None)}) * real.index({"...", Slice(None, -1)});
    auto grad_cos_decay =   grad_real_input.index({"...", Slice(1, None)}) * real.index({"...", Slice(None, -1)})
                          + grad_imag_input.index({"...", Slice(1, None)}) * imag.index({"...", Slice(None, -1)});

    if(torch::numel(cos_decay) == 1) {
        grad_sin_decay = torch::sum(grad_sin_decay.flatten(), 0, true);
        grad_cos_decay = torch::sum(grad_cos_decay.flatten(), 0, true);
    } else {
        grad_sin_decay = torch::sum(grad_sin_decay, c10::IntArrayRef({0, -1}));
        grad_cos_decay = torch::sum(grad_cos_decay, c10::IntArrayRef({0, -1}));
        if(grad_sin_decay.ndimension()!=1) {
            grad_sin_decay = torch::sum(grad_sin_decay.reshape({grad_sin_decay.size(0), -1}), 1);
            grad_cos_decay = torch::sum(grad_cos_decay.reshape({grad_cos_decay.size(0), -1}), 1);
        }
    }

    return {grad_real_input, grad_imag_input, grad_sin_decay, grad_cos_decay};
}


class AdResDynamics : public torch::autograd::Function<AdResDynamics> {
public:
    static variable_list forward(
        AutogradContext* ctx, 
        const Variable real_input,
        const Variable imag_input,
        const Variable sin_decay,
        const Variable cos_decay,
        const Variable ref_decay,
        const Variable th_decay,
        const Variable real_state,
        const Variable imag_state,
        const Variable ref_state,
        const Variable th_state,
        float th_scale,
        float th0,
        int w_scale
    ) {
        auto result = AdResDynamicsFwd(
                real_input, imag_input, 
                sin_decay, cos_decay, 
                ref_decay, th_decay,
                real_state, imag_state, 
                ref_state, th_state,
                th_scale, th0,
                w_scale
            );

        ctx->save_for_backward({result[0], result[1], sin_decay, cos_decay});

        return result;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_output) {
        // expand gradient outputs
        auto grad_real = grad_output[0];
        auto grad_imag = grad_output[1];
        // expand saved variables and parameters
        auto saved = ctx->get_saved_variables();
        auto real = saved[0];
        auto imag = saved[1];
        auto sin_decay = saved[2];
        auto cos_decay = saved[3];
        
        auto grads = AdResDynamicsBwd(
                grad_real, grad_imag, 
                real, imag, 
                sin_decay, cos_decay
            );

        return {
            grads[0], // grad_real_input, 
            grads[1], // grad_imag_input, 
            grads[2], // grad_sin_decay,
            grads[3], // grad_cos_decay,
            torch::Tensor(), // Gradients of non-tensor arguments to forward must be `torch::Tensor()`.
            torch::Tensor(), 
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

std::vector<torch::Tensor> AdResDynamicsFx(
    const torch::Tensor& real_input,
    const torch::Tensor& imag_input,
    const torch::Tensor& sin_decay,
    const torch::Tensor& cos_decay,
    const torch::Tensor& ref_decay,
    const torch::Tensor& th_decay,
    const torch::Tensor& real_state,
    const torch::Tensor& imag_state,
    const torch::Tensor& ref_state,
    const torch::Tensor& th_state,
    float th_scale,
    float th0,
    int w_scale
) {
    auto result = AdResDynamics::apply(
            real_input, imag_input, 
            sin_decay, cos_decay, ref_decay, th_decay,
            real_state, imag_state, ref_state, th_state,
            th_scale, th0, w_scale
        );

    return {result[0], result[1], result[2], result[3]};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("dynamics", &AdResDynamicsFx, "Dynamics of Loihi Adaptive Resonator.");
    m.def("fwd", &AdResDynamicsFwd, "Fwd dynamics of Loihi Adaptive Resonator.");
    m.def("bwd", &AdResDynamicsBwd, "Bwd dynamics of Loihi Adaptive Resonator.");
}