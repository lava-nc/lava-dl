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
__global__ void convKernel(
    T* output, 
    const T* input, const T* filter, 
    unsigned input_size, unsigned filter_size, unsigned num_neurons,
    float Ts
) {
    // calcualte the threadID
    // this is the index of the signal along time axis
    unsigned tID = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned nID = blockIdx.y * blockDim.y + threadIdx.y;

    if(tID >= input_size)	return;
    if(nID >= num_neurons)  return;

    // declare local variables
    float result = 0.0f;

    // calculate convolution sum
    for(unsigned i=0; i<filter_size; ++i)
    {
        int id = tID - i;
        if(id >= 0)	    result += input[id + nID * input_size] * filter[i];
    }
    output[tID + nID * input_size] = result * Ts;	
    return;
}

template <class T>
__global__ void corrKernel(
    T* output, 
    const T* input, const T* filter, 
    unsigned input_size, unsigned filter_size, unsigned num_neurons,
    float Ts
) {
	// calcualte the threadID
	// this is the index of the signal along time axis
    unsigned tID = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned nID = blockIdx.y * blockDim.y + threadIdx.y;

    if(tID >= input_size)	return;
    if(nID >= num_neurons)  return;

    // declare local variables
    float result = 0.0f;

    // calculate convolution sum
    for(unsigned i=0; i<filter_size; ++i)
    {
        int id = tID + i;
        if(id < input_size) result += input[id + nID * input_size] * filter[i];
    }
    output[tID + nID * input_size] = result * Ts;	
    return;
}

Variable conv(
    const Variable input,
    const Variable filter,
    float Ts=1
) {
    CHECK_INPUT(input);
    CHECK_INPUT(filter);
    CHECK_DEVICE(input, filter);
    cudaSetDevice(input.device().index());

    auto device = input.device().type();
    auto dtype = input.dtype();
    auto output = torch::zeros_like(input);

    int num_neurons = input.numel() / input.size(-1);
    int input_size = input.size(-1);
    int filter_size = filter.numel();
    
    dim3 thread(128, 8, 1);
    int num_grid = ceil( 1.0f * num_neurons / thread.y / 65535);
	int neurons_per_grid = ceil(1.0f * num_neurons / num_grid);

    for(auto i=0; i<num_grid; ++i) {
        int start = i * neurons_per_grid;
        int neurons_in_grid = (start + neurons_per_grid <= num_neurons) ? neurons_per_grid : num_neurons - start;
        if(neurons_in_grid < 0) break;
        dim3 block(
            ceil(1.0f * input_size / thread.x),
            ceil(1.0f * neurons_in_grid / thread.y),
            1
        );
        // these should never be trigerred
		if(block.y >= 65535)	AT_ERROR("maximum blockDim.y exceeded.");
		if(block.z >= 65535)	AT_ERROR("maximum blockDim.z exceeded.");

        convKernel<float><<< block, thread >>>(
            output.data<float>() + start * input_size,
            input.data<float>() + start * input_size,
            filter.data<float>(), 
            input_size, filter_size,
            neurons_in_grid, Ts
        );
    }

    return output;
}

Variable corr(
    const Variable input,
    const Variable filter,
    float Ts=1
) {
    CHECK_INPUT(input);
    CHECK_INPUT(filter);
    CHECK_DEVICE(input, filter);
    cudaSetDevice(input.device().index());

    auto device = input.device().type();
    auto dtype = input.dtype();
    auto output = torch::zeros_like(input);

    int num_neurons = input.numel() / input.size(-1);
    int input_size = input.size(-1);
    int filter_size = filter.numel();
    
    dim3 thread(128, 8, 1);
    int num_grid = ceil( 1.0f * num_neurons / thread.y / 65535);
	int neurons_per_grid = ceil(1.0f * num_neurons / num_grid);

    for(auto i=0; i<num_grid; ++i) {
        int start = i * neurons_per_grid;
        int neurons_in_grid = (start + neurons_per_grid <= num_neurons) ? neurons_per_grid : num_neurons - start;
        if(neurons_in_grid < 0) break;
        dim3 block(
            ceil(1.0f * input_size / thread.x),
            ceil(1.0f * neurons_in_grid / thread.y),
            1
        );
        // these should never be trigerred
		if(block.y >= 65535)	AT_ERROR("maximum blockDim.y exceeded.");
		if(block.z >= 65535)	AT_ERROR("maximum blockDim.z exceeded.");

        corrKernel<float><<< block, thread >>>(
            output.data<float>() + start * input_size,
            input.data<float>() + start * input_size,
            filter.data<float>(), 
            input_size, filter_size,
            neurons_in_grid, Ts
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &conv, "Convolution filtering in time.");
    m.def("bwd", &corr, "Correlation filtering in time.");
}
