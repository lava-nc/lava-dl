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
__global__ void shiftKernel( // shift each neuron by a constant value
    T* output, 
	const T* input, const T shift_value, 
	unsigned input_size, unsigned num_neurons,
	float Ts
) {
    // calcualte the threadID
    // this is the index of the signal along time axis
    unsigned tID = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned nID = blockIdx.y * blockDim.y + threadIdx.y;

    if(tID >= input_size)	return;
    if(nID >= num_neurons)  return;

    // floor the shift to integer value
	int shift_blocks = static_cast<int>(shift_value/Ts);

    float temp = 0;
	auto neuron_offset = input_size * nID;
	// shift the elements
	int id = tID - shift_blocks;
	if(id >= 0 && id <input_size)	temp = input[id + neuron_offset];

	output[tID + neuron_offset] = temp;
	return;
}

template <class T>
__global__ void shiftKernel( // shift each neuron individually
    T* output,
    const T* input, const T* shift_lut,
    unsigned input_size, unsigned num_neurons, 
    float Ts
) {
    // calcualte the threadID
    // this is the index of the signal along time axis
    unsigned tID = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned nID = blockIdx.y * blockDim.y + threadIdx.y;

    if(tID >= input_size)	return;
    if(nID >= num_neurons)  return;

    // floor the shift to integer value
    int shift_blocks = static_cast<int>(shift_lut[nID]/Ts);

    float temp = 0;
    auto neuronOffset = input_size * nID;
    // shift the elements
    int id = tID - shift_blocks;
    if(id >= 0 && id <input_size)	temp = input[id + neuronOffset];

    output[tID + neuronOffset] = temp;
    return;
}

template <class T>
void shiftFx(
    T* output,
    const T* input,
    float shift_value,
    unsigned input_size,
    unsigned num_neurons,
    float sampling_time
) {
    dim3 thread(128, 8, 1);
    int num_grid = ceil( 1.0f * num_neurons / thread.y / 65535 );
    int neurons_per_grid = ceil(1.0f * num_neurons / num_grid);

    for(auto i=0; i<num_grid; ++i)
    {
        int start_offset = i * neurons_per_grid;
        int neurons_in_grid = (start_offset + neurons_per_grid <= num_neurons) ? neurons_per_grid : num_neurons - start_offset;

        if(neurons_in_grid < 0)	break;

        dim3 block(	ceil( 1.0f * input_size      /thread.x ), 
                    ceil( 1.0f * neurons_in_grid /thread.y ), 
                    1 );

        // these should never be trigerred
        if(block.y >= 65535)	AT_ERROR("maximum blockDim.y exceeded.");
        if(block.z >= 65535)	AT_ERROR("maximum blockDim.z exceeded.");

        shiftKernel<T><<< block, thread >>>(
            output + start_offset * input_size, 
            input  + start_offset * input_size, 
            shift_value, input_size, neurons_in_grid, sampling_time
        );
    }
}

template <class T>
void shiftFx(
    T* output,
    const T* input,
    const T* shift_lut,
    unsigned input_size,
    unsigned num_neurons,
    unsigned num_batch,
    float sampling_time
) {
    dim3 thread(128, 8, 1);
    int num_grid = ceil( 1.0f * num_neurons / thread.y / 65535);
    int neurons_per_grid = ceil(1.0f * num_neurons / num_grid);

    for(unsigned i=0; i<num_batch; ++i)
    {
        for(auto j=0; j<num_grid; ++j)
        {	
            int start_offset = j * neurons_per_grid;
            int neurons_in_grid = (start_offset + neurons_per_grid <= num_neurons) ? neurons_per_grid : num_neurons - start_offset;

            if(neurons_in_grid < 0)	break;

            dim3 block(	ceil( 1.0f * input_size    /thread.x ), 
                        ceil( 1.0f * neurons_in_grid /thread.y ), 
                        1 );

            // these should never be trigerred
            if(block.y >= 65535)	AT_ERROR("maximum blockDim.y exceeded.");
            if(block.z >= 65535)	AT_ERROR("maximum blockDim.z exceeded.");

            shiftKernel<T><<< block, thread >>>(
                output + (i * num_neurons + start_offset) * input_size, 
                input  + (i * num_neurons + start_offset) * input_size, 
                shift_lut + start_offset, 
                input_size, neurons_in_grid, sampling_time
            );
        }
    }
}

Variable shiftGlobal(
    const Variable input,
    float shift_step,
    float sampling_time=1
) {
    CHECK_INPUT(input);
    cudaSetDevice(input.device().index());

    auto output = torch::zeros_like(input);

    auto input_size = input.size(-1);
    auto num_neurons = input.numel() / input.size(-1);
    shiftFx<float>(
        output.data<float>(), input.data<float>(), shift_step,
        input_size, num_neurons, sampling_time
    );     

    return output;
}

Variable shift(
    const Variable input,
    const Variable shift_lut,
    float sampling_time=1
) {
    CHECK_INPUT(input);
    CHECK_INPUT(shift_lut);
    CHECK_DEVICE(input, shift_lut);
    cudaSetDevice(input.device().index());

    auto device = input.device().type();
    auto dtype = input.dtype();
    auto output = torch::zeros_like(input);

    auto input_size = input.size(-1);
    if(shift_lut.numel() == 1) {// global shift
        auto num_neurons = input.numel() / input.size(-1);
        shiftFx<float>(
            output.data<float>(), input.data<float>(), shift_lut.item<float>(),
            input_size, num_neurons, sampling_time
        );     
    } else { // individual shift
        auto num_batch = input.size(0);
        auto num_neurons = input.numel() / input.size(-1) / num_batch;
        AT_ASSERTM(shift_lut.numel() == num_neurons, "shift and number of neurons must be same");
        shiftFx<float>(
            output.data<float>(), input.data<float>(), shift_lut.data<float>(),
            input_size, num_neurons, num_batch, sampling_time
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shift", &shift,       "Element shift in time.");
    m.def("shift", &shiftGlobal, "Element shift in time.");
}
