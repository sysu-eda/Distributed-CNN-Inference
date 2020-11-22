#ifndef OPWRAPPER
#define OPWRAPPER
 
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"
#include "utils/Utils.h"
#include <string>

using namespace arm_compute;
using namespace utils;

namespace opWrapper{
	 
	//These are Layer Wrappers
	//They can automatically configure weights and add memory manager
	//However the input and output tensor must be handled outside
	Tensor * configure1DTensor(int dim0);
	Tensor * configure2DTensor(int dim0, int dim1);
	Tensor * configure3DTensor(int dim0, int dim1, int dim2);		
	Tensor * configure4DTensor(int dim0, int dim1, int dim2, int dim3);	
	NEConvolutionLayer * ConvolutionLayer(Tensor * input, Tensor * output,  
											int stride, int padding, 
											int w_h, int w_w, int w_d, int w_c);//, const std::string &name
	NEPoolingLayer * MaxPoolLayer(Tensor * input, Tensor * output, int poolsize, int stride);
	
	NEBatchNormalizationLayer * BNLayer(Tensor * input, Tensor * output, int v);
	
	NEDepthwiseConvolutionLayer * DWConvolutionLayer(Tensor * input, Tensor * output, int stride, int padding,  int w_h, int w_w, int w_c);
	
	NEChannelShuffleLayer * CSLayer(Tensor *input, Tensor *output, int num_groups);
	
	NEArithmeticAddition * ElementAddOp(Tensor * input1, Tensor * input2, Tensor * output);
	
	NEReshapeLayer	* ReshapeOp(Tensor * input, Tensor * output);
	
	NETranspose * TransposeOp(Tensor * input, Tensor * output);
	
	NEConcatenateLayer * ConcatLayer(std::vector<ITensor *> inputs_vector,Tensor *output);
	
	NESplit * SplitLayer(Tensor * input, std::vector<ITensor *> outputs, unsigned int axis);
	
	NEReduceMean * ReduceMeanLayer(Tensor * input, Tensor * output, Coordinates reduction_axis);
	
	NEFullyConnectedLayer * FullyConnectedLayer(Tensor * input, Tensor * output, int in, int out);
	
	
	
	//NEPermute * PermuteOp(Tensor *input, Tensor *output, PermutationVector &perm);
	
	//NEChannelExtract * ChannelExtractOp();
	
	
	
	//NETranspose * TransposeOp(Tensor * input, Tensor * output);
	
	//NESoftmaxLayer	*	softmaxLayer(){};
	//NEFullyConnectedLayer * fcLayer(){};
	
	
 }
 
 
#endif