#include "opWrapper.h"
#include "dataLoader.h"

namespace opWrapper{
	 
	//These are Layer Wrappers
	//They can automatically configure weights and add memory manager
	//However the input and output tensor must be handled outside
	 
	Tensor * configure1DTensor(const int dim0){
		const TensorShape ts_shape(dim0);		
		Tensor * ts = new Tensor();
		ts->allocator()->init(TensorInfo(ts_shape, 1, DataType::F32));
		return ts;
	}	
	Tensor * configure2DTensor(const int dim0, const int dim1){
		const TensorShape ts_shape(dim0, dim1);		
		Tensor * ts = new Tensor();
		ts->allocator()->init(TensorInfo(ts_shape, 1, DataType::F32));
		return ts;
	}	
	Tensor * configure3DTensor(const int dim0, const int dim1, const int dim2){
		const TensorShape ts_shape(dim0, dim1, dim2);		
		Tensor * ts = new Tensor();
		ts->allocator()->init(TensorInfo(ts_shape, 1, DataType::F32));
		return ts;
	}	
	Tensor * configure4DTensor(const int dim0, const int dim1, const int dim2, const int dim3){
		const TensorShape ts_shape(dim0, dim1, dim2);		
		Tensor * ts = new Tensor();
		ts->allocator()->init(TensorInfo(ts_shape, 1, DataType::F32));
		return ts;
	}		
	
	
	

	NEConvolutionLayer * ConvolutionLayer(Tensor * input, Tensor * output, int stride, int padding, const std::string &npy_filename){ 
		
		Tensor * weights = new Tensor();
		NPLoader weightLoader;
		weightLoader.open(npy_filename, DataLayout::NHWC);
		weightLoader.init_tensor(*weights, DataType::F32);
			
		
		NEConvolutionLayer * conv = new NEConvolutionLayer();
		conv->configure(input, weights, nullptr, output, PadStrideInfo(stride, stride, padding, padding), WeightsInfo(),
             Size2D(1U, 1U), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));				
				
		weights->allocator()->allocate();
		output->allocator()->allocate();	
		
		if(weightLoader.is_open())
		{
			weightLoader.fill_tensor(*weights);
			weightLoader.close();
		}	
		
		return conv;
	}
	
	NEPoolingLayer * MaxPoolLayer(Tensor * input, Tensor * output, int poolsize, int stride){		
		NEPoolingLayer * pool = new NEPoolingLayer();
		pool->configure(input, output, PoolingLayerInfo(PoolingType::MAX, poolsize, PadStrideInfo(2, 2, 0, 0)));
		output->allocator()->allocate();
		//std::cout<<output->info()->tensor_shape()[0]<<std::endl;
		return pool; 
	}
	
	NEBatchNormalizationLayer * BNLayer(Tensor * input, Tensor * output, const std::string &base_filename){		
		
		Tensor * mean = new Tensor();
		Tensor * var = new Tensor();
		Tensor * gamma = new Tensor();
		Tensor * beta = new Tensor();
		
		NPLoader meanLoader;
		NPLoader varLoader; 
		NPLoader gammaLoader; 
		NPLoader betaLoader;
		
		meanLoader.open(base_filename+"_moving_mean_0.npy", DataLayout::NHWC);
		varLoader.open(base_filename+"_moving_variance_0.npy", DataLayout::NHWC); 
		gammaLoader.open(base_filename+"_gamma_0.npy", DataLayout::NHWC); 
		betaLoader.open(base_filename+"_beta_0.npy", DataLayout::NHWC);
		
		meanLoader.init_tensor(*mean, DataType::F32);
		varLoader.init_tensor(*var, DataType::F32);
		gammaLoader.init_tensor(*gamma, DataType::F32);
		betaLoader.init_tensor(*beta, DataType::F32);
		
		NEBatchNormalizationLayer * bnl = new NEBatchNormalizationLayer();
		
		bnl->configure(input, output, mean, var, beta, gamma, 0.001f);
		
		mean->allocator()->allocate();
		var->allocator()->allocate();
		gamma->allocator()->allocate();
		beta->allocator()->allocate();		
		output->allocator()->allocate();
		
		if(meanLoader.is_open())
		{
			meanLoader.fill_tensor(*mean);
			meanLoader.close();
		}						
		if(varLoader.is_open())
		{
			varLoader.fill_tensor(*var);
			varLoader.close();
		}	
		if(gammaLoader.is_open())
		{
			gammaLoader.fill_tensor(*gamma);
			gammaLoader.close();
		}	
		if(betaLoader.is_open())
		{
			betaLoader.fill_tensor(*beta);
			betaLoader.close();
		}				
		/* int x = 0;
		for(int j =0; j<24 ; j++){
				std::cout<<*reinterpret_cast<float *>(mean->allocator()->data()+x)<<"  "<<std::endl;
				x=x+4;
		}
		x=0;
		for(int j =0; j<24 ; j++){
				std::cout<<*reinterpret_cast<float *>(var->allocator()->data()+x)<<"  "<<std::endl;
				x=x+4;
		}
		x=0;
		for(int j =0; j<24 ; j++){
				std::cout<<*reinterpret_cast<float *>(beta->allocator()->data()+x)<<"  "<<std::endl;
				x=x+4;
		}
		x=0;
		for(int j =0; j<24 ; j++){
				std::cout<<*reinterpret_cast<float *>(gamma->allocator()->data()+x)<<"  "<<std::endl;
				x=x+4;
		} */
		
		return bnl;
	}
		
	
		
	NEDepthwiseConvolutionLayer * DWConvolutionLayer(Tensor * input, Tensor * output, int stride, int padding, const std::string &base_filename){

		Tensor * weights = new Tensor();
		NPLoader weightLoader;
		weightLoader.open(base_filename, DataLayout::NHWC);
		weightLoader.init_tensor(*weights, DataType::F32);	
		
		NEDepthwiseConvolutionLayer * dwcl = new NEDepthwiseConvolutionLayer();
		dwcl->configure(input, weights, nullptr, output, PadStrideInfo(stride,stride,padding,padding),1);
		
		weights->allocator()->allocate();
		output->allocator()->allocate();		
		
		if(weightLoader.is_open())
		{
			weightLoader.fill_tensor(*weights);
			weightLoader.close();
		}
		
		return dwcl;
		
	}
	
	NEChannelShuffleLayer * CSLayer(Tensor *input, Tensor *output, int num_groups)
	{
		NEChannelShuffleLayer * csl = new NEChannelShuffleLayer();
		csl->configure(input, output, num_groups);
		//output->info()->set_data_layout(DataLayout::NHWC);
		output->allocator()->allocate();
		return csl;
	}
	
	NEArithmeticAddition * ElementAddOp(Tensor * input1, Tensor * input2, Tensor * output)
	{
		NEArithmeticAddition * eal = new NEArithmeticAddition();		
		const TensorShape x(input1->info()->tensor_shape()[0],input1->info()->tensor_shape()[1],input1->info()->tensor_shape()[2]);
		output->allocator()->init(TensorInfo(x,1,DataType::F32));
		eal->configure(input1, input2, output, ConvertPolicy::SATURATE);
		output->allocator()->allocate();		
		return eal;
	}
	
	NEReshapeLayer	* ReshapeOp(Tensor * input, Tensor * output){
		NEReshapeLayer	* rsop = new NEReshapeLayer();
		rsop->configure(input, output);
		output->allocator()->allocate();
		return rsop;
	}
	
	NETranspose * TransposeOp(Tensor * input, Tensor * output){
		NETranspose * trans = new NETranspose();
		trans->configure(input, output);
		output->allocator()->allocate();
		return trans;
	}
	
	NEConcatenateLayer * ConcatLayer(std::vector<ITensor *> inputs_vector, Tensor * output)
	{
		NEConcatenateLayer * cc = new NEConcatenateLayer();
		cc->configure(inputs_vector, output, 2);
		output->allocator()->allocate();
		return cc;
	}
	
	NESplit * SplitLayer(Tensor * input, std::vector<ITensor *> outputs, unsigned int axis)
	{
		NESplit * sl = new NESplit();
		sl->configure (input, outputs, axis);
		//此处复用了内存
		//outputs[0]->allocator()->allocate();
		//outputs[1]->allocator()->allocate();
		//outputs[2]->allocator()->allocate();
		//outputs[3]->allocator()->allocate();		
		return sl;
	}
	
	
	NEReduceMean * ReduceMeanLayer(Tensor * input, Tensor * output, Coordinates reduction_axis)
	{		
		NEReduceMean * rml = new NEReduceMean();
		rml->configure(input, reduction_axis, true, output);
		output->allocator()->allocate();
		return rml;
	}
	
	NEFullyConnectedLayer * FullyConnectedLayer(Tensor * input, Tensor * output, const std::string &base_filename)
	{
		Tensor * weights = new Tensor();
		NPLoader weightLoader;
		weightLoader.open(base_filename+"classifier_weights_0.npy", DataLayout::NHWC);
		weightLoader.init_tensor(*weights, DataType::F32);
		
		Tensor * biases = new Tensor();
		NPLoader biasesLoader;
		biasesLoader.open(base_filename+"classifier_biases_0.npy", DataLayout::NHWC);
		biasesLoader.init_tensor(*biases, DataType::F32);
		
		
		std::cout<<weights->info()->tensor_shape()[0]<<std::endl;
		std::cout<<weights->info()->tensor_shape()[1]<<std::endl;
		std::cout<<weights->info()->tensor_shape()[2]<<std::endl;
		
		std::cout<<biases->info()->tensor_shape()[0]<<std::endl;
		std::cout<<biases->info()->tensor_shape()[1]<<std::endl;
		std::cout<<biases->info()->tensor_shape()[2]<<std::endl;
		
		const TensorShape out_shape(1000);
		output->allocator()-> init(TensorInfo(out_shape, 1, DataType::F32));
		
		NEFullyConnectedLayer * fcl = new NEFullyConnectedLayer();
		fcl->configure(input, weights, biases, output);
		
		std::cout<<output->info()->tensor_shape()[0]<<std::endl;
		std::cout<<output->info()->tensor_shape()[1]<<std::endl;
		std::cout<<output->info()->tensor_shape()[2]<<std::endl;
		
		weights->allocator()->allocate();
		biases->allocator()->allocate();
		output->allocator()->allocate();
		
		if(weightLoader.is_open())
		{
			weightLoader.fill_tensor(*weights);
			weightLoader.close();
		}
		
		if(biasesLoader.is_open())
		{
			biasesLoader.fill_tensor(*biases);
			biasesLoader.close();
		}
		
		return fcl;
	}
	
	
	
	/* NEPermute * PermuteOp(Tensor *input, Tensor *output, PermutationVector &perm)
	{
		NEPermute * pop = new NEPermute();
		pop.configure(input, output, perm);
		output->allocator->allocate();
		return pop;
	} */
	
	//NESoftmaxLayer	*	softmaxLayer(){};
	//NEFullyConnectedLayer * fcLayer(){};
	
 }
 
