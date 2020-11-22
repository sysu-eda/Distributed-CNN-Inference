#ifndef _EASY_UNIT_
#define _EASY_UNIT_

#include "opWrapper.h"

namespace opWrapper{
	
class shuffleUnit
{

		
public:
	shuffleUnit(Tensor * input, Tensor * output, int bn, int in_d, int out_d) //const std::string &base_filename, const std::string &conv_number
	{
		std::string dp;
		branch_number = bn;
		if(branch_number == 1)
		{
			//Tensor for intermediate output
			/* 
			Tensor * ts_out_1 = new Tensor();
			Tensor * ts_out_2 = new Tensor();
			Tensor * ts_out_3 = new Tensor();	
			Tensor * ts_out_4 = new Tensor();			
			Tensor * ts_out_5 = new Tensor();
			Tensor * ts_out_6 = new Tensor(); */
			
			
			
			//std::cout<<"Only 1 Branch"<<std::endl;			
			layer_group_before_1 = opWrapper::ConvolutionLayer(input,ts_out_1, 1, 0, 1, 1, in_d, out_d);
			//std::cout<<dp+"groupconv_before_"+conv_number+"_weights_0.npy"<<std::endl;
			
			layer_group_before_1_bn = opWrapper::BNLayer(ts_out_1,ts_out_2, out_d);
			//std::cout<<dp+"groupconv_before_"+conv_number+"_batch_norm"<<std::endl;
			
			layer_dw_1 = opWrapper::DWConvolutionLayer(ts_out_2, ts_out_3, 1, 1,  3, 3, out_d);
			//std::cout<<dp+"depthwise_"+conv_number+"_weights_0.npy"<<std::endl;
			
			layer_dw_1_bn = opWrapper::BNLayer(ts_out_3,ts_out_4, out_d);
			//std::cout<<dp+"depthwise_"+conv_number+"_batch_norm"<<std::endl;
			
			layer_group_after_1 = opWrapper::ConvolutionLayer(ts_out_4, ts_out_5, 1, 0, 1, 1, out_d, out_d);			
			//std::cout<<dp+"groupconv_after_"+conv_number+"_weights_0.npy"<<std::endl;
			
			layer_group_after_1_bn = opWrapper::BNLayer(ts_out_5,ts_out_6, out_d);
			//std::cout<<dp+"groupconv_after_"+conv_number+"_batch_norm"<<std::endl;
			
			
			/* std::cout<<input->info()->tensor_shape()[0]<<std::endl;
			std::cout<<input->info()->tensor_shape()[1]<<std::endl;
			std::cout<<input->info()->tensor_shape()[2]<<std::endl;  
			
			std::cout<<ts_out_6->info()->tensor_shape()[0]<<std::endl;
			std::cout<<ts_out_6->info()->tensor_shape()[1]<<std::endl;
			std::cout<<ts_out_6->info()->tensor_shape()[2]<<std::endl;   */
			
			layer_branchAdd = opWrapper::ElementAddOp(input, ts_out_5, output);
		}
		else
		{
			
			
			//First Branch
			
			//std::cout<<"Two Branch"<<std::endl;
			layer_group_before_1 = opWrapper::ConvolutionLayer(input,ts_out_1, 1, 0, 1, 1, in_d, out_d);
			//std::cout<<dp+"groupconv_before_"+conv_number+"_weights_0.npy"<<std::endl;
			layer_group_before_1_bn = opWrapper::BNLayer(ts_out_1,ts_out_2, out_d);
			//std::cout<<dp+"groupconv_before_"+conv_number+"_batch_norm"<<std::endl;
			layer_dw_1 = opWrapper::DWConvolutionLayer(ts_out_2, ts_out_3, 2, 1,  3, 3, out_d);
			//std::cout<<dp+"depthwise_"+conv_number+"_weights_0.npy"<<std::endl;
			layer_dw_1_bn = opWrapper::BNLayer(ts_out_3,ts_out_4, out_d);
			//std::cout<<dp+"depthwise_"+conv_number+"_batch_norm"<<std::endl;
			layer_group_after_1 = opWrapper::ConvolutionLayer(ts_out_4, ts_out_5, 1, 0, 1, 1 , out_d, out_d);			
			//std::cout<<dp+"groupconv_after_"+conv_number+"_weights_0.npy"<<std::endl;
			layer_group_after_1_bn = opWrapper::BNLayer(ts_out_5,ts_out_6, out_d);
			//std::cout<<dp+"groupconv_after_"+conv_number+"_batch_norm"<<std::endl;
			
			//Second Branch
			
			layer_dw_2 = opWrapper::DWConvolutionLayer(input, ts_out_7, 2, 1, 3, 3, in_d);
			//std::cout<<dp+"depthwise_"+conv_number+"_weights_0.npy"<<std::endl;
			layer_dw_2_bn = opWrapper::BNLayer(ts_out_7,ts_out_8, in_d);
			//std::cout<<dp+"depthwise_"+conv_number+"_batch_norm"<<std::endl;
			layer_group_after_2 = opWrapper::ConvolutionLayer(ts_out_8, ts_out_9, 1, 0, 1, 1, in_d, out_d);	
			//std::cout<<dp+"groupconv_after_"+conv_number+"_weights_0.npy"<<std::endl;
			layer_group_after_2_bn = opWrapper::BNLayer(ts_out_9,ts_out_10, out_d);
			//std::cout<<dp+"groupconv_after_"+conv_number+"_batch_norm"<<std::endl;
			
			layer_branchAdd = opWrapper::ElementAddOp(ts_out_6, ts_out_10, output);	
			
				
		}
	}
	
	~shuffleUnit() = default;
	
	//void configure(Tensor * input, Tensor * output, int branch_number, const std::string &base_filename, const std::string &conv_number);
	
	
	void run()
	{
		if(branch_number == 1)
		{
			layer_group_before_1->run();
			//std::cout<<*reinterpret_cast<float *>(ts_out_1->allocator()->data())<<std::endl;
			layer_group_before_1_bn->run();
			//std::cout<<*reinterpret_cast<float *>(ts_out_2->allocator()->data())<<std::endl;
			layer_dw_1->run();
			//std::cout<<*reinterpret_cast<float *>(ts_out_3->allocator()->data())<<std::endl;
			layer_dw_1_bn->run();
			//std::cout<<*reinterpret_cast<float *>(ts_out_4->allocator()->data())<<std::endl;
			layer_group_after_1->run();
			//std::cout<<*reinterpret_cast<float *>(ts_out_5->allocator()->data())<<std::endl;
			layer_group_after_1_bn->run();
			//std::cout<<*reinterpret_cast<float *>(ts_out_6->allocator()->data())<<std::endl;
			layer_branchAdd->run();
			
			/* std::cout<<*reinterpret_cast<float *>(ts_out_1->allocator()->data())<<std::endl;
			std::cout<<*reinterpret_cast<float *>(ts_out_2->allocator()->data())<<std::endl;
			std::cout<<*reinterpret_cast<float *>(ts_out_3->allocator()->data())<<std::endl;
			std::cout<<*reinterpret_cast<float *>(ts_out_4->allocator()->data())<<std::endl;
			std::cout<<*reinterpret_cast<float *>(ts_out_5->allocator()->data())<<std::endl;
			std::cout<<*reinterpret_cast<float *>(ts_out_6->allocator()->data())<<std::endl; */
		}
		else
		{
			layer_group_before_1->run();
			layer_group_before_1_bn->run();
			layer_dw_1->run();
			layer_dw_1_bn->run();
			layer_group_after_1->run();
			layer_group_after_1_bn->run();
			
			layer_dw_2->run();
			layer_dw_2_bn->run();
			layer_group_after_2->run();		
			layer_group_after_2_bn->run();

			layer_branchAdd->run();			
		}
	}
	
	
private:
int branch_number = 0;


Tensor * ts_out_1 = new Tensor();
Tensor * ts_out_2 = new Tensor();
Tensor * ts_out_3 = new Tensor();
Tensor * ts_out_4 = new Tensor();
Tensor * ts_out_5 = new Tensor();
Tensor * ts_out_6 = new Tensor();
Tensor * ts_out_7 = new Tensor();
Tensor * ts_out_8 = new Tensor();
Tensor * ts_out_9 = new Tensor();
Tensor * ts_out_10 = new Tensor();


//First Branch
NEConvolutionLayer *  layer_group_before_1 = new NEConvolutionLayer();
NEBatchNormalizationLayer * layer_group_before_1_bn = new NEBatchNormalizationLayer();
NEDepthwiseConvolutionLayer * layer_dw_1 = new NEDepthwiseConvolutionLayer();
NEBatchNormalizationLayer * layer_dw_1_bn = new NEBatchNormalizationLayer();
NEConvolutionLayer * layer_group_after_1 = new NEConvolutionLayer();
NEBatchNormalizationLayer * layer_group_after_1_bn = new NEBatchNormalizationLayer();

//Second Branch
NEDepthwiseConvolutionLayer * layer_dw_2 = new NEDepthwiseConvolutionLayer();
NEBatchNormalizationLayer * layer_dw_2_bn = new NEBatchNormalizationLayer();
NEConvolutionLayer * layer_group_after_2 = new NEConvolutionLayer();
NEBatchNormalizationLayer * layer_group_after_2_bn = new NEBatchNormalizationLayer();


//Concat
NEArithmeticAddition * layer_branchAdd = new NEArithmeticAddition();
    
};



}



#endif 