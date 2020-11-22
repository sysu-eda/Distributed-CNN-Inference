#include "opWrapper.h"
#include "dataLoader.h"
#include <arm_compute/runtime/Scheduler.h>

#include "unistd.h"
#include "mpi.h"
#include <iostream>
#include <vector>
#include <functional>

using namespace arm_compute;
using namespace utils;
using namespace std;

void partlyGather(void *sendbuf, void *recvbuf, MPI_Comm comm, int rank, int size, int count, MPI_Request array_of_requests[])
{
	MPI_Request request_tmp;
	for(int i = 0; i<size; i++)
	{
		if(i != rank)
		{
			MPI_Irecv((uint8_t*)recvbuf+count*i*4, count, MPI_FLOAT, i, 100, comm, array_of_requests);//+count*i*16
			array_of_requests++;
		}
		
	}
	
	for(int j = 0; j<size; j++)
	{
		if(j != rank)
		{
			MPI_Isend((uint8_t*)sendbuf+count*j*4, count, MPI_FLOAT, j, 100, comm, &request_tmp);//+count*j*16
		}
	}
}

int main (int argc, char **argv)
{
	
	if(argc != 4)
	{
		std::cout<<"Usage: mpiexec -hostfile [hosts] -np [4] -host [raspberrypi0,raspberrypi1] ./main [numberThread(1)] [numberIteration(100) [scale]]"<<std::endl;
		return 0;
	}	
	
	arm_compute::Scheduler::get().set_num_threads(atoi(argv[1]));
	
	int scale = atoi(argv[3]);
	
	int rank, size;
    //char version[MPI_MAX_LIBRARY_VERSION_STRING];
	//Init MPI Env
    MPI_Init (&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	double start , finish;
	
	//Define Output Tensor
	Tensor * conv1_out = new Tensor();
	Tensor * bn1_out = new Tensor();
	Tensor * pool1_out = opWrapper::configure3DTensor(56,56,scale);
	//Layer0
	Tensor * layer0_block0_conv0_out = new Tensor();
	Tensor * layer0_block0_bn0_out = new Tensor();
	Tensor * layer0_block0_conv1_out = new Tensor();
	Tensor * layer0_block0_bn1_out = new Tensor();
	Tensor * layer0_block0_conv2_out = new Tensor();
	Tensor * layer0_block0_bn2_out = new Tensor();
	Tensor * layer0_block0_residual_conv_out = new Tensor();
	Tensor * layer0_block0_residual_bn_out = new Tensor();
	Tensor * layer0_block0_add_out = new Tensor();
	
	Tensor * layer0_block1_conv0_out = new Tensor();
	Tensor * layer0_block1_bn0_out = new Tensor();
	Tensor * layer0_block1_conv1_out = new Tensor();
	Tensor * layer0_block1_bn1_out = new Tensor();
	Tensor * layer0_block1_conv2_out = new Tensor();
	Tensor * layer0_block1_bn2_out = new Tensor();
	Tensor * layer0_block1_add_out = new Tensor();
	
	Tensor * layer0_block2_conv0_out = new Tensor();
	Tensor * layer0_block2_bn0_out = new Tensor();
	Tensor * layer0_block2_conv1_out = new Tensor();
	Tensor * layer0_block2_bn1_out = new Tensor();
	Tensor * layer0_block2_conv2_out = new Tensor();
	Tensor * layer0_block2_bn2_out = new Tensor();
	Tensor * layer0_block2_add_out = new Tensor();
	//Layer1
	Tensor * layer1_block0_conv0_out = new Tensor();
	Tensor * layer1_block0_bn0_out = new Tensor();
	Tensor * layer1_block0_conv1_out = new Tensor();
	Tensor * layer1_block0_bn1_out = new Tensor();
	Tensor * layer1_block0_conv2_out = new Tensor();
	Tensor * layer1_block0_bn2_out = new Tensor();
	Tensor * layer1_block0_residual_conv_out = new Tensor();
	Tensor * layer1_block0_residual_bn_out = new Tensor();
	Tensor * layer1_block0_add_out = new Tensor();
	
	Tensor * layer1_block1_conv0_out = new Tensor();
	Tensor * layer1_block1_bn0_out = new Tensor();
	Tensor * layer1_block1_conv1_out = new Tensor();
	Tensor * layer1_block1_bn1_out = new Tensor();
	Tensor * layer1_block1_conv2_out = new Tensor();
	Tensor * layer1_block1_bn2_out = new Tensor();
	Tensor * layer1_block1_add_out = new Tensor();
	
	Tensor * layer1_block2_conv0_out = new Tensor();
	Tensor * layer1_block2_bn0_out = new Tensor();
	Tensor * layer1_block2_conv1_out = new Tensor();
	Tensor * layer1_block2_bn1_out = new Tensor();
	Tensor * layer1_block2_conv2_out = new Tensor();
	Tensor * layer1_block2_bn2_out = new Tensor();
	Tensor * layer1_block2_add_out = new Tensor();
	
	Tensor * layer1_block3_conv0_out = new Tensor();
	Tensor * layer1_block3_bn0_out = new Tensor();
	Tensor * layer1_block3_conv1_out = new Tensor();
	Tensor * layer1_block3_bn1_out = new Tensor();
	Tensor * layer1_block3_conv2_out = new Tensor();
	Tensor * layer1_block3_bn2_out = new Tensor();
	Tensor * layer1_block3_add_out = new Tensor();
	//Layer2
	Tensor * layer2_block0_conv0_out = new Tensor();
	Tensor * layer2_block0_bn0_out = new Tensor();
	Tensor * layer2_block0_conv1_out = new Tensor();
	Tensor * layer2_block0_bn1_out = new Tensor();
	Tensor * layer2_block0_conv2_out = new Tensor();
	Tensor * layer2_block0_bn2_out = new Tensor();
	Tensor * layer2_block0_residual_conv_out = new Tensor();
	Tensor * layer2_block0_residual_bn_out = new Tensor();
	Tensor * layer2_block0_add_out = new Tensor();
	
	Tensor * layer2_block1_conv0_out = new Tensor();
	Tensor * layer2_block1_bn0_out = new Tensor();
	Tensor * layer2_block1_conv1_out = new Tensor();
	Tensor * layer2_block1_bn1_out = new Tensor();
	Tensor * layer2_block1_conv2_out = new Tensor();
	Tensor * layer2_block1_bn2_out = new Tensor();
	Tensor * layer2_block1_add_out = new Tensor();
	
	Tensor * layer2_block2_conv0_out = new Tensor();
	Tensor * layer2_block2_bn0_out = new Tensor();
	Tensor * layer2_block2_conv1_out = new Tensor();
	Tensor * layer2_block2_bn1_out = new Tensor();
	Tensor * layer2_block2_conv2_out = new Tensor();
	Tensor * layer2_block2_bn2_out = new Tensor();
	Tensor * layer2_block2_add_out = new Tensor();
	
	Tensor * layer2_block3_conv0_out = new Tensor();
	Tensor * layer2_block3_bn0_out = new Tensor();
	Tensor * layer2_block3_conv1_out = new Tensor();
	Tensor * layer2_block3_bn1_out = new Tensor();
	Tensor * layer2_block3_conv2_out = new Tensor();
	Tensor * layer2_block3_bn2_out = new Tensor();
	Tensor * layer2_block3_add_out = new Tensor();
	
	Tensor * layer2_block4_conv0_out = new Tensor();
	Tensor * layer2_block4_bn0_out = new Tensor();
	Tensor * layer2_block4_conv1_out = new Tensor();
	Tensor * layer2_block4_bn1_out = new Tensor();
	Tensor * layer2_block4_conv2_out = new Tensor();
	Tensor * layer2_block4_bn2_out = new Tensor();
	Tensor * layer2_block4_add_out = new Tensor();
	
	Tensor * layer2_block5_conv0_out = new Tensor();
	Tensor * layer2_block5_bn0_out = new Tensor();
	Tensor * layer2_block5_conv1_out = new Tensor();
	Tensor * layer2_block5_bn1_out = new Tensor();
	Tensor * layer2_block5_conv2_out = new Tensor();
	Tensor * layer2_block5_bn2_out = new Tensor();
	Tensor * layer2_block5_add_out = new Tensor();
	//Layer3
	Tensor * layer3_block0_conv0_out = new Tensor();
	Tensor * layer3_block0_bn0_out = new Tensor();
	Tensor * layer3_block0_conv1_out = new Tensor();
	Tensor * layer3_block0_bn1_out = new Tensor();
	Tensor * layer3_block0_conv2_out = new Tensor();
	Tensor * layer3_block0_bn2_out = new Tensor();
	Tensor * layer3_block0_residual_conv_out = new Tensor();
	Tensor * layer3_block0_residual_bn_out = new Tensor();
	Tensor * layer3_block0_add_out = new Tensor();
	
	Tensor * layer3_block1_conv0_out = new Tensor();
	Tensor * layer3_block1_bn0_out = new Tensor();
	Tensor * layer3_block1_conv1_out = new Tensor();
	Tensor * layer3_block1_bn1_out = new Tensor();
	Tensor * layer3_block1_conv2_out = new Tensor();
	Tensor * layer3_block1_bn2_out = new Tensor();
	Tensor * layer3_block1_add_out = new Tensor();
	
	Tensor * layer3_block2_conv0_out = new Tensor();
	Tensor * layer3_block2_bn0_out = new Tensor();
	Tensor * layer3_block2_conv1_out = new Tensor();
	Tensor * layer3_block2_bn1_out = new Tensor();
	Tensor * layer3_block2_conv2_out = new Tensor();
	Tensor * layer3_block2_bn2_out = new Tensor();
	Tensor * layer3_block2_add_out = new Tensor();
	
	Tensor * pool2_out = opWrapper::configure3DTensor(1,1,32*scale);
	
	Tensor * fc1_out = new Tensor();
	//Define Input Tensor
	pmLoader ppm;
	ppm.open("/home/pi/NeurIoT_mpi/go_kart.ppm");	//
	Tensor * input = new Tensor();
	ppm.init_image(*input, Format::F32);
	
	
	
	NEConvolutionLayer * conv1 = opWrapper::ConvolutionLayer(input, conv1_out, 2, 3, 7, 7, 3, 16);
	NEBatchNormalizationLayer * bn1 = opWrapper::BNLayer(conv1_out,bn1_out, scale);
	NEPoolingLayer * pool1 = opWrapper::MaxPoolLayer(bn1_out, pool1_out, 3, 2); //pool1_out
	
	//Layer0
	NEConvolutionLayer * 		layer0_block0_conv0 = opWrapper::ConvolutionLayer(pool1_out, layer0_block0_conv0_out, 1, 0, 1, 1, 16, scale);
	NEBatchNormalizationLayer * layer0_block0_bn0 = opWrapper::BNLayer(layer0_block0_conv0_out,layer0_block0_bn0_out, scale);
	NEConvolutionLayer * 		layer0_block0_conv1 = opWrapper::ConvolutionLayer(layer0_block0_bn0_out, layer0_block0_conv1_out, 1, 1, 3, 3, scale, scale);
	NEBatchNormalizationLayer * layer0_block0_bn1 = opWrapper::BNLayer(layer0_block0_conv1_out, layer0_block0_bn1_out, scale);
	NEConvolutionLayer * 		layer0_block0_conv2 = opWrapper::ConvolutionLayer(layer0_block0_bn1_out, layer0_block0_conv2_out, 1, 0, 1, 1, scale, 4*scale);
	NEBatchNormalizationLayer * layer0_block0_bn2 = opWrapper::BNLayer(layer0_block0_conv2_out, layer0_block0_bn2_out, 4*scale);
	NEConvolutionLayer * 		layer0_block0_residual_conv = opWrapper::ConvolutionLayer(pool1_out, layer0_block0_residual_conv_out, 1, 0, 1, 1, scale, 4*scale);
	NEBatchNormalizationLayer * layer0_block0_residual_bn = opWrapper::BNLayer(layer0_block0_residual_conv_out, layer0_block0_residual_bn_out, 4*scale);		
	NEArithmeticAddition * 		layer0_block0_add = opWrapper::ElementAddOp(layer0_block0_bn2_out, layer0_block0_residual_bn_out, layer0_block0_add_out); //layer0_block0_add_out
	
	NEConvolutionLayer * 		layer0_block1_conv0 = opWrapper::ConvolutionLayer(layer0_block0_add_out, layer0_block1_conv0_out, 1, 0, 1, 1, 4*scale, scale);
	NEBatchNormalizationLayer * layer0_block1_bn0 = opWrapper::BNLayer(layer0_block1_conv0_out,layer0_block1_bn0_out, scale);
	NEConvolutionLayer * 		layer0_block1_conv1 = opWrapper::ConvolutionLayer(layer0_block1_bn0_out, layer0_block1_conv1_out, 1, 1, 3, 3, scale, scale);
	NEBatchNormalizationLayer * layer0_block1_bn1 = opWrapper::BNLayer(layer0_block1_conv1_out, layer0_block1_bn1_out, scale);
	NEConvolutionLayer * 		layer0_block1_conv2 = opWrapper::ConvolutionLayer(layer0_block1_bn1_out, layer0_block1_conv2_out, 1, 0, 1, 1, scale, 4*scale);
	NEBatchNormalizationLayer * layer0_block1_bn2 = opWrapper::BNLayer(layer0_block1_conv2_out, layer0_block1_bn2_out, 4*scale);	
	NEArithmeticAddition * 		layer0_block1_add = opWrapper::ElementAddOp(layer0_block0_add_out, layer0_block1_bn2_out, layer0_block1_add_out); //layer0_block0_add_out
	
	NEConvolutionLayer * 		layer0_block2_conv0 = opWrapper::ConvolutionLayer(layer0_block1_add_out, layer0_block2_conv0_out, 1, 0, 1, 1, 4*scale, scale);
	NEBatchNormalizationLayer * layer0_block2_bn0 = opWrapper::BNLayer(layer0_block2_conv0_out,layer0_block2_bn0_out, scale);
	NEConvolutionLayer * 		layer0_block2_conv1 = opWrapper::ConvolutionLayer(layer0_block2_bn0_out, layer0_block2_conv1_out, 1, 1, 3, 3, scale, scale);
	NEBatchNormalizationLayer * layer0_block2_bn1 = opWrapper::BNLayer(layer0_block2_conv1_out, layer0_block2_bn1_out, scale);
	NEConvolutionLayer * 		layer0_block2_conv2 = opWrapper::ConvolutionLayer(layer0_block2_bn1_out, layer0_block2_conv2_out, 1, 0, 1, 1, scale, 4*scale);
	NEBatchNormalizationLayer * layer0_block2_bn2 = opWrapper::BNLayer(layer0_block2_conv2_out, layer0_block2_bn2_out, 4*scale);	
	NEArithmeticAddition * 		layer0_block2_add = opWrapper::ElementAddOp(layer0_block1_add_out, layer0_block2_bn2_out, layer0_block2_add_out); //layer0_block0_add_out
	
	//Layer1
	NEConvolutionLayer * 		layer1_block0_conv0 = opWrapper::ConvolutionLayer(layer0_block2_add_out, layer1_block0_conv0_out, 1, 0, 1, 1, 4*scale, 2*scale);
	NEBatchNormalizationLayer * layer1_block0_bn0 = opWrapper::BNLayer(layer1_block0_conv0_out,layer1_block0_bn0_out, 2*scale);
	NEConvolutionLayer * 		layer1_block0_conv1 = opWrapper::ConvolutionLayer(layer1_block0_bn0_out, layer1_block0_conv1_out, 2, 1, 3, 3, 2*scale, 2*scale);
	NEBatchNormalizationLayer * layer1_block0_bn1 = opWrapper::BNLayer(layer1_block0_conv1_out, layer1_block0_bn1_out, 2*scale);
	NEConvolutionLayer * 		layer1_block0_conv2 = opWrapper::ConvolutionLayer(layer1_block0_bn1_out, layer1_block0_conv2_out, 1, 0, 1, 1, 2*scale, 8*scale);
	NEBatchNormalizationLayer * layer1_block0_bn2 = opWrapper::BNLayer(layer1_block0_conv2_out, layer1_block0_bn2_out, 8*scale);
	NEConvolutionLayer * 		layer1_block0_residual_conv = opWrapper::ConvolutionLayer(layer0_block2_add_out, layer1_block0_residual_conv_out, 2, 0, 1, 1, 4*scale, 8*scale);
	NEBatchNormalizationLayer * layer1_block0_residual_bn = opWrapper::BNLayer(layer1_block0_residual_conv_out, layer1_block0_residual_bn_out, 8*scale);		
	NEArithmeticAddition * 		layer1_block0_add = opWrapper::ElementAddOp(layer1_block0_bn2_out, layer1_block0_residual_bn_out, layer1_block0_add_out); //layer0_block0_add_out	
	
	NEConvolutionLayer * 		layer1_block1_conv0 = opWrapper::ConvolutionLayer(layer1_block0_add_out, layer1_block1_conv0_out, 1, 0, 1, 1, 8*scale, 2*scale);
	NEBatchNormalizationLayer * layer1_block1_bn0 = opWrapper::BNLayer(layer1_block1_conv0_out,layer1_block1_bn0_out, 2*scale);	
	NEConvolutionLayer * 		layer1_block1_conv1 = opWrapper::ConvolutionLayer(layer1_block1_bn0_out, layer1_block1_conv1_out, 1, 1, 3, 3, 2*scale, 2*scale);
	NEBatchNormalizationLayer * layer1_block1_bn1 = opWrapper::BNLayer(layer1_block1_conv1_out, layer1_block1_bn1_out, 2*scale);
	NEConvolutionLayer * 		layer1_block1_conv2 = opWrapper::ConvolutionLayer(layer1_block1_bn1_out, layer1_block1_conv2_out, 1, 0, 1, 1, 2*scale, 8*scale);
	NEBatchNormalizationLayer * layer1_block1_bn2 = opWrapper::BNLayer(layer1_block1_conv2_out, layer1_block1_bn2_out, 8*scale);	
	NEArithmeticAddition * 		layer1_block1_add = opWrapper::ElementAddOp(layer1_block0_add_out, layer1_block1_bn2_out, layer1_block1_add_out);
	
	NEConvolutionLayer * 		layer1_block2_conv0 = opWrapper::ConvolutionLayer(layer1_block1_add_out, layer1_block2_conv0_out, 1, 0, 1, 1, 8*scale, 2*scale);
	NEBatchNormalizationLayer * layer1_block2_bn0 = opWrapper::BNLayer(layer1_block2_conv0_out,layer1_block2_bn0_out, 2*scale);
	NEConvolutionLayer * 		layer1_block2_conv1 = opWrapper::ConvolutionLayer(layer1_block2_bn0_out, layer1_block2_conv1_out, 1, 1, 3, 3, 2*scale, 2*scale);
	NEBatchNormalizationLayer * layer1_block2_bn1 = opWrapper::BNLayer(layer1_block2_conv1_out, layer1_block2_bn1_out, 2*scale);
	NEConvolutionLayer * 		layer1_block2_conv2 = opWrapper::ConvolutionLayer(layer1_block2_bn1_out, layer1_block2_conv2_out, 1, 0, 1, 1, 2*scale, 8*scale);
	NEBatchNormalizationLayer * layer1_block2_bn2 = opWrapper::BNLayer(layer1_block2_conv2_out, layer1_block2_bn2_out, 8*scale);	
	NEArithmeticAddition * 		layer1_block2_add = opWrapper::ElementAddOp(layer1_block1_add_out, layer1_block2_bn2_out, layer1_block2_add_out); //layer0_block0_add_out
	
	NEConvolutionLayer * 		layer1_block3_conv0 = opWrapper::ConvolutionLayer(layer1_block2_add_out, layer1_block3_conv0_out, 1, 0, 1, 1, 8*scale, 2*scale);
	NEBatchNormalizationLayer * layer1_block3_bn0 = opWrapper::BNLayer(layer1_block3_conv0_out,layer1_block3_bn0_out, 2*scale);
	NEConvolutionLayer * 		layer1_block3_conv1 = opWrapper::ConvolutionLayer(layer1_block3_bn0_out, layer1_block3_conv1_out, 1, 1, 3, 3, 2*scale, 2*scale);
	NEBatchNormalizationLayer * layer1_block3_bn1 = opWrapper::BNLayer(layer1_block3_conv1_out, layer1_block3_bn1_out, 2*scale);
	NEConvolutionLayer * 		layer1_block3_conv2 = opWrapper::ConvolutionLayer(layer1_block3_bn1_out, layer1_block3_conv2_out, 1, 0, 1, 1, 2*scale, 8*scale);
	NEBatchNormalizationLayer * layer1_block3_bn2 = opWrapper::BNLayer(layer1_block3_conv2_out, layer1_block3_bn2_out, 8*scale);	
	NEArithmeticAddition * 		layer1_block3_add = opWrapper::ElementAddOp(layer1_block2_add_out, layer1_block3_bn2_out, layer1_block3_add_out); //layer0_block0_add_out	
	//Layer2
	NEConvolutionLayer * 		layer2_block0_conv0 = opWrapper::ConvolutionLayer(layer1_block3_add_out, layer2_block0_conv0_out, 1, 0, 1, 1, 8*scale, 4*scale);
	NEBatchNormalizationLayer * layer2_block0_bn0 = opWrapper::BNLayer(layer2_block0_conv0_out,layer2_block0_bn0_out, 4*scale);
	NEConvolutionLayer * 		layer2_block0_conv1 = opWrapper::ConvolutionLayer(layer2_block0_bn0_out, layer2_block0_conv1_out, 2, 1, 3, 3, 4*scale, 4*scale);
	NEBatchNormalizationLayer * layer2_block0_bn1 = opWrapper::BNLayer(layer2_block0_conv1_out, layer2_block0_bn1_out, 4*scale);
	NEConvolutionLayer * 		layer2_block0_conv2 = opWrapper::ConvolutionLayer(layer2_block0_bn1_out, layer2_block0_conv2_out, 1, 0, 1, 1, 4*scale, 16*scale);
	NEBatchNormalizationLayer * layer2_block0_bn2 = opWrapper::BNLayer(layer2_block0_conv2_out, layer2_block0_bn2_out, 16*scale);
	NEConvolutionLayer * 		layer2_block0_residual_conv = opWrapper::ConvolutionLayer(layer1_block3_add_out, layer2_block0_residual_conv_out, 2, 0, 1, 1, 8*scale, 16*scale);
	NEBatchNormalizationLayer * layer2_block0_residual_bn = opWrapper::BNLayer(layer2_block0_residual_conv_out, layer2_block0_residual_bn_out, 16*scale);		
	NEArithmeticAddition * 		layer2_block0_add = opWrapper::ElementAddOp(layer2_block0_bn2_out, layer2_block0_residual_bn_out, layer2_block0_add_out); //layer0_block0_add_out
	
	NEConvolutionLayer * 		layer2_block1_conv0 = opWrapper::ConvolutionLayer(layer2_block0_add_out, layer2_block1_conv0_out, 1, 0, 1, 1, 16*scale, 4*scale);
	NEBatchNormalizationLayer * layer2_block1_bn0 = opWrapper::BNLayer(layer2_block1_conv0_out,layer2_block1_bn0_out, 4*scale);
	NEConvolutionLayer * 		layer2_block1_conv1 = opWrapper::ConvolutionLayer(layer2_block1_bn0_out, layer2_block1_conv1_out, 1, 1, 3, 3, 4*scale, 4*scale);
	NEBatchNormalizationLayer * layer2_block1_bn1 = opWrapper::BNLayer(layer2_block1_conv1_out, layer2_block1_bn1_out, 4*scale);
	NEConvolutionLayer * 		layer2_block1_conv2 = opWrapper::ConvolutionLayer(layer2_block1_bn1_out, layer2_block1_conv2_out, 1, 0, 1, 1, 4*scale, 16*scale);
	NEBatchNormalizationLayer * layer2_block1_bn2 = opWrapper::BNLayer(layer2_block1_conv2_out, layer2_block1_bn2_out, 16*scale);	
	NEArithmeticAddition * 		layer2_block1_add = opWrapper::ElementAddOp(layer2_block0_add_out, layer2_block1_bn2_out, layer2_block1_add_out);
	
	NEConvolutionLayer * 		layer2_block2_conv0 = opWrapper::ConvolutionLayer(layer2_block1_add_out, layer2_block2_conv0_out, 1, 0, 1, 1, 16*scale, 4*scale);
	NEBatchNormalizationLayer * layer2_block2_bn0 = opWrapper::BNLayer(layer2_block2_conv0_out,layer2_block2_bn0_out, 4*scale);
	NEConvolutionLayer * 		layer2_block2_conv1 = opWrapper::ConvolutionLayer(layer2_block2_bn0_out, layer2_block2_conv1_out, 1, 1, 3, 3, 4*scale, 4*scale);
	NEBatchNormalizationLayer * layer2_block2_bn1 = opWrapper::BNLayer(layer2_block2_conv1_out, layer2_block2_bn1_out, 4*scale);
	NEConvolutionLayer * 		layer2_block2_conv2 = opWrapper::ConvolutionLayer(layer2_block2_bn1_out, layer2_block2_conv2_out, 1, 0, 1, 1, 4*scale, 16*scale);
	NEBatchNormalizationLayer * layer2_block2_bn2 = opWrapper::BNLayer(layer2_block2_conv2_out, layer2_block2_bn2_out, 16*scale);	
	NEArithmeticAddition * 		layer2_block2_add = opWrapper::ElementAddOp(layer2_block1_add_out, layer2_block2_bn2_out, layer2_block2_add_out);
	
	NEConvolutionLayer * 		layer2_block3_conv0 = opWrapper::ConvolutionLayer(layer2_block2_add_out, layer2_block3_conv0_out, 1, 0, 1, 1, 16*scale, 4*scale);
	NEBatchNormalizationLayer * layer2_block3_bn0 = opWrapper::BNLayer(layer2_block3_conv0_out,layer2_block3_bn0_out, 4*scale);
	NEConvolutionLayer * 		layer2_block3_conv1 = opWrapper::ConvolutionLayer(layer2_block3_bn0_out, layer2_block3_conv1_out, 1, 1, 3, 3, 4*scale, 4*scale);
	NEBatchNormalizationLayer * layer2_block3_bn1 = opWrapper::BNLayer(layer2_block3_conv1_out, layer2_block3_bn1_out, 4*scale);
	NEConvolutionLayer * 		layer2_block3_conv2 = opWrapper::ConvolutionLayer(layer2_block3_bn1_out, layer2_block3_conv2_out, 1, 0, 1, 1, 4*scale, 16*scale);
	NEBatchNormalizationLayer * layer2_block3_bn2 = opWrapper::BNLayer(layer2_block3_conv2_out, layer2_block3_bn2_out, 16*scale);	
	NEArithmeticAddition * 		layer2_block3_add = opWrapper::ElementAddOp(layer2_block2_add_out, layer2_block3_bn2_out, layer2_block3_add_out);
	
	NEConvolutionLayer * 		layer2_block4_conv0 = opWrapper::ConvolutionLayer(layer2_block3_add_out, layer2_block4_conv0_out, 1, 0, 1, 1, 16*scale, 4*scale);
	NEBatchNormalizationLayer * layer2_block4_bn0 = opWrapper::BNLayer(layer2_block4_conv0_out,layer2_block4_bn0_out, 4*scale);
	NEConvolutionLayer * 		layer2_block4_conv1 = opWrapper::ConvolutionLayer(layer2_block4_bn0_out, layer2_block4_conv1_out, 1, 1, 3, 3, 4*scale, 4*scale);
	NEBatchNormalizationLayer * layer2_block4_bn1 = opWrapper::BNLayer(layer2_block4_conv1_out, layer2_block4_bn1_out, 4*scale);
	NEConvolutionLayer * 		layer2_block4_conv2 = opWrapper::ConvolutionLayer(layer2_block4_bn1_out, layer2_block4_conv2_out, 1, 0, 1, 1, 4*scale, 16*scale);
	NEBatchNormalizationLayer * layer2_block4_bn2 = opWrapper::BNLayer(layer2_block4_conv2_out, layer2_block4_bn2_out, 16*scale);	
	NEArithmeticAddition * 		layer2_block4_add = opWrapper::ElementAddOp(layer2_block3_add_out, layer2_block4_bn2_out, layer2_block4_add_out);
	
	NEConvolutionLayer * 		layer2_block5_conv0 = opWrapper::ConvolutionLayer(layer2_block4_add_out, layer2_block5_conv0_out, 1, 0, 1, 1, 16*scale, 4*scale);
	NEBatchNormalizationLayer * layer2_block5_bn0 = opWrapper::BNLayer(layer2_block5_conv0_out,layer2_block5_bn0_out, 4*scale);
	NEConvolutionLayer * 		layer2_block5_conv1 = opWrapper::ConvolutionLayer(layer2_block5_bn0_out, layer2_block5_conv1_out, 1, 1, 3, 3, 4*scale, 4*scale);
	NEBatchNormalizationLayer * layer2_block5_bn1 = opWrapper::BNLayer(layer2_block5_conv1_out, layer2_block5_bn1_out, 4*scale);
	NEConvolutionLayer * 		layer2_block5_conv2 = opWrapper::ConvolutionLayer(layer2_block5_bn1_out, layer2_block5_conv2_out, 1, 0, 1, 1, 4*scale, 16*scale);
	NEBatchNormalizationLayer * layer2_block5_bn2 = opWrapper::BNLayer(layer2_block5_conv2_out, layer2_block5_bn2_out, 16*scale);	
	NEArithmeticAddition * 		layer2_block5_add = opWrapper::ElementAddOp(layer2_block4_add_out, layer2_block5_bn2_out, layer2_block5_add_out);
	//Layer3
	
	NEConvolutionLayer * 		layer3_block0_conv0 = opWrapper::ConvolutionLayer(layer2_block5_add_out, layer3_block0_conv0_out, 1, 0, 1, 1, 16*scale, 8*scale);
	NEBatchNormalizationLayer * layer3_block0_bn0 = opWrapper::BNLayer(layer3_block0_conv0_out,layer3_block0_bn0_out, 8*scale);
	NEConvolutionLayer * 		layer3_block0_conv1 = opWrapper::ConvolutionLayer(layer3_block0_bn0_out, layer3_block0_conv1_out, 2, 1, 3, 3, 8*scale, 8*scale);
	NEBatchNormalizationLayer * layer3_block0_bn1 = opWrapper::BNLayer(layer3_block0_conv1_out, layer3_block0_bn1_out, 8*scale);
	NEConvolutionLayer * 		layer3_block0_conv2 = opWrapper::ConvolutionLayer(layer3_block0_bn1_out, layer3_block0_conv2_out, 1, 0, 1, 1, 8*scale, 32*scale);
	NEBatchNormalizationLayer * layer3_block0_bn2 = opWrapper::BNLayer(layer3_block0_conv2_out, layer3_block0_bn2_out, 32*scale);
	NEConvolutionLayer * 		layer3_block0_residual_conv = opWrapper::ConvolutionLayer(layer2_block5_add_out, layer3_block0_residual_conv_out, 2, 0, 1, 1, 16*scale, 32*scale);
	NEBatchNormalizationLayer * layer3_block0_residual_bn = opWrapper::BNLayer(layer3_block0_residual_conv_out, layer3_block0_residual_bn_out, 32*scale);		
	NEArithmeticAddition * 		layer3_block0_add = opWrapper::ElementAddOp(layer3_block0_bn2_out, layer3_block0_residual_bn_out, layer3_block0_add_out); //layer0_block0_add_out
	
	NEConvolutionLayer * 		layer3_block1_conv0 = opWrapper::ConvolutionLayer(layer3_block0_add_out, layer3_block1_conv0_out, 1, 0, 1, 1, 32*scale, 8*scale);
	NEBatchNormalizationLayer * layer3_block1_bn0 = opWrapper::BNLayer(layer3_block1_conv0_out,layer3_block1_bn0_out, 8*scale);
	NEConvolutionLayer * 		layer3_block1_conv1 = opWrapper::ConvolutionLayer(layer3_block1_bn0_out, layer3_block1_conv1_out, 1, 1, 3, 3, 8*scale, 8*scale);
	NEBatchNormalizationLayer * layer3_block1_bn1 = opWrapper::BNLayer(layer3_block1_conv1_out, layer3_block1_bn1_out, 8*scale);
	NEConvolutionLayer * 		layer3_block1_conv2 = opWrapper::ConvolutionLayer(layer3_block1_bn1_out, layer3_block1_conv2_out, 1, 0, 1, 1, 8*scale, 32*scale);
	NEBatchNormalizationLayer * layer3_block1_bn2 = opWrapper::BNLayer(layer3_block1_conv2_out, layer3_block1_bn2_out, 32*scale);	
	NEArithmeticAddition * 		layer3_block1_add = opWrapper::ElementAddOp(layer3_block0_add_out, layer3_block1_bn2_out, layer3_block1_add_out);
	
	NEConvolutionLayer * 		layer3_block2_conv0 = opWrapper::ConvolutionLayer(layer3_block1_add_out, layer3_block2_conv0_out, 1, 0, 1, 1, 32*scale, 8*scale);
	NEBatchNormalizationLayer * layer3_block2_bn0 = opWrapper::BNLayer(layer3_block2_conv0_out,layer3_block2_bn0_out, 8*scale);
	NEConvolutionLayer * 		layer3_block2_conv1 = opWrapper::ConvolutionLayer(layer3_block2_bn0_out, layer3_block2_conv1_out, 1, 1, 3, 3, 8*scale, 8*scale);
	NEBatchNormalizationLayer * layer3_block2_bn1 = opWrapper::BNLayer(layer3_block2_conv1_out, layer3_block2_bn1_out, 8*scale);
	NEConvolutionLayer * 		layer3_block2_conv2 = opWrapper::ConvolutionLayer(layer3_block2_bn1_out, layer3_block2_conv2_out, 1, 0, 1, 1, 8*scale, 32*scale);
	NEBatchNormalizationLayer * layer3_block2_bn2 = opWrapper::BNLayer(layer3_block2_conv2_out, layer3_block2_bn2_out, 32*scale);	
	NEArithmeticAddition * 		layer3_block2_add = opWrapper::ElementAddOp(layer3_block1_add_out, layer3_block1_bn2_out, layer3_block2_add_out);
	
	NEPoolingLayer * 			pool2 = opWrapper::MaxPoolLayer(layer3_block2_add_out, pool2_out, 7, 1); //pool1_out
	
	NEFullyConnectedLayer *		fcl = opWrapper::FullyConnectedLayer(pool2_out, fc1_out, 32*scale, 1000);
	
	
	//Construct Function Array	
	
	vector<std::function<void()>> m_vecFuc;
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,bn1));
	m_vecFuc.push_back(std::bind(&NEPoolingLayer::run,pool1));
	//Layer0
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer0_block0_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer0_block0_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer0_block0_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer0_block0_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer0_block0_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer0_block0_bn2));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer0_block0_residual_conv));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer0_block0_residual_bn));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer0_block0_add));
	
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer0_block1_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer0_block1_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer0_block1_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer0_block1_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer0_block1_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer0_block1_bn2));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer0_block1_add));
	
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer0_block2_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer0_block2_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer0_block2_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer0_block2_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer0_block2_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer0_block2_bn2));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer0_block2_add));
	
	//Layer1
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer1_block0_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer1_block0_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer1_block0_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer1_block0_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer1_block0_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer1_block0_bn2));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer1_block0_residual_conv));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer1_block0_residual_bn));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer1_block0_add));
	
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer1_block1_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer1_block1_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer1_block1_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer1_block1_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer1_block1_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer1_block1_bn2));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer1_block1_add));
	
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer1_block2_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer1_block2_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer1_block2_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer1_block2_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer1_block2_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer1_block2_bn2));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer1_block2_add));
	
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer1_block3_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer1_block3_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer1_block3_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer1_block3_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer1_block3_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer1_block3_bn2));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer1_block3_add));
	
	//Layer2
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block0_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block0_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block0_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block0_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block0_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block0_bn2));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block0_residual_conv));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block0_residual_bn));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer2_block0_add));
	
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block1_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block1_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block1_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block1_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block1_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block1_bn2));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer2_block1_add));
	
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block2_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block2_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block2_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block2_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block2_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block2_bn2));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer2_block2_add));
	
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block3_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block3_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block3_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block3_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block3_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block3_bn2));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer2_block3_add));
	
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block4_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block4_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block4_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block4_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block4_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block4_bn2));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer2_block4_add));
	
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block5_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block5_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block5_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block5_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer2_block5_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer2_block5_bn2));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer2_block5_add));
	
	//Layer3
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer3_block0_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer3_block0_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer3_block0_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer3_block0_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer3_block0_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer3_block0_bn2));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer3_block0_residual_conv));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer3_block0_residual_bn));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer3_block0_add));
	
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer3_block1_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer3_block1_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer3_block1_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer3_block1_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer3_block1_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer3_block1_bn2));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer3_block1_add));
	
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer3_block2_conv0));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer3_block2_bn0));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer3_block2_conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer3_block2_bn1));
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,layer3_block2_conv2));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,layer3_block2_bn2));
	m_vecFuc.push_back(std::bind(&NEArithmeticAddition::run,layer3_block2_add));
	
	
	m_vecFuc.push_back(std::bind(&NEPoolingLayer::run,pool2));
	m_vecFuc.push_back(std::bind(&NEFullyConnectedLayer::run,fcl));
	//Run
	input->allocator()->allocate();
	if(ppm.is_open())
	{
		ppm.fill_image(*input);
	}	
	ppm.close(); 
	
	void *pTmp0_0 = (void*)(new float[75264]);
	void *pTmp0_1 = (void*)(new float[37632]);
	void *pTmp0_2 = (void*)(new float[18816]);
	void * pArr0[3] = {pTmp0_0, pTmp0_1, pTmp0_2};
	
	void * outArr[3] = {layer0_block2_bn2_out->buffer(), layer1_block3_bn2_out->buffer(), layer2_block5_bn2_out->buffer()};
	
	
	int layer_num = (int)m_vecFuc.size();
	cout<<"The number of layers is: "<< layer_num <<endl;
	
	int commArr[3] = {24,55,99};
	int sizeArr[3] = {18816, 9408, 4704};
	
	MPI_Request arrReq0[size-1];
	//MPI_Request arrReq1[size-1];
	MPI_Status array_of_statuses[size-1];
	
	int flag0 = 0;
	int iters = 0;
	int i_comm = 0;
	
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	while(iters<atoi(argv[2]))
	{
		
		for(int i = 0; i<layer_num; i++){
			m_vecFuc[i]();
			if(i == commArr[i_comm])
			{
				partlyGather(outArr[i_comm], pArr0[i_comm], MPI_COMM_WORLD, rank, size, sizeArr[i_comm], arrReq0);
				while(!flag0)
				{
					usleep(10);
					MPI_Testall(size-1, arrReq0, &flag0, array_of_statuses);
				}
				flag0 = 0;
				i_comm++;
			}
		}		
		iters++;
		i_comm = 0;
	}
	finish = MPI_Wtime();	
	std::cout<<rank<<", Running Finished!"<<std::endl;
	if(rank==0){
		std::cout<<"Elapsed time is "<< finish-start <<" seconds"<<std::endl;
		std::cout<<"时间精度是 "<<MPI_Wtick()<<" 秒钟"<<std::endl;
	}
	
	MPI_Finalize ();
	return 0;
}