//
// Program Entrance of NeurIoT
// Here Define the overall flow
//
#include "opWrapper.h"
#include "dataLoader.h"
#include "shuffleUnit.h"
#include <arm_compute/runtime/Scheduler.h>

#include "unistd.h"

#include "mpi.h"
#include <iostream>
#include <vector>
#include <functional>

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

void pointExchange(void * tmp0, void * tmp1){
	void *tmp2 = tmp0;
	tmp0 = tmp1;
	tmp1 = tmp2;
}

int main(int argc, char **argv)
{
	if(argc != 3)
	{
		std::cout<<"Usage: mpiexec -hostfile [hosts] -np [4] -host [raspberrypi0,raspberrypi1] ./main [numberThread(1)] [numberIteration(100)]"<<std::endl;
		MPI_Finalize ();
		return 0;
	}	
	
	arm_compute::Scheduler::get().set_num_threads(atoi(argv[1]));
	//通过控制
	
	
    int rank, size;
    //char version[MPI_MAX_LIBRARY_VERSION_STRING];
	//Init MPI Env
    MPI_Init (&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	double start , finish;
   
	/******************************
		Init Tensor and Layer
	******************************/
	
	//Output Tensor
	Tensor * conv1_out = new Tensor();
	Tensor * maxpool_out = opWrapper::configure3DTensor(56,56,6);
	Tensor * conv1_bn_out = new Tensor();	
	Tensor * unit1_out = new Tensor();
 	Tensor * unit2_out = new Tensor();
	Tensor * unit3_out = new Tensor();
	Tensor * unit4_out = new Tensor();
	//Tensor * cs4_out_1 = new Tensor();
	Tensor * unit5_out = new Tensor();
	Tensor * unit6_out = new Tensor();
	Tensor * unit7_out = new Tensor();
	Tensor * unit8_out = new Tensor();
	Tensor * unit9_out = new Tensor();
	Tensor * unit10_out = new Tensor();
	Tensor * unit11_out = new Tensor();
	Tensor * unit12_out = new Tensor();
	// Tensor * cs12_out = new Tensor();
	Tensor * unit13_out = new Tensor();
	Tensor * unit14_out = new Tensor();
	Tensor * unit15_out = new Tensor();
	Tensor * unit16_out = new Tensor();
	// Tensor * cs16_out_1 = new Tensor(); 
	Tensor * conv5_out = new Tensor();
	Tensor * conv5_bn_out = new Tensor();
	
	//Input Tensor
	pmLoader ppm;
	ppm.open("/home/pi/NeurIoT_mpi/go_kart.ppm");	
	Tensor * input = new Tensor();
	ppm.init_image(*input, Format::F32);	
	
	//Layer
	
	NEConvolutionLayer * conv1 = opWrapper::ConvolutionLayer(input, conv1_out, 2, 1, 3, 3, 3, 6);
	NEBatchNormalizationLayer * conv1_bn = opWrapper::BNLayer(conv1_out,conv1_bn_out, 6);
	NEPoolingLayer * maxpool = opWrapper::MaxPoolLayer(conv1_bn_out, maxpool_out, 3, 2);
	opWrapper::shuffleUnit * Unit1 = new opWrapper::shuffleUnit(maxpool_out, unit1_out, 2, 6, 68);
	opWrapper::shuffleUnit * Unit2 = new opWrapper::shuffleUnit(unit1_out, unit2_out, 1, 68,68);
	opWrapper::shuffleUnit * Unit3 = new opWrapper::shuffleUnit(unit2_out, unit3_out, 1, 68,68);
	opWrapper::shuffleUnit * Unit4 = new opWrapper::shuffleUnit(unit3_out, unit4_out, 1, 68,68);
	opWrapper::shuffleUnit * Unit5 = new opWrapper::shuffleUnit(unit4_out, unit5_out, 2, 68,136);
	opWrapper::shuffleUnit * Unit6 = new opWrapper::shuffleUnit(unit5_out, unit6_out, 1, 136,136);
	opWrapper::shuffleUnit * Unit7 = new opWrapper::shuffleUnit(unit6_out, unit7_out, 1, 136,136);
	opWrapper::shuffleUnit * Unit8 = new opWrapper::shuffleUnit(unit7_out, unit8_out, 1, 136,136);
	opWrapper::shuffleUnit * Unit9 = new opWrapper::shuffleUnit(unit8_out, unit9_out, 1, 136,136);
	opWrapper::shuffleUnit * Unit10 = new opWrapper::shuffleUnit(unit9_out, unit10_out, 1, 136,136);
	opWrapper::shuffleUnit * Unit11 = new opWrapper::shuffleUnit(unit10_out, unit11_out, 1, 136,136);
	opWrapper::shuffleUnit * Unit12 = new opWrapper::shuffleUnit(unit11_out, unit12_out, 1, 136,136);
	opWrapper::shuffleUnit * Unit13 = new opWrapper::shuffleUnit(unit12_out, unit13_out, 2, 136, 272);
	opWrapper::shuffleUnit * Unit14 = new opWrapper::shuffleUnit(unit13_out, unit14_out, 1, 272, 272);
	opWrapper::shuffleUnit * Unit15 = new opWrapper::shuffleUnit(unit14_out, unit15_out, 1, 272, 272);
	opWrapper::shuffleUnit * Unit16 = new opWrapper::shuffleUnit(unit15_out, unit16_out, 1, 272, 272);
	NEConvolutionLayer * conv5 = opWrapper::ConvolutionLayer(unit16_out, conv5_out,1, 0, 1, 1, 272, 256);
	NEBatchNormalizationLayer * conv5_bn = opWrapper::BNLayer(conv5_out,conv5_bn_out, 256);
	std::cout<<rank<<", Initialization Finished!"<<std::endl;
	/******************************
		Init Run Function
	******************************/
	//Fill Input
	input->allocator()->allocate();
	if(ppm.is_open())
	{
		ppm.fill_image(*input);
	}	
	ppm.close(); 
	
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
		
	//Construct Function Array
	std::vector<std::function<void()>> m_vecFuc;
		
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,conv1));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,conv1_bn));
	m_vecFuc.push_back(std::bind(&NEPoolingLayer::run,maxpool));
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit1));
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit2));
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit3));
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit4));//结束通信 6	
	//m_vecFuc.pushback(partlyGather)	
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit5));
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit6));
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit7));
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit8));
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit9));
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit10));
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit11));
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit12));//结束通信 14	
	//m_vecFuc.pushback(partlyGather)	
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit13));
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit14));
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit15));
	m_vecFuc.push_back(std::bind(&opWrapper::shuffleUnit::run,Unit16));//结束通信 18	
	//_vecFuc.pushback(partlyGather)	
	m_vecFuc.push_back(std::bind(&NEConvolutionLayer::run,conv5));
	m_vecFuc.push_back(std::bind(&NEBatchNormalizationLayer::run,conv5_bn));	
	
	
	int flag0 = 0;
	//int flag1 = 1;
	
	void *pTmp0_0 = (void*)(new float[53312]);
	void *pTmp0_1 = (void*)(new float[26656]);
	void *pTmp0_2 = (void*)(new float[13328]);
	
	//void *pTmp1_0 = (void*)(new float[53312]);
	//void *pTmp1_1 = (void*)(new float[26656]);
	//void *pTmp1_2 = (void*)(new float[13328]);
	
	void * pArr0[3] = {pTmp0_0, pTmp0_1, pTmp0_2};
	//void * pArr1[3] = {pTmp1_0, pTmp1_1, pTmp1_2};	
	void * outArr[3] = {unit4_out->buffer(), unit12_out->buffer(), unit16_out->buffer()};
	int commArr[3] = {6,14,18};
	int sizeArr[3] = {13328, 6664, 3332};
	MPI_Request arrReq0[size-1];
	//MPI_Request arrReq1[size-1];
	MPI_Status array_of_statuses[size-1];
	
	int iters = 0;
	int queue0[21] = {0};
	int queue1[21] = {0};
	
	int j = 0;
	
	int i_comm = 0;
	//int j_comm = 0;
	int onlyOnce = 0;
	
	//for(int i = 0; i<atoi(argv[2]); i++){
	while(iters<atoi(argv[2]))
	{
		if(rank == 0)
		{
			std::cout<<"Iters:"<<iters<<std::endl;
		}
		for(int i = 0; i<21; i++){
			//if(rank == 0){std::cout<<i<<std::endl;}
			if(queue0[i] == 0)
			{
				m_vecFuc[i]();
				if(i == commArr[i_comm])//判断是否需要通信
				{
					//if(rank == 0){std::cout<<"comm"<<std::endl;}
					partlyGather(outArr[i_comm], pArr0[i_comm], MPI_COMM_WORLD, rank, size, sizeArr[i_comm], arrReq0);//buffer, pTmp, 13328, array_of_request
					while(!flag0)
					{
												
						//if((i>j)&&(queue1[j] == 0)&&(flag1==1))
						//{
						//	if(rank == 0){std::cout<<j<<std::endl;}
						if(onlyOnce == 0)
						{
							m_vecFuc[i_comm]();
							queue1[j] = 1;
							j++;	
							onlyOnce = 1;
						}
													
						//}
						//while(!flag0 && !flag1)
						//{
							//MPI_Testall(size-1, arrReq0, &flag0, array_of_statuses);
							//MPI_Testall(size-1, arrReq1, &flag1, array_of_statuses);
						usleep(10);
						//}
						MPI_Testall(size-1, arrReq0, &flag0, array_of_statuses);
						 //Second Judge
					}
					flag0 = 0;
					onlyOnce = 0;
					i_comm++;
				}
				queue0[i] = 1;
			}
		}
		
		j = 0;
		//move the second queue forward
		/* if(rank == 0){
			for(int m = 0; m<21; m++)
			{
				std::cout<<queue0[m];
			}
			std::cout<<std::endl;
			for(int m = 0; m<21; m++)
			{
				std::cout<<queue1[m];
			}
			std::cout<<std::endl;
		} */
		
		//pointExchange(queue0,queue1);
		for(int m = 0; m<21; m++)
		{
			queue0[m] = queue1[m];
			queue1[m] = 0;
		}

		/* if(rank == 0){
			for(int m = 0; m<21; m++)
			{
				std::cout<<queue0[m];
			}
			std::cout<<std::endl;
			for(int m = 0; m<21; m++)
			{
				std::cout<<queue1[m];
			}
			std::cout<<std::endl;
		} */
		
		iters++;
		i_comm = 0;
		//j_comm = 0;
	}
	//}
		
	/* //Run Per Layer
		conv1->run();	
		conv1_bn->run();
		maxpool->run();
		Unit1->run();
		Unit2->run();
		Unit3->run();
		Unit4->run();
		//comm 28*28*272
		//unit4_out
		
		partlyGather(unit4_out->buffer(), pTmp0, MPI_COMM_WORLD, rank, size, 13328, array_of_requests);		
		
		while(!flag)
		{
			usleep(100);
			MPI_Testall(size-1, array_of_requests,&flag, array_of_statuses);			
		}
		flag = 0;
		pointExchange(pTmp0, unit4_out->buffer());
		
		Unit5->run();
		Unit6->run();
		Unit7->run();
		Unit8->run();
		Unit9->run();
		Unit10->run();
		Unit11->run();
		Unit12->run();
		//comm 14*14*544
		//unit12_out
		partlyGather(unit12_out->buffer(), pTmp1, MPI_COMM_WORLD, rank, size, 6664, array_of_requests);			
		while(!flag)
		{
			usleep(100);
			MPI_Testall(size-1, array_of_requests,&flag, array_of_statuses);			
		}
		flag = 0;
		pointExchange(pTmp1, unit12_out->buffer());
		
		Unit13->run();
		Unit14->run();
		Unit15->run();
		Unit16->run();
		//comm 7*7*1088
		//unit16_out
		partlyGather(unit16_out->buffer(), pTmp2, MPI_COMM_WORLD, rank, size, 3332, array_of_requests);
		while(!flag)
		{
			usleep(100);
			MPI_Testall(size-1, array_of_requests,&flag, array_of_statuses);			
		}
		flag = 0;
		pointExchange(pTmp2, unit16_out->buffer());
		
		
		conv5->run();
		conv5_bn->run(); */
	
	
	finish = MPI_Wtime();
	
	std::cout<<rank<<", Running Finished!"<<std::endl;
	
	if(rank==0){
		std::cout<<"Elapsed time is "<< finish-start <<" seconds"<<std::endl;
		std::cout<<"时间精度是 "<<MPI_Wtick()<<" 秒钟"<<std::endl;
	}
	
	/******************************
		run and comm
	******************************/

	
	
	
   
   
   
	//MPI Finalize
	MPI_Finalize ();

    return 0;
}
