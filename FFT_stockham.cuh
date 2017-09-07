#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <iostream>
using namespace std;
#pragma comment(lib, "cutil32D.lib")
#pragma comment(lib, "cudart.lib")

#define M_PI 3.141592657540454
#define M_FACTOR 30
#define THREAD_X 512
#define FFT_FORWARD 1
#define FFT_INVERSE -1

void Transform(float2* d_dataIn, float2* d_dataOut, int N1, int N2);

__global__ void Trans(float2* d_dataIn, float2* d_dataOut, int N1, int N2);

void whirl_factor(float2* dataI, float2 *dataO, int N1, int N2);

void ExeFft(int N, int cN,float2* dataI, float2* dataO, int k);

void DoFft(int N, int cN,float2* dataI, float2* dataO, int k, cudaStream_t cudastream);

__global__ void GPU_FFT_stockham(int N, int R, int Ns,float2* dataI, float2* dataO, int k);

__device__ void FftIteration(int j, int N, int R, int Ns, float2* data0, float2* data1, int k);

__device__ void FFT_2(float2* v);

__device__ int  expand(int idxL, int N1,int N2);

float2 h_multi(float2 a, float2 b);

