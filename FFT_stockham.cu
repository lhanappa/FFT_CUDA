#include "FFT_stockham.cuh"
#define S_NUM 1

void ExeFft(int N1, int N2,float2* h_dataI, float2* h_dataO, int k)
{

	float2 *d_dataI;
	float2 *d_dataO;
	float2 *round_one;
	float2 *round_two;
	float2 *h_dataO_temp;
	round_one = (float2 *)malloc(N1*N2*sizeof(float2));
	round_two = (float2 *)malloc(N2*N1*sizeof(float2));
	float2 *d_round_one;
	float2 *d_round_two;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_round_one, N1*S_NUM*sizeof(float2)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_dataO, N1*S_NUM*sizeof(float2)));
	unsigned int timer = 0;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
	cudaStream_t* cudastream;
	
	Transform(h_dataI, round_one, N2, N1);

	cudastream = new cudaStream_t[S_NUM];
	for (int i=0;i<N2;i++)
	{
		CUDA_SAFE_CALL(cudaStreamCreate(cudastream+i%S_NUM));
		CUDA_SAFE_CALL(cudaMemcpyAsync(d_round_one + (i%S_NUM)*N1, round_one + i*N1, N1*sizeof(float2), cudaMemcpyHostToDevice, cudastream[i%S_NUM]));
		DoFft(N1, 1, d_round_one + (i%S_NUM)*N1, d_dataO + (i%S_NUM)*N1 , k, cudastream[i%S_NUM]);
		CUDA_SAFE_CALL(cudaMemcpyAsync(round_one + i*N1, d_dataO + (i%S_NUM)*N1, sizeof(float2)*N1, cudaMemcpyDeviceToHost, cudastream[i%S_NUM]));
	}
	cudaThreadSynchronize();

	delete[] cudastream;
	CUDA_SAFE_CALL(cudaFree(d_round_one));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_round_two, N2*S_NUM*sizeof(float2)));
	CUDA_SAFE_CALL(cudaFree(d_dataO));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_dataO, N2*S_NUM*sizeof(float2)));

	clock_t start_t, end_t;
	start_t = clock();
	Transform(round_one, round_two, N1, N2);
	whirl_factor(round_two, round_two, N1, N2);
	end_t = clock();
	cout << "\tsh time:\t" << end_t-start_t << "\t" << N1 << "\t" << N2<< endl; 

	h_dataO_temp = round_one;
	cudastream = new cudaStream_t[S_NUM];
	for (int i=0;i<N1;i++)
	{
		CUDA_SAFE_CALL(cudaStreamCreate(cudastream+i%S_NUM));
		CUDA_SAFE_CALL(cudaMemcpyAsync(d_round_two+ (i%S_NUM)*N2, round_two + i*N2, N2*sizeof(float2), cudaMemcpyHostToDevice, cudastream[i%S_NUM]));
		DoFft(N2, 1, d_round_two + (i%S_NUM)*N2, d_dataO + (i%S_NUM)*N2, k, cudastream[i%S_NUM]);	
		CUDA_SAFE_CALL(cudaMemcpyAsync(h_dataO_temp + i*N2, d_dataO + (i%S_NUM)*N2, N2*sizeof(float2), cudaMemcpyDeviceToHost, cudastream[i%S_NUM]));
	}
	cudaThreadSynchronize();//同步操作，保证操作的一致性
	cutStopTimer(timer);
	float t = cutGetTimerValue(timer);
	cutDeleteTimer(timer);
	cout << "sh time:\t" << t << "\t" << N1 << "\t" << N2<< endl;

	for (int i=0;i<N1;i++)
	{
		for (int j=0;j<N2;j++)
		{
			h_dataO[j*N1+i] = h_dataO_temp[i*N2+j];
		}
	}
	delete[] cudastream;
	delete[] round_one;
	delete[] round_two;
	CUDA_SAFE_CALL(cudaFree(d_round_two));
}

void DoFft(int N, int cN,float2* dataI, float2* dataO, int k, cudaStream_t cudastream)
{
	int R = 2;
	int T = (N/R<THREAD_X)? N/R:THREAD_X;
	int BX = ((N-1)/(R*T) + 1<65536)? ((N-1)/(R*T) + 1):65536;
	int BY =  cN*(-1)*(k - 1)*M_FACTOR/2;//为了小波变换的变通道数 表达式
	if (BY==0)
	{
		BY = cN;
	}
	dim3 dimgrid(BX,BY,1);
	dim3 dimblock(T,1,1);

	float2* temp = dataO;
	float2* dataIn=dataI;

	for (int Ns=1; Ns<N; Ns*=R)
	{
		GPU_FFT_stockham<<<dimgrid, dimblock, 0, cudastream>>>(N, R, Ns, dataIn, temp, k);
		cudaThreadSynchronize();

		float2 *change;
		change = temp;
		temp = dataIn;
		dataIn = change;		
	}

	if (dataIn != dataO)
	{
		cudaMemcpyAsync(dataO,dataIn,sizeof(float2)*N*BY, cudaMemcpyDeviceToDevice, cudastream);
	}


}

void Transform(float2* dataIn, float2* dataOut, int N1, int N2)
	//float2* d_dataIn, float2* d_dataOut, int N1, int N2
{
	//Fermi架构CUDA编程与优化
	//Whitepaper
	//NVIDIA’s Next Generation
	//	CUDATM Compute Architecture:
	//	FermiTM

	//dim3 threads(16,16,1);
	//dim3 blocks(1,1,1);
	//blocks.x = (N1+threads.x-1)/threads.x;
	//blocks.y = (N2+threads.y-1)/threads.y;
	//Trans<<<blocks, threads>>>(d_dataIn, d_dataOut, N1, N2);
	int idin;
	int idout;
	for (int i=0;i<N1;i++)
	{
		for (int j=0;j<N2;j++)
		{
			idin = j*N1+i;
			idout = i*N2+j;
			dataOut[idout].x = dataIn[idin].x;
			dataOut[idout].y = dataIn[idin].y;
		}
	}
	return ;
}

void whirl_factor(float2* dataI, float2 *dataO, int N1, int N2)
{
	//A4(k,j) = cos(2*pi*(k-1)*(j-1)/(n1*n2))-(-1)^0.5*sin(2*pi*(k-1)*(j-1)/(n1*n2));
	//dim3 blocks;
	//dim3 threads;
	int index;
	float id_x;
	float id_y;
	for (int i=0;i<N1;i++)
	{
		for (int j=0;j<N2;j++)
		{
			id_x = i;
			id_y = j;
			index = id_x * N2 + id_y;
			float2 w;
			float an = 2.0*M_PI*(id_x)*(id_y)/(N1*N2);
			w.x = cos(an);
			w.y = -sin(an);
			dataO[index] = h_multi(w, dataI[index]);
		}
	}

}

__global__ void GPU_FFT_stockham(int N, int R, int Ns,float2* dataI, float2* dataO ,int k)
{
	int b, T, t;
	b = blockIdx.x;
	T = blockDim.x;
	t = threadIdx.x;
	int j  = (blockIdx.x)*T + t; 
	if (j< N/R)
	{
		FftIteration(j, N, R, Ns, dataI+blockIdx.y*N, dataO+blockIdx.y*N, k);		
	}

}

__device__ 
void FftIteration(int j, int N, int R, int Ns, float2* data0, float2* data1, int k)
{
	float2 v[2];
	//float2 v[4];
	//float2 v[8];
	//float2 v[16];
	int b = blockIdx.x;
	int t = threadIdx.x;
	int T = blockDim.x;
 
	int idxS = j;
	float angle = (-2) * M_PI * (j%Ns) / (Ns*R);
	for (int r= 0;r<R;r++)
	{
		v[r].x = data0[idxS + r*N/R].x;
		v[r].y = k*data0[idxS + r*N/R].y;

		float2 temp;
		temp = v[r];

		v[r].x = temp.x*__cosf(angle*r) - temp.y*__sinf(angle*r);
		v[r].y = temp.y*__cosf(angle*r) + temp.x*__sinf(angle*r);
	}
	FFT_2(v);
	//FFT_4(v);
	//FFT_8(v);
	//FFT_16(v);
	int idxD;
	idxD = expand(j, Ns, R);
	for (int r=0;r<R;r++)
	{

		data1[idxD+r*Ns].x = v[r].x;
		data1[idxD+r*Ns].y = k*v[r].y;
	}
	/*idxD = (t/Ns)*R + (t%Ns);
	exchange(v, R, 1, idxD, Ns, j, T);
	idxD = b * T * R + t;
	for (int r=0;r<R;r++)
	{
		data1[idxD+r*T] = v[r]; 
	}*/
	return ;
}

__device__ 
	void FFT_2(float2* v)
{
	float2 v0;
	v0.x = v[0].x;
	v0.y = v[0].y;

	v[0].x = v0.x + v[1].x;
	v[0].y = v0.y + v[1].y;

	v[1].x = v0.x - v[1].x;
	v[1].y = v0.y - v[1].y;
}

__device__ 
	int  expand(int idxL, int N1,int N2)
{ 
	return ((idxL/N1)*N1*N2 + (idxL%N1));
}

__device__
	float2 multi(float2 a, float2 b)
{
	float2 c;

	c.x = a.x*b.x - a.y*b.y;
	c.y = a.x*b.y + a.y*b.x;
	return c;
}


//__device__ void FFT(float2* v)
//{
//	float2 v0[16], v1[16];
//	for (int i=0;i<16;i++)
//	{
//		v0[i] = v[i];
//	}
//
//	for (int n=0;n<16;n++)
//	{
//		v[n].x = 0;
//		v[n].y = 0;
//		for (int k=0;k<R;k++)
//		{
//			v1[k].x = __cosf(k*n*M_PI/8);
//			v1[k].y = -1*__sinf(k*n*M_PI/8);
//
//			v[n].x += multi(v0[k], v1[k]).x;
//			v[n].y += multi(v0[k], v1[k]).y;
//
//		}
//	}
//	return ;
//}


__global__ void Trans(float2* d_dataIn, float2* d_dataOut, int N1, int N2)
{
	//int tidx = threadIdx.x;
	//int tidy = threadIdx.y;
	//int bidx = blockIdx.x;
	//int bidy = blockIdx.y;
	//int index = (bidy*gridDim.x + bidx)*(blockDim.x*blockDim.y) + tidy*blockDim.x + tidx;
	//int id_x,id_y;
	//id_x = index%N1;
	//id_y = index/N1;
	//int total = gridDim.x*gridDim.y*(blockDim.x*blockDim.y);

	//while((tidx+(blockDim.x*bidx))<N1&&(tidy+(blockDim.y*bidy))<N2&&(id_x<id_y))
	//{
	//	float2 tempx;
	//	float2 tempy;
	//	tempx.x = d_dataIn[id_x + id_y*N1].x;
	//	tempx.y = d_dataIn[id_x + id_y*N1].y;
	//	tempy.x = d_dataIn[id_y + id_x*N1].x;
	//	tempy.y = d_dataIn[id_y + id_x*N1].y;
	//	d_dataOut[id_x].x = tempy.x;
	//	d_dataOut[id_x].y = tempy.y;
	//	d_dataOut[id_y].x = tempx.x;
	//	d_dataOut[id_y].y = tempx.y;

	//	index += total;
	//	id_x = index%N1;
	//	id_y = index/N1;
	//}
}

__global__ void whirl(float2* d_dataI, float2 *d_dataO, int N1, int N2)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	int index = (bidy*gridDim.x + bidx)*(blockDim.x*blockDim.y) + tidy*blockDim.x + tidx;
	int id_x,id_y;
	id_x = index%N1;
	id_y = index/N1;
	int total = gridDim.x*gridDim.y*(blockDim.x*blockDim.y);
	while((tidx+(blockDim.x*bidx))<N1&&(tidy+(blockDim.y*bidy))<N2&&(id_x<id_y))
	{
		//A4(k,j) = cos(2*pi*(k-1)*(j-1)/(n1*n2))-(-1)^0.5*sin(2*pi*(k-1)*(j-1)/(n1*n2));
		float2 w;
		w.x = __cosf(2*M_PI*(id_x)*(id_y)/(N1*N2));
		w.y = -__sinf(2*M_PI*(id_x)*(id_y)/(N1*N2));
		d_dataO[index].x = multi(w, d_dataI[index]).x;
		d_dataO[index].y = multi(w, d_dataI[index]).y;

		index += total;
		id_x = index%N1;
		id_y = index/N1;
	}
}

float2 h_multi(float2 a, float2 b)
{
	float2 c;

	c.x = a.x*b.x - a.y*b.y;
	c.y = a.x*b.y + a.y*b.x;
	return c;
}

