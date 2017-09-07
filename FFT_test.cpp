//#include "FFT_cooleytukey.cuh"
#include "FFT_stockham.cuh"

int main(int arg, char** argv)
{
	float2 *h_dataI;
	float2 *h_dataO;
	int N1,N2;
	int num;
	int loop = atoi(argv[1]);
	N1 = 512*(int)pow(2.0, (int)(loop/2));
	N2 = 512*pow(2.0, loop-(int)(loop/2));
	//N1 = 4;
	//N2 = 4;
	num = N1*N2;
	h_dataI = new float2[num];
	h_dataO = new float2[num];
	for (int i=0;i<num;i++)
	{
		h_dataI[i].x = i;
		h_dataI[i].y = 0;
	}
	ExeFft(N1, N2, h_dataI, h_dataO, FFT_FORWARD);

	//for (int i=1;i<num;i*=2)
	//{
	//	cout << h_dataO[i].x << " + i*" << h_dataO[i].y << endl;
	//}
	return 0;
}