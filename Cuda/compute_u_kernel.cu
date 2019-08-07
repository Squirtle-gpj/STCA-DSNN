#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define maxthread 256
__global__ void compute_u_kernel( double *output, double *u, const double * input, const int* int_param,const double *param)
{

	if(threadIdx.x > maxthread - 1){ return;}
	 int cur_id = blockIdx.x*blockDim.x +threadIdx.x;
	if(cur_id > int_param[1] - 1 ) {return;}

	 __shared__ int T_size;
	 // __shared__ int Numcurlayer;
	 __shared__ double threshold;
	 // __shared__ double subthreshold;
	 __shared__ double decay1;
	 __shared__ double decay2;
	 __shared__ double V_cur[maxthread];
	 T_size =  int_param[0];
	 // Numcurlayer = int_param[1];
	 threshold = param[0];
	 // subthreshold = param[1];
	 decay1 = param[1];
	 decay2 = param[2];

	// int afferent_id = blockIdx.x*blockDim.x +threadIdx.x;
	// if(afferent_id > Numcurlayer - 1 ) {return;}
	int curid = cur_id*T_size;
    int endid = curid + T_size;
    double m = 0;
    double s = 0;
    double e = 0;
    // V_cur[threadIdx.x] = 0;
    bool fired_pre = false;


    while(curid < endid)
    {
			m = m*decay1;
			s = s*decay2;
			//now, V is a tmp
			V_cur[threadIdx.x] = input[curid];
			if(V_cur[threadIdx.x] != 0)
			{
				m = m + V_cur[threadIdx.x];
				s = s + V_cur[threadIdx.x];
			}
			e = e*decay1;
			if (fired_pre)
			{
				e = e + threshold;
			}
			// now, fired_pre is fired_cur.

			V_cur[threadIdx.x] = m -s -e;
			u[curid] = V_cur[threadIdx.x];
			fired_pre = (V_cur[threadIdx.x] > threshold);
			if(fired_pre)
			{
				output[curid] = 1.0;
			}
			curid++;

    }



}
