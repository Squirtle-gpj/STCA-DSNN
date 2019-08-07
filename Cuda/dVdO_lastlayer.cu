#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

__global__ void dVdO_lastlayer( double *dVdO, const double *dVdinput, const double *weights, const int* t_alter, const double param, const int* int_param)
{
	 int  pre_id = blockIdx.x*blockDim.x + threadIdx.x;
	 if(pre_id >  int_param[1] - 1){return;}

	 int last_id = blockIdx.y*blockDim.y + threadIdx.y;
	 if(last_id >  int_param[2] - 1){return;}

	 int T_id = blockIdx.z;
	 if(T_id > int_param[3]-1){return;}
	 T_id = t_alter[last_id]-1 - T_id;
	 if(T_id < 0){return;}
  //determine the id of the thread
	 __shared__ int T_size;
	 // __shared__ int maxsteps;
	 __shared__ int nNeurons[2];//pre,last
	 __shared__ double V0;
	 // __shared__ double threshold;
	 // __shared__ double decay1;
	 // __shared__ double decay2;

	 T_size =  int_param[0];
	 nNeurons[0] = int_param[1];
	 nNeurons[1] = int_param[2];
	 // maxsteps = int_param[3];
	 V0 = param;
	 pre_id = nNeurons[0]*last_id+pre_id;
	 dVdO[pre_id*T_size+ T_id] = V0*weights[pre_id]*dVdinput[T_size*last_id+T_id];
	 // dVdO[pre_id*T_size+ T_id] = T_size*last_id+T_id;







}
