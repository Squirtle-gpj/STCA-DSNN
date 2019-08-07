#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

__global__ void dVdO_hiddenlayer( double *dVdO, const double *dVdinput, const double *weights, const int* t_alter, const double param, const int* int_param)
{
	 int pre_id = blockIdx.x*blockDim.x + threadIdx.x;
	 if(pre_id >  int_param[1] - 1){return;}
	 int last_id = blockIdx.y*blockDim.y + threadIdx.y;
	 if(last_id >  int_param[3] - 1){return;}

	 // int cur_id = blockIdx.z;
	 // if(cur_id > int_param[2]-1){return;}
	 int T_id = blockIdx.z;
	 if(T_id > int_param[4]-1){return;}
	 T_id = t_alter[last_id]-1 - T_id;
	 if(T_id < 0){return;}
  //determine the id of the thread
	 __shared__ int T_size;
	 __shared__ int nNeurons[3];//pre,cur,last
	 __shared__ double V0;
	 // __shared__ double threshold;
	 // __shared__ double decay1;
	 // __shared__ double decay2;

	 T_size =  int_param[0];
	 nNeurons[0] = int_param[1];
	 nNeurons[1] = int_param[2];
	 nNeurons[2] = int_param[3];
	 V0 = param;
	 // double tmp = V0*weights[cur_id*nNeurons[0]+pre_id];
	 // pre_id = nNeurons[0]*last_id+pre_id;
	 // dVdO[pre_id*T_size+ T_id] = V0*weights[pre_id]*dVdinput[T_size*last_id+T_id];
	 // for(int i = 0;i<nNeurons[1];i++){
		//  tmp += dVdinput[(last_id*nNeurons[1]+i)*T_size+T_id]*weights[i*nNeurons[0]+pre_id];
	 // }
	 double tmp = 0;
	 last_id = last_id*nNeurons[1]*T_size + T_id;//dvdinput_id

	 for(int i = 0;i<nNeurons[1];i++){
		 tmp += dVdinput[last_id+i*T_size]*weights[i*nNeurons[0]+pre_id];
	 }
	 tmp = tmp*V0;
	 last_id = blockIdx.y*blockDim.y + threadIdx.y;
	 dVdO[(pre_id+last_id*nNeurons[0])*T_size+T_id] = tmp;
	 // pre_id = (last_id*nNeurons[0]+pre_id)*T_size;
	 // cur_id = (last_id*nNeurons[1]+cur_id)*T_size;
	 // for(int i = 0;i<t_alter[last_id];i++){
		//  atomicAdd(&(dVdO[pre_id+i]), dVdinput[cur_	id+i]*tmp);
	 // }








}
