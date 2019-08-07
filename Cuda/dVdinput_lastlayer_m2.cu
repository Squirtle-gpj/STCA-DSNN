#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#define maxthread 256
__global__ void dVdM_lastlayer_m2( double *dVdinput,const double* h, const int* t_alter,const double* direction, const double *param, const int* int_param)
{

  //determine the id of the thread
	 __shared__ int T_size;
	 __shared__ int nNeurons;
	 __shared__ int maxsteps;
	 __shared__ double threshold;
	 __shared__ double decay1;
	 __shared__ double decay2;
	 // __shared__ bool ispos[maxthread];
	 __shared__ double tmp[maxthread];
	 T_size =  int_param[0];
	 nNeurons = int_param[1];
	 maxsteps = int_param[1];
	 threshold = param[0];
	 decay1 = param[1];
	 decay2 = param[2];


	int neuron_id = blockIdx.x*blockDim.x +threadIdx.x;
	if(neuron_id > nNeurons - 1 ) {return;}
	int i = t_alter[neuron_id]-1;
	// bool ispos = ();
	neuron_id = neuron_id*T_size;

	double dM = 1;
	double dS = -1;
	double dE = -1;
	// ispos[threadIdx.x] = true;

	if(direction[neuron_id]<0){
		dM = -1;
		dS = 1;
		dE = 1;
		// ispos[threadIdx.x] = false;
	}
	// double dE = 1;
	int step = maxsteps;
	tmp[threadIdx.x] = 0;
	while((i>=0)&&(step>=0)){
		dVdinput[neuron_id+i] = dM + dS;
		i--;
		step--;
		if(i<0){break;}
		tmp[threadIdx.x] = dE* h[neuron_id+i]*threshold;
		// if(ispos[threadIdx.x]){
			dM = decay1*dM+tmp[threadIdx.x];
			dS = decay2*dS-tmp[threadIdx.x];
		// }else{
		// 	dM = decay1*dM-tmp[threadIdx.x];
		// 	dS = decay2*dS+tmp[threadIdx.x];
		// }

		dE = decay1*dE;

	}






}
