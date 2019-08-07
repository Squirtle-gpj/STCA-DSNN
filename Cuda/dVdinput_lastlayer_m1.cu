#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

__global__ void dVdM_lastlayer_m1( double *dVdinput,const double* h, const int* t_alter,const double* direction, const double *param, const int* int_param)
{

  //determine the id of the thread
	 __shared__ int T_size;
	 __shared__ int nNeurons;
	 __shared__ int maxsteps;
	 __shared__ double threshold;
	 __shared__ double decay1;
	 __shared__ double decay2;

	 T_size =  int_param[0];
	 nNeurons = int_param[1];
	 maxsteps = int_param[2];
	 threshold = param[0];
	 decay1 = param[1];
	 decay2 = param[2];


	int neuron_id = blockIdx.x*blockDim.x +threadIdx.x;
	if(neuron_id > nNeurons - 1 ) {return;}
	int i = t_alter[neuron_id]-1;
	double dM = 1;
	double dS = -1;
	if(direction[neuron_id]<0){
		dM = -1;
		dS = 1;
	}
	// bool ispos = ();
	neuron_id = neuron_id*T_size;


	// double dE = 1;
	int step = maxsteps;
	// double tmp = 0;
	while((i>=0)&&(step>=0)){
		dVdinput[neuron_id+i] = dM + dS;
		i--;
		step--;
		dM = decay1*dM;
		dS = decay2*dS;
	}






}
