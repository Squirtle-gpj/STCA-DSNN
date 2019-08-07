#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

__global__ void dVdM_lastlayer( double *dVdM, double *dVdS, const double *h, const int* t_alter,const double* direction, const double *param, const int* int_param)
{

  //determine the id of the thread
	 __shared__ int T_size;
	 __shared__ int nNeurons;
	 __shared__ double threshold;
	 __shared__ double decay1;
	 __shared__ double decay2;

	 T_size =  int_param[0];
	 nNeurons = int_param[1];
	 threshold = param[0];
	 decay1 = param[1];
	 decay2 = param[2];


	int neuron_id = blockIdx.x*blockDim.x +threadIdx.x;
	if(neuron_id > nNeurons - 1 ) {return;}
	int i = t_alter[neuron_id]-1;
	bool ispos = (direction[neuron_id]>0);
	neuron_id = neuron_id*T_size;

	double dM = 1;
	double dS = 1;
	double dE = 1;
	double tmp = 0;

	if(i>=0){
		if(ispos){
			dVdM[neuron_id+i] = dM;
			dVdS[neuron_id+i] = dS;
		}else{
			dVdM[neuron_id+i] = -1.0*dM;
			dVdS[neuron_id+i] = -1.0*dS;
		}

		i--;
	}
	if(i>=0){

		tmp = threshold*h[neuron_id+i];
		dM = decay1*dM -dE*tmp;
		dS = decay2*dS -dE*tmp;

		if(ispos){
			dVdM[neuron_id+i] = dM;
			dVdS[neuron_id+i] = dS;
		}else{
			dVdM[neuron_id+i] = -1.0*dM;
			dVdS[neuron_id+i] = -1.0*dS;
		}
		i--;
	}

	while(i>=0){

		dE = dE*(decay1-tmp);
		tmp = threshold*h[neuron_id+i];
		dM = decay1*dM -dE*tmp;
		dS = decay2*dS -dE*tmp;

		if(ispos){
			dVdM[neuron_id+i] = dM;
			dVdS[neuron_id+i] = dS;
		}else{
			dVdM[neuron_id+i] = -1.0*dM;
			dVdS[neuron_id+i] = -1.0*dS;
		}

		i--;
	}





}
