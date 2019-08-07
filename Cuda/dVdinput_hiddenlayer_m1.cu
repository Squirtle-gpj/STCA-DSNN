#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

__global__ void dVdM_hiddenlayer( double *dVdinput, const double *dVdO,const double *h, const int* t_alter, const double *param, const int* int_param)
{

	int neuron_id = blockIdx.x*blockDim.x + threadIdx.x;
	if(neuron_id > int_param[1] - 1 ) {return;}
	int last_id = blockIdx.y*blockDim.y + threadIdx.y;
	if (last_id >int_param[2] - 1) {return;}

	// int T_id = blockIdx.x;
	// if(T_id > t_alter[last_id]-1-T_id) {return;}
  //determine the id of the thread
	 __shared__ int T_size;
	 // __shared__ int T_max;//T_max must be higher than 1
	 __shared__ int nNeurons[2];// cur,  last(need to be altered)
	 __shared__ int maxsteps;
	 __shared__ double threshold;
	 __shared__ double decay1;
	 __shared__ double decay2;
	 // __shared__ double V_0;

	 T_size =  int_param[0];
	 // T_max = t_alter[last_id];
	 nNeurons[0] = int_param[1];
	 nNeurons[1] = int_param[2];
	 maxsteps = int_param[3];
	 // nNeurons[2] = int_param[4];
	 threshold = param[0];
	 decay1 = param[1];
	 decay2 = param[2];
	 // V_0 = param[3];

	 int i = t_alter[last_id] - 1;
	 int step = maxsteps;
	//to compute all dV_(neuron_id)^(t_j)/dM(S)_(neuron_id)^(T_id)  T_id < t_j <T_max
	last_id = (last_id*nNeurons[0]+ neuron_id)*T_size;// int dV_startid = (last_id*nNeurons[0]+ neuron_id)*T_size;
	neuron_id = neuron_id*T_size;// int h_startid = neuron_id*T_size;

	double tmp = dVdO[last_id+i]*h[neuron_id+i]*threshold;
	// double tmp2 = dVdO[last_id+i]*tmp;
	double dM = 1*tmp;
	double dS = -1*tmp;
	// double dE = -1*tmp2;
	// double dE = -1;
	while((i>=0)&&(step>=0)){
		dVdinput[last_id + i] = dM+dS;
		i--;
		step--;
		if(i<0){break;}
		tmp = dVdO[last_id+i]*h[neuron_id+i]*threshold;
		// tmp2 = dVdO[last_id+i]*tmp;
		// tmp = dVdO[last_id+i]*h[neuron_id+i]*threshold;
		dM = decay1*dM+tmp;
		dS = decay2*dS - tmp;
		// dE = dE*decay1;
	}

}
