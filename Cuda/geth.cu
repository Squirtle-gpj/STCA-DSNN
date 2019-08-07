#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

__global__ void geth( double *h,const double *u, const int* int_param,const double *param	)
{

  //determine the id of the thread
	 __shared__ int T_size;
	 __shared__ int nNeurons;
	 __shared__ int c;//case: 1 rectangular 2 probability
	 __shared__ double a;
	 __shared__ double threshold;

	 T_size =  int_param[0];
	 nNeurons = int_param[1];
	 c = int_param[2];
	 a = param[0];
	 threshold = param[1];

	int T_id = blockIdx.x*blockDim.x + threadIdx.x;//each thread computes h among 32 time points
	int neuron_id = blockIdx.y*blockDim.y + threadIdx.y;
	if(T_id > T_size - 1 ) {return;}
	if(neuron_id > nNeurons - 1 ) {return;}
	neuron_id = neuron_id*T_size + T_id;


	switch(c){
		case 1:
			// for(int i=0;i<32;i++){
						// if(T_id >(T_size-1)){
						// 	break;
						// }
						if(fabs(u[neuron_id]-threshold)<=(0.5*a)){
							h[neuron_id] = 1/a;
						}
			// }
			break;

		case 2:
			// for(int i=0;i<32;i++){
						// if(T_id >(T_size-1)){
						// 	break;
						// }
						h[neuron_id] = a*exp(-2*a*fabs(u[neuron_id]-threshold));
			// }
			break;

	}

}
