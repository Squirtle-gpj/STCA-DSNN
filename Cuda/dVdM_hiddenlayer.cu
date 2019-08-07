#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

__global__ void dVdM_hiddenlayer( double *dVdM, double *dVdS, const double *dVdO,const double *h, const int* t_alter, const double *param, const int* int_param)
{
	int last_id = blockIdx.z;
	if (last_id >int_param[2] - 1) {return;}
	int neuron_id = blockIdx.y*blockDim.x + threadIdx.x;
	if(neuron_id > int_param[1] - 1 ) {return;}
	int T_id = blockIdx.x;
	if(T_id > t_alter[last_id]-1-T_id) {return;}
  //determine the id of the thread
	 __shared__ int T_size;
	 __shared__ int T_max;//T_max must be higher than 1
	 __shared__ int nNeurons[2];// cur,  last(need to be altered)
	 __shared__ double threshold;
	 __shared__ double decay1;
	 __shared__ double decay2;
	 // __shared__ double V_0;

	 T_size =  int_param[0];
	 T_max = t_alter[last_id];
	 nNeurons[0] = int_param[1];
	 nNeurons[1] = int_param[2];
	 // nNeurons[2] = int_param[4];
	 threshold = param[0];
	 decay1 = param[1];
	 decay2 = param[2];
	 // V_0 = param[3];


	//to compute all dV_(neuron_id)^(t_j)/dM(S)_(neuron_id)^(T_id)  T_id < t_j <T_max
	last_id = (last_id*nNeurons[0]+ neuron_id)*T_size;// int dV_startid = (last_id*nNeurons[0]+ neuron_id)*T_size;
	neuron_id = neuron_id*T_size;// int h_startid = neuron_id*T_size;

	double dM,dS,d1,d2,tmp,dM_sum_tmp,dS_sum_tmp;
	dM = 1;
	dS = 1;
	d1 = 1;
	d2 = 1;
	tmp = h[neuron_id + T_id]*dVdO[last_id + T_id];
	dM_sum_tmp = tmp*dM;
	dS_sum_tmp = tmp*dS;


	for(int t_j = T_id+1; t_j < T_max;t_j++){
		d1 = d1*decay1;
		d2 = d2*decay2;
		tmp = decay1 - threshold*h[neuron_id + t_j-1];
		dM = d1 + dM*tmp;
		dS = d2 + dS*tmp;
		tmp = h[neuron_id + t_j]*dVdO[last_id + t_j];
		dM_sum_tmp += tmp*dM;
		dS_sum_tmp += tmp*dS;
	}
	dVdM[last_id + T_id] = dM_sum_tmp;
	dVdS[last_id + T_id] = dS_sum_tmp;

	if(T_id == (T_max-1-T_id)){return;}
	T_id = T_max-1-T_id;
	dM = 1;
	dS = 1;
	d1 = 1;
	d2 = 1;
	tmp = h[neuron_id + T_id]*dVdO[last_id + T_id];
	dM_sum_tmp = tmp*dM;
	dS_sum_tmp = tmp*dS;
	for(int t_j = T_id+1; t_j < T_max;t_j++){
		d1 = d1*decay1;
		d2 = d2*decay2;
		tmp = decay1 - threshold*h[neuron_id + t_j-1];
		dM = d1 + dM*tmp;
		dS = d2 + dS*tmp;
		tmp = h[neuron_id + t_j]*dVdO[last_id + t_j];
		dM_sum_tmp += tmp*dM;
		dS_sum_tmp += tmp*dS;
	}
	dVdM[last_id + T_id] = dM_sum_tmp;
	dVdS[last_id + T_id] = dS_sum_tmp;


	// for(int inext = 0;i < nNeurons[1];inext++){
	// 	tmp = h[neuron_id*T_size + t_j]*V_0*weights[nNeurons[0]*inext+neuron_id];
	// 	for(int ilast = 0;i< nNeurons[2];ilast++){
	// 		if(t_j <t_alter[ilast]){
	// 			tmp = dVdMS_lastlayer[(ilast*nNeurons[1]+inext)*T_size + t_j]*tmp;//T_size,next,last
	// 			dVdM[(ilast*nNeurons[0]+neuron_id)*T_size + T_id] += tmp*dM;
	// 			dVdS[(ilast*nNeurons[0]+neuron_id)*T_size + T_id] += tmp*dS;
	// 		}
	// 	}
	// }
 // for(int t_j = T_id;t_j<T_max;t_j++){
	//  dE = dE*(decay1 - threshold*h[neuron_id*T_size + t_j+1]);
	//  tmp = -1*dE*threshold*h[neuron_id*T_size + t_j];
	//  dM += dM+tmp;
	//  dE += dE+tmp;
 // }
	// int i = t_alter[neuron_id]-1;
	// if(i>=0){
	// 	dVdM[startid+i] = dM;
	// 	dVdS[startid+i] = dS;
	// 	i--;
	// }
	// if(i>=0){
 //
	// 	tmp = threshold*h[startid+i];
	// 	dM = decay1*dM -dE*tmp;
	// 	dS = decay2*dS -dE*tmp;
	// 	dVdM[startid+i] = dM;
	// 	dVdS[startid+i] = dS;
	// 	i--;
	// }
 //
	// while(i>=0){
	// 	i--;
	// 	dE = dE*(decay1-tmp);
	// 	tmp = threshold*h[startid+i];
	// 	dM = decay1*dM -dE*tmp;
	// 	dS = decay2*dS -dE*tmp;
	// 	dVdM[startid+i] = dM;
	// 	dVdS[startid+i] = dS;
	// }
 //
	// }



}
