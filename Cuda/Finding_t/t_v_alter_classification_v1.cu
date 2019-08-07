#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define maxthread 256
__global__ void t_v_alter_classification( double *output, double *u, int *t_alter, double *direction,const int *desired,const double * input, const int* int_param,const double *param)
{

	 if(threadIdx.x > maxthread - 1){ return;}
	 int neuron_id = blockIdx.x*blockDim.x +threadIdx.x;
	if(neuron_id > int_param[1] - 1 ) {return;}

	 __shared__ int T_size;
	 // __shared__ int Numcurlayer;
	 __shared__ double threshold;
	 __shared__ double subthreshold;
	 __shared__ double decay1;
	 __shared__ double decay2;
	 __shared__ double V_prepre[maxthread];//上一时刻的V - 上上时刻的V;
	 __shared__ double V_pre[maxthread];//上一时刻的V;
	 __shared__ double V_cur[maxthread];//上一时刻的V;
	 T_size =  int_param[0];
	 // Numcurlayer = int_param[1];
	 threshold = param[0];
	 subthreshold = param[1];
	 decay1 = param[2];
	 decay2 = param[3];

	int curid = neuron_id*T_size;
  int endid = curid + T_size;
  double m = 0;
  double s = 0;
  double e = 0;
  // V_cur[threadIdx.x] = 0;
	bool isfired = false;
	// double up_min = 1.7976931348623158e+308;
	double down_max = 0;
	int t_up_min = endid-1;
	int t_down_max = t_up_min;
	// double V_nothr[2] = {0,0};// 0: 上一时刻的nothr_V - 上上一时刻的nothr_V; 1: 上一时刻的nothr_V
	bool fired_pre = false;

	V_prepre[threadIdx.x] = 0;
	V_pre[threadIdx.x] = 0;
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

			V_cur[threadIdx.x] = m -s - e;
			u[curid] = V_cur[threadIdx.x];
			fired_pre = (V_cur[threadIdx.x] > threshold);
			if(fired_pre)
			{
				isfired = true;
				output[curid] = 1.0;
				// nSpikes++;
				// if(V_pre[threadIdx.x] < up_min)
				// { up_min = V_pre[threadIdx.x];
				// 	t_up_min = curid-1;
				// }
				t_up_min = curid-1;
			}else if((V_prepre[threadIdx.x] <  V_pre[threadIdx.x])&&(V_pre[threadIdx.x] >V_cur[threadIdx.x])){
				// output[curid] = 1.0;
				if(V_pre[threadIdx.x]>down_max){
					// output[curid] = 3.0;
					down_max = V_pre[threadIdx.x];
					t_down_max = curid-1;
				}

			}
			V_prepre[threadIdx.x] = V_pre[threadIdx.x];
			V_pre[threadIdx.x] = V_cur[threadIdx.x];
			curid++;

    }

	if(desired[neuron_id]==1){
		if(isfired&&(down_max <(threshold - subthreshold))){
			;
		}else{
			direction[neuron_id] = 1.0;
			t_alter[neuron_id] = t_down_max-endid + T_size+1;
		}
	}else{

		if(isfired){
			direction[neuron_id] = -1.0;
			t_alter[neuron_id] = t_up_min-endid + T_size+1;
		}else if(down_max >(threshold - subthreshold))
		{
			direction[neuron_id] = -1.0;
			t_alter[neuron_id] = t_down_max-endid + T_size+1;
		}
	}
	// direction[neuron_id] = t_up_min-endid + T_size+1;
	// if(nSpikes > desired[neuron_id]){
	// 	direction[neuron_id] = -1;
	// 	t_alter[neuron_id] = t_up_min -endid + T_size+1;//transform to matlab
	// }
	// else if(nSpikes < desired[neuron_id]){
	// 	direction[neuron_id] = 1;
	// 	t_alter[neuron_id] = t_down_max -endid + T_size+1;
	// }


}
