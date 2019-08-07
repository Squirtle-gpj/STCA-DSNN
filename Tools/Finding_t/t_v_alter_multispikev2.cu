#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void t_v_alter_multispikev2( double *output, double *u, int *t_alter, double *direction,const int *desired,const double * input, const int T_size, const int Numcurlayer,  const double decay1, const double decay2, const double threshold, const double subthreshold)
{


	int neuron_id = blockIdx.x*blockDim.x +threadIdx.x;
	if(neuron_id > Numcurlayer - 1 ) {return;}
	int curid = neuron_id*T_size;
  int endid = curid + T_size;
  double m = 0;
  double s = 0;
  double e = 0;
  double V = 0;
	int nSpikes = 0;
	double up_min = 1.7976931348623158e+308;
	double down_max = 0;
	double t_up_min = endid-1;
	double t_down_max = t_up_min;
	double V_nothr[2] = {0,0};// 0: 上一时刻的nothr_V - 上上一时刻的nothr_V; 1: 上一时刻的nothr_V
	bool fired_pre = false;


    while(curid < endid)
    {
			m = m*decay1;
			s = s*decay2;
			//now, V is a tmp
			V = input[curid];
			if(V != 0)
			{
				m = m + V;
				s = s + V;
			}
			e = e*decay1;
			if (fired_pre)
			{
				e = e + threshold;
			}
			// now, fired_pre is fired_cur.

			V = m -s - e;
			u[curid] = V;
			fired_pre = (V > threshold);\
			if(fired_pre)
			{
				output[curid] = 1.0;
				nSpikes++;

			}

			if((V_nothr[0]>0)&&(V_nothr[1] >V)){
				// output[curid] = 1.0;
				if((V_nothr[1]>threshold)&&(V_nothr[1]<up_min)){
					// output[curid] = 2.0;
					up_min = V_nothr[1];
					t_up_min = curid-1;
				}
				else if((V_nothr[1]<=threshold)&&(V_nothr[1]>down_max)){
					// output[curid] = 3.0;
					down_max = V;
					t_down_max = curid-1;
				}

			}
			V_nothr[0] = V - V_nothr[1];
			V_nothr[1] = V;
			curid++;

    }


	if(nSpikes > desired[neuron_id]){
		direction[neuron_id] = -1;
		t_alter[neuron_id] = t_up_min -endid + T_size+1;//transform to matlab
	}
	else if(nSpikes < desired[neuron_id]){
		direction[neuron_id] = 1;
		t_alter[neuron_id] = t_down_max -endid + T_size+1;
	}


}
