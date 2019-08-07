#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//t+浣嶄簬鍙戞斁鑴夊啿鏁版渶灏戠殑spike cluster鐨勬渶鍚庝竴涓剦鍐插彂鏀剧偣
//t-浣嶄簬涓嶅湪spike_cluster涓殑闃堝?涓嬫渶楂樻瀬澶у?鐐?
//t+位于发放脉冲数最少的spike cluster的最后一个脉冲发放点
//t-位于不在spike_cluster中的阈值下最高极大值点
__global__ void t_v_alter_multispikev3( double *output, double *u, int *nSpikeClusters,int *t_alter, double *direction,const int *desired,const double * input, const int T_size, const int Numcurlayer,  const double decay1, const double decay2, const double threshold, const int interval)
{


	int neuron_id = blockIdx.x*blockDim.x +threadIdx.x;
	if(neuron_id > Numcurlayer - 1 ) {return;}
	int curid = neuron_id*T_size;
  int endid = curid + T_size;
  double m = 0;
  double s = 0;
  double e = 0;
  double V = 0;
	int nSpike_clusters = 0;
	double nSpikes = 0;
	double up_min = 1.79769313486231570E+308;
	double down_max = -1.79769313486231570E+308;
	double t_up_min = endid-1;
	double t_down_max = t_up_min;
	double V_nothr[2] = {0,0};// 0: 涓婁竴鏃跺埢鐨刵othr_V - 涓婁笂涓?椂鍒荤殑nothr_V; 1: 涓婁竴鏃跺埢鐨刵othr_V
	bool fired_pre = false;
	bool incluster = false;
	int dur = 0;


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
			fired_pre = (V > threshold);
			if(fired_pre)
			{
				output[curid] = 1.0;
				incluster = true;
				dur = 0;
				nSpikes += V;
				 // nSpikes++;

			}else{
				dur++;
			}
			if(((dur>=interval)||(curid >=(endid-1)))&&(incluster == true)){
				nSpike_clusters++;
				incluster = false;
				if(nSpikes < up_min){
					up_min = nSpikes;
					t_up_min = curid-1-dur;
				}
				nSpikes = 0;

			}

			if((V_nothr[0]>0)&&(V_nothr[1] >V)){
				// output[curid] = 1.0;
				// if((V_nothr[1]>threshold)&&(V_nothr[1]<up_min)){
				// 	// output[curid] = 2.0;
				// 	up_min = V_nothr[1];
				// 	t_up_min = curid-1;
				// }
				// else
				if((V_nothr[1]<=threshold)&&(V_nothr[1]>down_max)&&(incluster == false)){
					// output[curid] = 3.0;
					down_max = V;
					t_down_max = curid-1;
				}

			}
			V_nothr[0] = V - V_nothr[1];
			V_nothr[1] = V;
			curid++;

    }

	nSpikeClusters[neuron_id] = nSpike_clusters;
	if(nSpike_clusters > desired[neuron_id]){
		direction[neuron_id] = -1;
		t_alter[neuron_id] = t_up_min -endid + T_size+1;//transform to matlab
	}
	else if(nSpike_clusters < desired[neuron_id]){
		direction[neuron_id] = 1;
		t_alter[neuron_id] = t_down_max -endid + T_size+1;
	}


}
