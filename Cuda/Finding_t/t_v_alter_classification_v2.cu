#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define maxthread 256
//对于正样本，�?��发放脉冲�?desired，如果没有满足这种情况，direction=1（增强权重），t+位于阈�?下最大的极�?�?
//对于负样本，�?��保持静默，有两种情况，如果发放了脉冲�?t-位于�?��的发放脉冲点，如果没有发放脉冲，但是�?��阈�?下极大�?点高于subthreshold，则t-位于该极值点
//��������������Ҫ����������>desired�����û���������������direction=1����ǿȨ�أ���t+λ����ֵ�����ļ�ֵ��
//���ڸ���������Ҫ���־�Ĭ���������������������������� t-λ����͵ķ�������㣬���û�з������壬���������ֵ�¼���ֵ�����subthreshold����t-λ�ڸü�ֵ��
__global__ void t_v_alter_classification( double *output, double *u, int *t_alter, double *direction,const int *cur_Groups,const double * input, const int* int_param,const double *param)
{

	 if(threadIdx.x > maxthread - 1){ return;}
	 int neuron_id = blockIdx.x*blockDim.x +threadIdx.x;
	if(neuron_id > int_param[1] - 1 ) {return;}

	 __shared__ int T_size;
	  __shared__ int desired;
	 __shared__ double threshold;
	 __shared__ double subthreshold;
	 __shared__ double decay1;
	 __shared__ double decay2;
	 __shared__ double V_prepre[maxthread];//上一时刻的V - 上上时刻的V;
	 __shared__ double V_pre[maxthread];//上一时刻的V;
	 __shared__ double V_cur[maxthread];//上一时刻的V;
	 __shared__ double  up_min[maxthread];
	 __shared__ double down_max[maxthread];
	 __shared__ double nSpikes[maxthread];
	 T_size =  int_param[0];
	 desired = int_param[2];
	 threshold = param[0];
	 subthreshold = threshold- param[1];
	 decay1 = param[2];
	 decay2 = param[3];

	int curid = neuron_id*T_size;
  int endid = curid + T_size;
  double m = 0;
  double s = 0;
  double e = 0;
  // V_cur[threadIdx.x] = 0;
	// bool isfired = false;
	up_min[threadIdx.x] = 1.7976931348623158e+308;
	down_max[threadIdx.x] = 0;
	nSpikes[threadIdx.x] = 0;
	int t_up_min = endid-1;
	int t_down_max = t_up_min;

	// double V_nothr[2] = {0,0};// 0: 上一时刻的nothr_V - 上上�?��刻的nothr_V; 1: 上一时刻的nothr_V
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
				// isfired = true;
				output[curid] = 1.0;
				nSpikes[threadIdx.x]++;
				if(V_pre[threadIdx.x] < up_min[threadIdx.x])
				{
					up_min[threadIdx.x] = V_pre[threadIdx.x];
					t_up_min = curid-1;
				}
				// t_up_min = curid-1;
			}else if((V_prepre[threadIdx.x] <  V_pre[threadIdx.x])&&(V_pre[threadIdx.x] >V_cur[threadIdx.x])){
				// output[curid] = 1.0;
				if(V_pre[threadIdx.x]>down_max[threadIdx.x]){
					// output[curid] = 3.0;
					down_max[threadIdx.x] = V_pre[threadIdx.x];
					t_down_max = curid-1;
				}

			}
			V_prepre[threadIdx.x] = V_pre[threadIdx.x];
			V_pre[threadIdx.x] = V_cur[threadIdx.x];
			curid++;

    }

	if((cur_Groups[neuron_id]==1)){
		if(nSpikes[threadIdx.x]<desired){
			direction[neuron_id] = 1;
			t_alter[neuron_id] = t_down_max -endid + T_size+1;
		}


	}else{
		if(nSpikes[threadIdx.x] > 0){
				direction[neuron_id] = -1;
				t_alter[neuron_id] = t_up_min -endid + T_size+1;
		}else if(down_max[threadIdx.x] >subthreshold){
			direction[neuron_id] = -1;
			t_alter[neuron_id] = t_down_max -endid + T_size+1;
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
