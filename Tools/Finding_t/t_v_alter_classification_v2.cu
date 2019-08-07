#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define maxthread 256
//瀵逛簬姝ｆ牱鏈紝闇?鍙戞斁鑴夊啿鏁?desired锛屽鏋滄病鏈夋弧瓒宠繖绉嶆儏鍐碉紝direction=1锛堝寮烘潈閲嶏級锛宼+浣嶄簬闃堝?涓嬫渶澶х殑鏋佸?鐐?
//瀵逛簬璐熸牱鏈紝闇?淇濇寔闈欓粯锛屾湁涓ょ鎯呭喌锛屽鏋滃彂鏀句簡鑴夊啿鍒?t-浣嶄簬鏈?綆鐨勫彂鏀捐剦鍐茬偣锛屽鏋滄病鏈夊彂鏀捐剦鍐诧紝浣嗘槸鏈?珮闃堝?涓嬫瀬澶у?鐐归珮浜巗ubthreshold锛屽垯t-浣嶄簬璇ユ瀬鍊肩偣
//对于正样本，需要发放脉冲数>desired，如果没有满足这种情况，direction=1（增强权重），t+位于阈值下最大的极值点
//对于负样本，需要保持静默，有两种情况，如果发放了脉冲则 t-位于最低的发放脉冲点，如果没有发放脉冲，但是最高阈值下极大值点高于subthreshold，则t-位于该极值点
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
	 __shared__ double V_prepre[maxthread];//涓婁竴鏃跺埢鐨刅 - 涓婁笂鏃跺埢鐨刅;
	 __shared__ double V_pre[maxthread];//涓婁竴鏃跺埢鐨刅;
	 __shared__ double V_cur[maxthread];//涓婁竴鏃跺埢鐨刅;
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

	// double V_nothr[2] = {0,0};// 0: 涓婁竴鏃跺埢鐨刵othr_V - 涓婁笂涓?椂鍒荤殑nothr_V; 1: 涓婁竴鏃跺埢鐨刵othr_V
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
