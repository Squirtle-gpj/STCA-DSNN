#include "math_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void compute_u_kernel( double *u, int *t_alter, double *direction,const double * input, const int* cur_Groups,const int T_size, const int Numcurlayer,  const double decay1, const double decay2, const double threshold, const double subthreshold)
{


	int afferent_id = blockIdx.x*blockDim.x +threadIdx.x;
	if(afferent_id > Numcurlayer - 1 ) {return;}
	int curid = afferent_id*T_size;
    int endid = curid + T_size;
    double m = 0;
    double s = 0;
    // double e = 0;
    double V = 0;
		double tmax = 0;
		double vmax = 0;
		double down_thr = threshold -subthreshold;
		double up_thr = threshold + subthreshold;
    // bool fired_pre = false;


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
			// e = e*decay1;
			// if (fired_pre)
			// {
			// 	e = e + threshold;
			// }
			// now, fired_pre is fired_cur.

			V = m -s;
			if(V > vmax)
			{
				tmax = curid - endid + T_size + 1;
				vmax = V;
			}
			u[curid] = V;
			// fired_pre = (V > threshold);
			// if(fired_pre)
			// {
			// 	output[curid] = 1.0;
			// }
			curid++;

    }


		//V is tmp.
		if((vmax > down_thr) && (cur_Groups[afferent_id] ==0))
		{
			direction[afferent_id] = -1;
			t_alter[afferent_id] = tmax;
		}
		else if(vmax <= up_thr && cur_Groups[afferent_id] ==1)
		{
			direction[afferent_id] = 1;
			if(vmax == 0)
			{
				t_alter[afferent_id] = T_size;
			}
			else
			{
				t_alter[afferent_id] = tmax;
			}
		}


}
