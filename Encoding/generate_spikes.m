% Converting an n-dimension vector of continuous values into n spike
% trains (represented by a T*n matrix, T is the length of time window in the spike trains.) 
% In each spike train 1 indicates a spike event while 0 indicates nothing.
% shceme: encoding methods. 
% 'fixed' is the rate coding, 
% 'poisson' is the possion coding, 
% 'latency' is the latency coding,
% 'order' is the rank-order coding
%
% input: the input vector
% scalor: this parameter controls the spike rate in the 'fixed' and
% 'poisson' schemes
% n: dimension of the input vectors
% T: the length of time window in the converted spike trains

function[spike_trains] = generate_spikes(scheme,input,scalor,n,T)
    switch scheme
        case{'fixed'}
            num_spike = round(T*scalor.*input);
            interval = T./num_spike;

            spike_trains = zeros(T,n);

            for i=1:n
                if(num_spike(i) > 0)
                    spike = ones(num_spike(i),1)*interval(i);
                    pos = int64(cumsum(spike)-interval(i)/2);
                    spike_trains(pos,i) = 1;
                end
            end

        case{'poisson'}
            input = input*scalor;
            spike_trains = zeros(T,n);

            for i=1:n
                if(input(i) ~= 0)
                    isis = -log(rand(round(input(i)*T),1))/(input(i));
                    spkt = int64(cumsum(isis)+1);
                    spike_trains(spkt,i) = 1;
                end
            end

            spike_trains = spike_trains(1:T,:);

        case{'latency'}
              spike_trains = zeros(T,n);
                max_num=max(max(input));
                for i=1:n
                    if input(i) ~= 0
                        spike_trains(T-ceil(T*(input(i)/max_num))+1,i) = 1;
                    end
                end

        case{'order'}
            unique_input=unique(input);
            unique_input=unique_input(unique_input~=0);
            sorted_input=sort(unique_input);
            [index_m,index_n]=size(unique_input);
            interval=T/(index_m*index_n);

            spike_trains = zeros(T,n);

            for i=1:n
                if(input(i)~=0)
                    spike_trains(T - int32(interval*(find(sorted_input==input(i)))) + 1,i) = 1;
%                       spike_trains(int32(interval*(find(sorted_input==input(i)))),i) = 1;
                end
            end
    end
end