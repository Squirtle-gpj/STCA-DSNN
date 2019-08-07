% Computing the number of time steps which make 1 decaay to a
function maxsteps = getMaxsteps(tau,dt,a)
        decay = exp(-(dt/tau));
        tmp = 1;
        maxsteps = 1;
        while(tmp>a)
            maxsteps = maxsteps + 1;
            tmp = tmp*decay;
        end
end