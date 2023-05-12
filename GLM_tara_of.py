from __future__ import print_function, division

import numpy as np
import pickle as pkl
from collections import Counter

import random
from scipy import *
import scipy.io
import sys, os

import time
import copy

from scipy.optimize import minimize

input_dir = "/Users/rcubero/Dropbox/Tuce_results/GLM/Tara_OF/"
behave_dir = "/Users/rcubero/Dropbox/Tuce_results/GLM/Tara_OF/"

# Pellet reaching
rodent_files = ["BinOasis__1205_performing",
               "BinOasis__1217_performing",
               "BinOasis__1232_observing",
               "BinOasis__1244_observing"]

rodent_behave = ['Behavioral annotation_1205_new_behaviours',
                'Behavioral annotation_1217_New_behaviours_n',
                'Behavioral annotation_1232_new',
                'Behavioral annotation_1244']


infile = sys.argv[1]
print("Name: ", infile)

file_index = np.array([int(sys.argv[2]), int(sys.argv[3])])
which_files = np.arange(file_index[0], file_index[1]+1, 1)
print("Which files: ", which_files)

captureframerate = sys.argv[4]
print("Capture frame rate: ", captureframerate)

pellet_condition = sys.argv[5]
print(pellet_condition, int(pellet_condition)==1)

if int(pellet_condition)==1:
    PERFORMANCE = True
else:
    PERFORMANCE = False

### currently not used at all!!!
TEMPORALOFFSETS = []
for t in np.arange(0, 21, 10):
    if(t==0):
        TEMPORALOFFSETS.append(t)
    else:
        TEMPORALOFFSETS.append(t)
        TEMPORALOFFSETS.append(-t)
TEMPORALOFFSETS = ravel(TEMPORALOFFSETS)

RANDOMIZETIME = False

BLOCKSHUFFLETIME = True
NUMBLOCKS = int(sys.argv[6])

TIMESHIFT = 0 # in bins

NUMBEROFFOLDS = 10

THRESHOLDFORMAXNUMBEROFSPIKESININTERVAL = 2

GOFULLBINARY = True

SUPERREGPARAM = 0.0 ## REGULARIZER, i.e. log-likelihood - SUPERREGPARAM * sum( parameters except for the "intercept" term )

USEPOISSON = False

basename = '%s_Regression'%(infile)

if(PERFORMANCE):
    basename = '%s_PERF'%basename
else:
    basename = '%s_OBSV'%basename

if SUPERREGPARAM > 0:
    basename = '%s_Regression_REG%05d_R%05d'%(basename, int(floor(-1000*log(SUPERREGPARAM))), round(rand()*10**4))
else:
    basename = '%s_Regression_NOREG_R%05d'%(basename, round(rand()*10**4))

basename = '%s_FOLDS%03d'%(basename, NUMBEROFFOLDS)

if(USEPOISSON):
    basename = '%s_POISSON'%basename
else:
    basename = '%s_BERNOUL'%basename

if(TIMESHIFT>-0.0001):
    basename = '%s_TSHIFT%04d'%(basename, TIMESHIFT)
else:
    basename = '%s_TSHIFTn%04d'%(basename, -TIMESHIFT)

if(BLOCKSHUFFLETIME):
    basename = '%s_%03d_BLK_SHUFF'%(basename, NUMBLOCKS)

if(RANDOMIZETIME):
    basename = '%s_TIME_RAND'%basename
else:
    basename = '%s_NORMAL_TIME'%basename


# Open field task
rodent_tasks = ['Running (CW)', 'Running (CCW)', 'Turning (CW)', 'Turning (CCW)', 'Tail touch', 'Foraging', 'Grooming', 'Eating/social touch', 'Looking down on the edges', 'Stationary', 'Pitch up', 'Squirming', 'Twitching', 'Running(CW)', 'Running(CCW)']

# Load binarized activities
S = np.array([])
for i in list(which_files):
    if S.shape[0]>0:
        S = np.append(S, np.loadtxt("%s%s.csv"%(input_dir,rodent_files[i]), delimiter=',')[:,1:], axis=0)
    else:
        S = np.loadtxt("%s%s.csv"%(input_dir,rodent_files[i]), delimiter=',')[:,1:]
S = S.T
print(S.shape)


# Make cellnames
cellnames = [infile+'_%d'%(i+1) for i in np.arange(S.shape[0])]

# Load behaviors
binning_time = captureframerate
rodent_behaviours = []
for i in list(which_files):
    expt_times = np.loadtxt("%s%s.csv"%(input_dir,rodent_files[0]), delimiter=',')[:,0]
    timestamps = np.zeros(len(expt_times))
    
    behave_names = pkl.load( open(input_dir+rodent_behave[i]+'_names.pkl','rb'), encoding='latin1' )
    behave_times = pkl.load( open(input_dir+rodent_behave[i]+'_times.pkl','rb'), encoding='latin1' )
    
    print(rodent_files[i], behave_names)
    for names in behave_names:
        if names in rodent_tasks:
            task_index = behave_names.index(names)
            
            if names == "Running(CW)": names = "Running (CW)"
            if names == "Running(CCW)": names = "Running (CCW)"

            t_start = [behave_times[task_index][t][0] for t in np.arange(len(behave_times[task_index])) if behave_times[task_index][t][1]==1]
            t_stop = [behave_times[task_index][t][0] for t in np.arange(len(behave_times[task_index])) if behave_times[task_index][t][1]==2]
            
            start_index = np.around((np.around(t_start,2)/0.05),0).astype('int')
            stop_index = np.around((np.around(t_stop,2)/0.05),0).astype('int')
            
            # this is to delete some
            if ((len(start_index)!=0) and (np.array(behave_times[task_index])[:,1][-1]==1)):
                start_index = np.delete(start_index,-1)
            
            if ((len(start_index)!=0) and (np.array(behave_times[task_index])[:,1][0]==2)):
                stop_index = np.delete(stop_index,0)
            
            if (len(start_index)>len(stop_index)):
                zero_index = np.where(np.diff(np.array(behave_times[task_index])[:,1])==0)[0]
                for z_index in zero_index:
                    start_index = np.delete(start_index,np.where(np.around(t_start,2)==np.around(behave_times[task_index][z_index][0],2))[0])
            
            if (len(start_index)<len(stop_index)):
                zero_index = np.where(np.diff(np.array(behave_times[task_index])[:,1])==0)[0]
                for z_index in zero_index:
                    stop_index = np.delete(stop_index,np.where(np.around(t_stop,2)==np.around(behave_times[task_index][z_index+1][0],2))[0])
            
            for behave_time_index in np.arange(len(start_index)):
                timestamps[start_index[behave_time_index]:stop_index[behave_time_index]] = rodent_tasks.index(names)+1

    rodent_behaviours.append(timestamps)

rodent_behaviours = np.array(rodent_behaviours).flatten()
print(len(rodent_behaviours))
print(Counter(rodent_behaviours))

possiblecovariates = {}
for i in np.delete(np.unique(rodent_behaviours),0).astype('int'):
    cov_array = np.zeros_like(rodent_behaviours)
    cov_array[np.where(rodent_behaviours == i)[0]] = 1
    possiblecovariates[rodent_tasks[i-1]] = cov_array
print(possiblecovariates)



whichcovariates = {}

#rodent_tasks = ['Running (CW)', 'Running (CCW)', 'Turning (CW)', 'Turning (CCW)', 'Tail touch', 'Foraging', 'Grooming', 'Eating/social touch',
#'Looking down on the edges', 'Stationary', 'Pitch up', 'Squirming', 'Twitching']

# Open field
whichcombinations = {}
if PERFORMANCE:
   whichcombinations['Running (CW)'] = ['Running (CW)']
   whichcombinations['Running (CCW)'] = ['Running (CCW)']
   whichcombinations['Turning (CCW)'] = ['Turning (CCW)']
   whichcombinations['Turning (CCW)'] = ['Turning (CCW)']
   whichcombinations['Tail touch'] = ['Tail touch']
   whichcombinations['Foraging'] = ['Foraging']
   whichcombinations['Grooming'] = ['Grooming']
   whichcombinations['Eating/social touch'] = ['Eating/social touch']
   whichcombinations['Looking down on the edges'] = ['Looking down on the edges']
   whichcombinations['Stationary'] = ['Stationary']
   whichcombinations['Pitch up'] = ['Pitch up']
else:
   whichcombinations['Running (CW)'] = ['Running (CW)']
   whichcombinations['Running (CCW)'] = ['Running (CCW)']
   whichcombinations['Twitching'] = ['Twitching']
   whichcombinations['Squirming'] = ['Squirming']

listofthingsthatshouldbeprocessed = []
for wc in list(whichcombinations.keys()):
    ara = whichcombinations[wc]
    for guy in ara:
        if(guy not in listofthingsthatshouldbeprocessed):
            listofthingsthatshouldbeprocessed.append(guy)

print(listofthingsthatshouldbeprocessed)




def shiftspikesintime(possiblecovariates, S, toff):
    T = len(S[0,:])
    if(toff<0):
        ii = 0
        jj = T+toff
        aa = abs(toff)
        bb = T
    else:
        ii = toff
        jj = T
        aa = 0
        bb = T-toff
    
    Snew = S[:,ii:jj]
    newposcov = {}
    for k in possiblecovariates:
        newposcov[k] = (possiblecovariates[k])[aa:bb]
    
    return newposcov, Snew


def randomizestuff(possiblecovariates, S):
    goodtimes = list(range(len(S[0,:])))
    random.shuffle(goodtimes)
    neworder = argsort(goodtimes)
    
    Snew = S[:,neworder]
    newposcov = {}
    for k in possiblecovariates:
        newposcov[k] = (possiblecovariates[k])[neworder]
    
    return newposcov, Snew


def getparametersofregressionmodel(P, h, binaryguy, covariates):
    global SUPERREGPARAM
    
    K = shape(covariates)[0]
    T = len(binaryguy)
    m = mean(binaryguy)
    if(P == None):
        h = 0.
        P = zeros(K)
    BC = zeros(K)

    if(sum(np.isnan(P))>0):
        P[np.isnan(P)] = 0.
    
    if(np.isnan(h)):
        h = 0.

    for j in arange(K):
        BC[j] = mean(binaryguy * covariates[j,:])
    
    global tally
    tally = 0
    def singleiter(vals, showvals=False):
        global tally
        global SUPERREGPARAM
        P = ravel(vals[:K])
        h = vals[K]
        H = dot(P,covariates) + h
        expH = exp(H)
        guyH = expH / (1.+ expH)
        dh = (m - mean(guyH))
        dP = zeros(K)
        for j in range(K):
            dP[j] = BC[j] - mean(guyH*covariates[j,:]) - SUPERREGPARAM*sign(P[j]) ## derivative!!
        dvals = np.append( ravel(dP), ravel(dh) )
        L = sum(ravel(binaryguy*H - log(1. + expH))) - SUPERREGPARAM * sum(abs(P))
        if(showvals):
            print(tally, 'L', L, h, max(ravel(P)), min(ravel(P)))
        tally += 1
        return -L, -dvals
    
    def simplegradientdescentofjusth(vals, showvals=False):
        global tally
        P = ravel(vals[:K])
        h = vals[K]
        for i in arange(0,51,1):
            #print i, max(ravel(P)), min(ravel(P)),
            expH = exp(h)
            guyH = expH / (1.+ expH)
            dh = (m - mean(guyH))
            ddh = mean(guyH - guyH**2)
            h += 0.8 * dh/ddh
            if(mod(i,25)==0 and showvals):
                L = sum(ravel(binaryguy*h - log(1. + expH)))
                print(tally, 'L', L, h)
            tally += 1
        vals = np.append( ravel(P), ravel(h) )
        return vals, sum(ravel(binaryguy*h - log(1. + expH)))
    
    def simplegradientdescent(vals, numiters):
        P = ravel(vals[:K])
        h = vals[K]
        for i in arange(0,numiters,1):
            #print i, max(ravel(P)), min(ravel(P)),
            L, dvals = singleiter(vals, False)
            #print L
            dP = ravel(dvals[:K])
            dh = dvals[K]
            P -= 0.1 * dP
            h -= 0.1 * dh
        vals = np.append( ravel(P), ravel(h) )
        return vals, L
    
    #infer with L-BFGS-B
    vals = np.append( ravel(P), ravel(h) )
    vals, Lnull = simplegradientdescentofjusth(vals)
    hnull = vals[K]+0.
    #vals = simplegradientdescent(vals, 6)
    res = minimize(singleiter, vals, method='L-BFGS-B', jac = True,
                   options={'ftol' : 1e-10, 'gtol': 1e-6, 'disp': False})
    vals = res.x + 0.
    vals, Lmod = simplegradientdescent(vals, 2)
    P = ravel(vals[:K])
    h = vals[K]
    return P, h, hnull, Lnull, Lmod


def fitandscorebernoullli(S_fit, finalcovariates_fit, S_test, finalcovariates_test, P, h, predcovs):
    if(sum(S_test)<1):
        print('No spikes in this fold!!!!')
        return np.nan, np.nan, np.nan, P, h, np.nan, np.nan, np.nan
    
    XX = (finalcovariates_fit)
    XXT = (finalcovariates_test)
    
    whichhaveNONzerovar = std(XX, 0) > 0.000000001
    XX = transpose(XX[:,whichhaveNONzerovar])
    XXT = transpose(XXT[:,whichhaveNONzerovar])
    if(P != None):
        beta_ = P[whichhaveNONzerovar]
    predcovs = predcovs[:,whichhaveNONzerovar]
    predcovs = predcovs[whichhaveNONzerovar,:]
    
    if(GOFULLBINARY):
        S_fit = (S_fit>0.5)*1. ## binarize it!
        S_test = (S_test>0.5)*1. ## binarize it!

    P, h, hnull, Lnullfit, Lmodfit = getparametersofregressionmodel(P, h, S_fit, XX)

    H = dot(P,XXT) + h
    Lmod = sum(ravel(S_test*H - log(1. + exp(H)) ))

    y_hat = exp(H) / (1.+ exp(H))

    H = hnull
    Lnull = sum(ravel(S_test*H - log(1. + exp(H)) ))
    
    pseudoR2 = 1. - Lmod / Lnull
    llratio_per_spike = (Lmod - Lnull) / sum(S_test)
    print('R', SUPERREGPARAM, 'Lfit', Lmodfit, Lnullfit, 'Ltest', Lmod, Lnull, 'min and max P', np.nanmin(P), np.nanmax(P), "PR2", pseudoR2, "Nspikes", sum(S_test), "Length", len(S_test))
    
    
    # pR2 is fGLM_Regressionor bernoulli this is 1 - Lmodel / Lnull
    # llration per spike is (Lmodel - Lnull)/sum(y)
    # for bernoulli this is 1 - Lmodel / Lnull
    
    H = dot(P,transpose(predcovs)) + h
    ypreds = exp(H) / (1.+ exp(H))
    if(len(ypreds)<len(whichhaveNONzerovar)):
        allypreds = zeros(len(whichhaveNONzerovar))
        allypreds[:] = np.nan
        allypreds[whichhaveNONzerovar] = ypreds
        ypreds = allypreds
    
    
    if(len(P)<len(whichhaveNONzerovar)):
        allbeta_ = zeros(len(whichhaveNONzerovar))
        allbeta_[:] = np.nan
        allbeta_[whichhaveNONzerovar] = P
        P = allbeta_
    
    return ypreds, pseudoR2, llratio_per_spike, P, h, Lmod, Lnull, sum(S_test)


def doinference(finalcovariates, S, daparams=None):
    global SUPERREGPARAM
    N = len(finalcovariates[0,:])
    K = NUMBEROFFOLDS
    T = len(finalcovariates[:,0])
    
    print(T)
    
    TENPERCENT = int(round( (1./float(K)) * float(len(finalcovariates[:,0])) ))
    THEREST = len(finalcovariates[:,0]) - TENPERCENT
    
    predvals = zeros((K, N))
    pseudoR2 = zeros(K)
    normllratio = zeros(K)
    Lmodel = zeros(K)
    Lnull = zeros(K)
    sum_y = zeros(K)
    beta = zeros((K, N))
    beta0 = zeros(K)
    
    #checkregparams = ravel(array([-0.0005,-0.00025,-0.0001,-0.00001,0]))
    #TODELETEpseudoR2 = zeros((len(checkregparams),K))
    #TODELETEnormllratio = zeros((len(checkregparams),K))
    
    ###### CENTER VARIABLES!!!
    valuesforpredicting = zeros((N,N))
    for i in range(N):
        if(std(finalcovariates[:,i])>0.):
            valuesforpredicting[i,i] = (1-mean(finalcovariates[:,i])) / std(finalcovariates[:,i])
            finalcovariates[:,i] = (finalcovariates[:,i] -  mean(finalcovariates[:,i])) / std(finalcovariates[:,i])


    for i in range(K):
        testinds = zeros(T)>1
        if(BLOCKSHUFFLETIME == False):
            if(i == 0):
                testinds[:TENPERCENT] = True
            elif(i == K-1):
                testinds[THEREST:] = True
            else:
                testinds[(i*TENPERCENT):((i+1)*TENPERCENT)] = True
        else:
            chunksize = int( floor( float(T) / float(NUMBLOCKS) ) + 1 )
            for j in range(NUMBLOCKS):
                testchunksize = int(round( float(chunksize) / float(K) ))
                iii = j*chunksize + i*testchunksize
                jjj = min([T, j*chunksize + (i+1)*testchunksize])
                testinds[iii:jjj] = True

        Sfit = S[~testinds]+0.
        Stest = S[testinds]+0.
        finalcovariatesfit = finalcovariates[~testinds, :]+0.
        finalcovariatestest = finalcovariates[testinds, :]+0.

        print('Fold', i, 'L(fit)', len(Sfit), 'S(fit)', sum(Sfit), 'L(test)', len(Stest), 'S(test)', sum(Stest), 'S(testinds)', sum(testinds), T)

        P = None
        h = None
        
        predvals[i,:], pseudoR2[i], normllratio[i], P, h, Lmodel[i], Lnull[i], sum_y[i] = fitandscorebernoullli(Sfit, finalcovariatesfit, Stest, finalcovariatestest, P, h, valuesforpredicting)
        beta0[i] = h
        beta[i,:] = P
        if(i>0):
            daparams = [mean(beta0[:(i+1)]), nanmean(beta[:(i+1),:], 0)]

    params = zeros((1, len( predvals[0,:] )))
    params[0,:] = nanmean(predvals,0)
    print('P R2', pseudoR2[:], mean(pseudoR2[:]), std(pseudoR2[:]))
    return params, pseudoR2, normllratio, daparams, beta, Lmodel, Lnull, sum_y



print('Here are the keys in the mat file:')
ckeys = sort(list(possiblecovariates.keys()))
for k in ckeys:
    print(k)


if(abs(TIMESHIFT)>0.00001):
    possiblecovariates, S = shiftspikesintime(possiblecovariates, S, TIMESHIFT)

if(RANDOMIZETIME):
    possiblecovariates, S = randomizestuff(possiblecovariates, S)

ckeys = list(possiblecovariates.keys())

allresultskeys = zeros((len(cellnames), len(list(whichcombinations.keys()))))

outputdict = {}
outputdict['filename'] = infile
outputdict['cellnames'] = cellnames

for icell in range(len(cellnames)):
    cname = cellnames[icell]
    Sorig = S[icell,:] + 0.
    singlecellresultskeys = []
    
    kkk = sort(list(whichcombinations.keys()))
    for j in range(len(kkk)):
        print(('Going for', kkk[j], 'with cell', cname))
        
        thekeyname = kkk[j]
        goodkeys = whichcombinations[thekeyname]
        covs = []
        chkkey = goodkeys[0]
        covs = possiblecovariates[chkkey]
        if len(goodkeys)>1:
            for i in range(1,len(goodkeys),1):
                chkkey = goodkeys[i]
                newcov = possiblecovariates[chkkey]
                covs = np.vstack([covs, newcov])
        covs = np.vstack([covs, np.ones(len(covs.T))])
        covs = covs.T
        print("Covariants shape: ", covs.shape)
        
        params, pr2value, normLLratio, dummy, b_params, Lmodel, Lnull, sum_y = doinference(covs, Sorig)
        
        tempkey = '%d'%int(round(1000000.*time.time()))
        allresultskeys[icell, j] = tempkey
        outputdict['%s-cellname'%tempkey] = cname
        outputdict['%s-combokey'%tempkey] = thekeyname
        outputdict['%s-ingreds'%tempkey] = copy.deepcopy(goodkeys)
        outputdict['%s-pr2_score'%tempkey] = pr2value+0.
        outputdict['%s-nLLr_score'%tempkey] = normLLratio+0.
        outputdict['%s-params' % tempkey] = b_params + 0.
        outputdict['%s-Lmodel' % tempkey] = Lmodel + 0.
        outputdict['%s-Lnull' % tempkey] = Lnull + 0.
        outputdict['%s-sum_y' % tempkey] = sum_y + 0.

outputdict['allresultskeys'] = allresultskeys

if(len(cellnames)>0):
    scipy.io.savemat('%s.mat'%(basename), outputdict)
