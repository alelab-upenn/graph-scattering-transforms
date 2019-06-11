# 2019/05/20~23
# Fernando Gama, fgama@seas.upenn.edu

# Graph scattering transform.

# Compute the representation difference on a small world graph.
# Representations considered:
#   GFT: unstable graph-dependent representation
#   Diffusion scattering: comparison with other works
#   Monic cubic polynomial wavelet: Hammond et al wavelets
#   Tight Hann wavelets: Shuman et al wavelets
# Theoretical bound obtained also shown in the graphs.
# The idea is just to show how the representation difference changes with
# perturbations of different size, and show that the representation difference
# in stable GST is (way) smaller than in the (unstable) GFT.

#%%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

#\\\ Standard libraries:
import os
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import pickle
import datetime

#\\\ Own libraries:
import Modules.graphScattering as GST
import Utils.graphTools as graphTools

#\\\ Own separate functions:
from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed

#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

thisFilename = 'GSTsmallWorld' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location where to save
    # anything that might need to be saved out of the run
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all the
    # results from each run

#\\\ Create .txt to store the values of the parameters of the setting for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # Append date and time of the run to the directory, to avoid several runs of
    # overwritting each other.
saveDir = saveDir + today
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
# Create the file where all the (hyper)parameters are results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

#\\\ Save seeds for reproducibility
#   Numpy seeds
numpyState = np.random.RandomState().get_state()
#   Collect all random states
randomStates = []
randomStates.append({})
randomStates[0]['module'] = 'numpy'
randomStates[0]['state'] = numpyState
#   This list and dictionary follows the format to then be loaded, if needed,
#   by calling the loadSeed function in Utils.miscTools
saveSeed(randomStates, saveDir)

########
# DATA #
########

nNodes = 100 # Number of nodes
graphType = 'SmallWorld' # Type of graph
graphOptions = {} # Dictionary of options to pass to the createGraph function
if graphType == 'SBM':
    graphOptions['nCommunities'] = 5 # Number of communities
    graphOptions['probIntra'] = 0.8 # Intracommunity probability
    graphOptions['probInter'] = 0.2 # Intercommunity probability
elif graphType == 'SmallWorld':
    graphOptions['probEdge'] = 0.5 # Edge probability
    graphOptions['probRewiring'] = 0.1 # Probability of rewiring

nTest = 1000 # Number of testing samples
nSimPoints = 10 # Number of simulation points (x-axis)
signalPower = 1. # Base variance of (random samples)
perturbationEpsilon = signalPower * np.arange(1, nSimPoints+1)/float(nSimPoints)
    # Value epsilon of the perturbation, it is relative to the signal power
    # and it goes from 1/nSimPoints to 1

nPerturbationRealizations = 10 # Number of realizations of the perturbation
    # Each perturbation is random, so how many different perturbations we want
    # to run before we average the results
nGraphRealizations = 10 # Number of graph realizations
    # Each graph is random, so how many graphs to create to average the results
    # The randomization of the graphs is the one that is plotted as error bars
    # (i.e. how much the performance changes with different graphs within
    # the same family)

#\\\ Save values:
writeVarValues(varsFile, {'nNodes': nNodes, 'graphType': graphType})
writeVarValues(varsFile, graphOptions)
writeVarValues(varsFile, {'nTest': nTest,
                          'nSimPoints': nSimPoints,
                          'signalPower': signalPower,
                          'nPerturbationRealizations':nPerturbationRealizations,
                          'nGraphRealizations': nGraphRealizations})

#################
# ARCHITECTURES #
#################

# Select which wavelets to use
doDiffusion = True # F. Gama, A. Ribeiro, and J. Bruna, "Diffusion scattering
    # transforms on graphs,” in Int. Conf. Learning Representations 2019.
    # New Orleans, LA: Assoc. Comput. Linguistics, 6-9 May 2019.
doMonicCubic = True # Eq. (65) in D. K. Hammond, P. Vandergheynst, and
    # R. Gribonval, "Wavelets on graphs via spectral graph theory," Appl.
    # Comput. Harmonic Anal., vol. 30, no. 2, pp. 129–150, March 2011.
doTightHann = True # Example 2, p. 4226 in D. I. Shuman, C. Wiesmeyr,
    # N. Holighaus, and P. Vandergheynst, "Spectrum-adapted tight graph wavelet
    # and vertex-frequency frames,” IEEE Trans. Signal Process., vol. 63,
    # no. 16, pp. 4223–4235, Aug. 2015.
doGFT = True # Compare against the GFT which is a (unstable) representation that
    # also depends on the graph
normalizeGSOforGFT = True # The GSO for the GFT is the Laplacian (if possible),
    # if not, it becomes the adjacency matrix. In either case, setting True
    # to this flag, gets the GSO normalized. Since we're usually comparing
    # against normalized matrix descriptions in the other cases, we give this
    # options.
computeBound = True # Compute the theoretical bound and show it (dashed) in the
    # final figure (the bound computed is for the monic cubic polynomial
    # wavelets due to easier closed-form expression that allows for
    # straightforward computation of the derivatives)

numScales = 6 # Number of scales J (the first element might be the "low-pass"
    # wavelet) so we would get J-1 "wavelet scales" and 1 (the first one, j=0)
    # "low-pass" wavelet
numLayers = 3 # Number of layers L (0, ..., L-1) with l=0 being just Ux

#\\\ Save values:
writeVarValues(varsFile, {'numScales': numScales, 'numLayers': numLayers})

modelList = [] # List to store the list of models chosen

# Obs.: These are the names that will appear in the legend of the figure
if doDiffusion:
    diffusionName = 'Diffusion'
    modelList.append(diffusionName)
if doMonicCubic:
    monicCubicName = 'Monic Cubic'
    modelList.append(monicCubicName)
if doTightHann:
    tightHannName = 'Tight Hann'
    modelList.append(tightHannName)
if doGFT:
    GFTname = 'GFT'
if computeBound:
    boundName = 'Bound'

###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running
doSaveVars = True # Save (pickle) useful variables
doFigs = True # Plot some figures (this only works if doSaveVars is True)
figSize = 5 # Overall size of the figure that contains the plot
lineWidth = 2 # Width of the plot lines
markerShape = 'o' # Shape of the markers
markerSize = 3 # Size of the markers

#\\\ Save values:
writeVarValues(varsFile,
               {'doPrint': doPrint,
                'doSaveVars': doSaveVars,
                'doFigs': doFigs,
                'figSize': figSize,
                'lineWidth': lineWidth,
                'markerShape': markerShape,
                'markerSize': markerSize,
                'saveDir': saveDir})

#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

#\\\ Save variables during evaluation.

reprError = {} # Representation error
    # This is a dictionary where each key corresponds to one of the models,
    # and each element in the dictionary is a list of lists storing the
    # mean representation error (averaged across nTest) for each graph
    # realization, for each perturbation value, and for each perturbation
    # realization.
for thisModel in modelList: # First list is for each graph realization
    reprError[thisModel] = [None] * nGraphRealizations
# The GFT representation error is on a different variable since it cannot be
# computed in the same for loop as the rest of the model (that would require
# creating a "scattering GFT" which makes no sense)
if doGFT:
    reprErrorGFT = [None] * nGraphRealizations
# Store the values of the bound for each realization
if computeBound:
    bound = [None] * nGraphRealizations

#%%##################################################################
#                                                                   #
#                    GRAPH REALIZATION                              #
#                                                                   #
#####################################################################

# Start generating a new graph for each of the number of graph realizations that
# we previously specified.

for graph in range(nGraphRealizations):

    if doPrint:
        print("Graph realization no. %d" % (graph+1))

    # The reprError variable, for each model, has a list with a total number of
    # elements equal to the number of graphs we will generate
    # Now, for each graph, we have multiple perturbation values (epsilon values)
    # so we want, for each graph, to create a list to hold each of those values
    for thisModel in modelList:
        reprError[thisModel][graph] = [None] * len(perturbationEpsilon)
    # Repeat for the GFT error
    if doGFT:
        reprErrorGFT[graph] = [None] * len(perturbationEpsilon)
    # The bound also depends on the specific value of epsilon so we also need
    # this
    if computeBound:
        bound[graph] = [None] * len(perturbationEpsilon)

    #%%##################################################################
    #                                                                   #
    #                    GRAPH CREATION                                 #
    #                                                                   #
    #####################################################################

    # Create graph
    G = graphTools.Graph(graphType, nNodes, graphOptions)

    #%%##################################################################
    #                                                                   #
    #                    GRAPH SCATTERING MODELS                        #
    #                                                                   #
    #####################################################################

    modelsGST = {} # Store each model as a key in this dictionary, then we can
        # can compute the output for each model inside a for (iterating over
        # the key), since all models have a computeTransform() method.

    if doDiffusion:
        modelsGST[diffusionName] = \
                              GST.DiffusionScattering(numScales, numLayers, G.W)

    if doMonicCubic:
        modelsGST[monicCubicName] = GST.MonicCubic(numScales, numLayers, G.W)

    if doTightHann:
        modelsGST[tightHannName] = GST.TightHann(numScales, numLayers, G.W)

    # Note that monic cubic polynomials and tight Hann's wavelets have other
    # parameters that are being set by default to the values in the respective
    # papers.

    # We want to determine which eigenbasis to use. We try to use the Laplacian
    # since it's the same used in the wavelet cases, and seems to be the one
    # holding more "interpretability". If the Laplacian doesn't exist (which
    # could happen if the graph is directed or has negative edge weights), then
    # we use the eigenbasis of the adjacency.
    if doGFT:
        if G.L is not None:
            S = G.L
            if normalizeGSOforGFT:
                S = graphTools.normalizeLaplacian(S)
            _, GFT = graphTools.computeGFT(S, order = 'increasing')
        else:
            S = G.W
            if normalizeGSOforGFT:
                S = graphTools.normalizeAdjacency(S)
            _, GFT = graphTools.computeGFT(S, order = 'totalVariation')

        GFT = GFT.conj().T

    # The bound will only be computed for the monic cubic polynomials (easier
    # closed-form expression), so we need to be sure that it is here to compute
    # the bound.
    if computeBound and doMonicCubic:
        # B is the maximum value of the filter on each scale (vector of size J)
        B = modelsGST[monicCubicName].getFilterBound()
        # Pick the maximum filter bound
        B = np.max(np.abs(B))
        # C is the integral Lipschitz constant for each filter (vector of size J)
        C = modelsGST[monicCubicName].getIntegralLipschitzConstant()
        # Pick the maximum
        C = np.max(np.abs(C)) # Get the maximum of each J
        # We now have to compute xiBJL which we will do by definition (i.e. the
        # sum), instead of the result. So we need to create the values of ell
        l = np.arange(0, numLayers) # l = 0, ..., L-1
        # And then compute the different sums (xiBJL is an array of shape 3,
        # where each element corresponds to one of the xiBJL constants)
        xiBJL = np.array(
                [np.sum(((B ** 2) * numScales) ** l),
                 np.sum(l * (((B ** 2) * numScales)**l)),
                 np.sum((l**2) * (((B ** 2) * numScales)**l))]
                )
        # The maximum value of U to compute its bound
        BU = np.max(np.abs(modelsGST[monicCubicName].U))

    # So far, we have created the models and computed all the necessary
    # quantities that depend on the unperturbed graph, now we have to start
    # checking each value of epsilon, and creating a nPerturbationRealizations
    # of random changes in the edges.

    #%%##################################################################
    #                                                                   #
    #                    PERTURBATION NOISE                             #
    #                                                                   #
    #####################################################################

    for itEpsilon in range(len(perturbationEpsilon)):

        # For each value of epsilon, we generate nPerturbationRealizations, and
        # for each of these realizations, we generate nTest white noise samples.
        epsilon = perturbationEpsilon[itEpsilon] # value of epsilon

        if doPrint:
            print("\tPerturbation value %.4f (no. %d)"%(epsilon, (itEpsilon+1)))

        # This is the third list: for each graph and each value of epsilon
        # we run nPerturbationRealizations and we store the values of each one
        # of those.
        for thisModel in modelList:
            reprError[thisModel][graph][itEpsilon] = \
                                              [None] * nPerturbationRealizations
        if doGFT:
            reprErrorGFT[graph][itEpsilon] = [None] * nPerturbationRealizations
        if computeBound:
            bound[graph][itEpsilon] = [None] * nPerturbationRealizations

        # For each random perturbation
        for perturbation in range(nPerturbationRealizations):

            # Perturbation matrix E such that ||E|| < epsilon/2 and
            # ||E/m_N - I|| < epsilon
            E = np.random.uniform(1-epsilon/2, 1+epsilon/2, nNodes) # ||E-I||<e
            E = np.diag(-epsilon/(2*(1+epsilon)) * E) # biggest absolute number
                # is 1+epsilon and we want this number to be less than epsilon/2
                # so that a (1+epsilon) <= epsilon/2
            # Compute What = W + E^H * W + W * E
            What = G.W + E.conj().T.dot(G.W) + G.W.dot(E)
                # We use this adjacency matrix as input to the methods which
                # then compute the appropriate representation of the graph

            # In the GFT case, we have to use this perturbed adjacency matrix
            # to obtain the appropriate eigenbasis (Laplacian or not,
            # normalized or not)
            if doGFT:
                if G.L is not None:
                    Shat = graphTools.adjacencyToLaplacian(What)
                    if normalizeGSOforGFT:
                        Shat = graphTools.normalizeLaplacian(Shat)
                else:
                    Shat = What
                    if normalizeGSOforGFT:
                        Shat = graphTools.normalizeAdjacency(Shat)

            # Now that we have the perturbed graph, we need to create the
            # corresponding GST models
            perturbedModelsGST = {}

            if doDiffusion:
                perturbedModelsGST[diffusionName] = \
                             GST.DiffusionScattering(numScales, numLayers, What)

            if doMonicCubic:
                perturbedModelsGST[monicCubicName] = \
                                      GST.MonicCubic(numScales, numLayers, What)

            if doTightHann:
                perturbedModelsGST[tightHannName] = \
                                       GST.TightHann(numScales, numLayers, What)

            if doGFT:
                if G.L is not None:
                    _, GFThat = graphTools.computeGFT(Shat, order='increasing')
                else:
                    _, GFThat=graphTools.computeGFT(Shat,order='totalVariation')
                GFThat = GFThat.conj().T

            if computeBound and doMonicCubic:
                # This is the difference between the low-pass operators U in the
                # perturbed and the unperturbed models
                epsilonU = np.linalg.norm(
                                modelsGST[monicCubicName].U \
                                - perturbedModelsGST[monicCubicName].U)

            # Now that we have used each perturbation, we can create the data.

            #%%##########################################################
            #                                                           #
            #                    DATA CREATION                          #
            #                                                           #
            #############################################################

            x = np.sqrt(signalPower) * np.random.randn(nTest, 1, nNodes)
                # Each row is a random graph signal (only 1 feature)

            #%%##########################################################
            #                                                           #
            #                    SCATTERING TRANSFORM                   #
            #                                                           #
            #############################################################

            for thisModel in modelList:

                y = modelsGST[thisModel].computeTransform(x).squeeze(1)
                     # y = Phi(S,x)
                yHat = perturbedModelsGST[thisModel].computeTransform(x)\
                            .squeeze(1)
                    # yHat = Phi(Shat,x)

                # Compute the relative representation error:
                #   ||Phi(S,x) - Phi(Shat,x)|| / ||Phi(S,x)||
                thisReprError = np.linalg.norm(y - yHat, axis = 1)\
                                                /np.linalg.norm(y, axis = 1)

                # Save the representation error (averaged across nTest)
                reprError[thisModel][graph][itEpsilon][perturbation] \
                                                        = np.mean(thisReprError)

            #%%##########################################################
            #                                                           #
            #                    GFT                                    #
            #                                                           #
            #############################################################

            if doGFT:

                # Multiplication by the left because the information is located
                # in row vectors
                yGFT = (x @ GFT).squeeze(1)
                yGFThat = (x @ GFThat).squeeze(1)

                # Compute the relative representation error
                thisReprErrorGFT = np.linalg.norm(yGFT - yGFThat, axis = 1)\
                                             /np.linalg.norm(yGFT, axis = 1)

                # Save the representation error
                reprErrorGFT[graph][itEpsilon][perturbation] \
                                                     = np.mean(thisReprErrorGFT)

            #%%##########################################################
            #                                                           #
            #                    BOUND                                  #
            #                                                           #
            #############################################################

            if computeBound and doMonicCubic:
                # Compute the bound and save it. The expression is the one
                # in the paper, in Theorem 1.
                bound[graph][itEpsilon][perturbation] = np.sqrt(
                        (epsilonU ** 2) * xiBJL[0] + \
                        2 * epsilonU * BU * epsilon * C / B * xiBJL[1] + \
                        (BU ** 2) * (epsilon * C / B) ** 2 * xiBJL[2]
                        ) * np.sqrt(signalPower)


#%%##################################################################
#                                                                   #
#                       RESULTS (FIGURES)                           #
#                                                                   #
#####################################################################

# Now that we have computed the representation error of all runs, we can obtain
# a final result (mean and standard deviation) and plot it

meanReprErrorPerGraph = {} # Average across all perturbation realizations
meanReprError = {} # Average across all graph realizations
stdDevReprError = {} # Standard deviation across all graph realizations

######################
# COMPUTE STATISTICS #
######################

# Compute for each model
for thisModel in modelList:
    # Convert the lists into a matrix (3 dimensions):
    #  nGraphRealizations x len(perturbationEpsilon) x nPerturbationRealizations
    reprError[thisModel] = np.array(reprError[thisModel])

    # And compute again the mean across perturbation realizations
    meanReprErrorPerGraph[thisModel] = np.mean(reprError[thisModel],
                                               axis = 2)
    # And this resulting two-dimensional matrix contains the errors per graph
    # for each of the epsilon values.
    meanReprError[thisModel] = np.mean(meanReprErrorPerGraph[thisModel],
                                       axis = 0)
    stdDevReprError[thisModel] = np.std(meanReprErrorPerGraph[thisModel],
                                        axis = 0)

# Compute for GFT
if doGFT:
    # Convert the lists into a matrix (3 dimensions):
    #  nGraphRealizations x len(perturbationEpsilon) x nPerturbationRealizations
    reprErrorGFT = np.array(reprErrorGFT)

    # And compute again the mean across perturbation realizations
    meanReprErrorPerGraphGFT = np.mean(reprErrorGFT, axis = 2)
    # And this resulting two-dimensional matrix contains the errors per graph
    # for each of the epsilon values.
    meanReprErrorGFT = np.mean(meanReprErrorPerGraphGFT, axis = 0)
    stdDevReprErrorGFT = np.std(meanReprErrorPerGraphGFT, axis = 0)

if computeBound and doMonicCubic:
    # Convert the lists into a matrix (3 dimensions):
    #  nGraphRealizations x len(perturbationEpsilon) x nPerturbationRealizations
    bound = np.array(bound)

    # And compute again the mean across perturbation realizations
    meanBoundPerGraph = np.mean(bound, axis = 2)
    # And this resulting two-dimensional matrix contains the errors per graph
    # for each of the epsilon values.
    meanBound = np.mean(meanBoundPerGraph, axis = 0)
    stdDevBound = np.std(meanBoundPerGraph, axis = 0)

################
# SAVE RESULTS #
################

# If we're going to save the results (either figures or pickled variables) we
# need to create the directory where to save them

if doSaveVars or doFigs:
    saveDirResults = os.path.join(saveDir,'results')
    if not os.path.exists(saveDirResults):
        os.makedirs(saveDirResults)

##################
# SAVE VARIABLES #
##################

if doSaveVars:
    # Save all these results that we use to reconstruct the values
    #   Save these variables
    varsDict = {}
    varsDict['reprError'] = reprError
    varsDict['meanReprErrorPerGraph'] = meanReprErrorPerGraph
    varsDict['meanReprError'] = meanReprError
    varsDict['stdDevReprError'] = stdDevReprError
    if doGFT:
        varsDict['reprErrorGFT'] = reprError
        varsDict['meanReprErrorPerGraphGFT'] = meanReprErrorPerGraphGFT
        varsDict['meanReprErrorGFT'] = meanReprErrorGFT
        varsDict['stdDevReprErrorGFT'] = stdDevReprErrorGFT
    if computeBound and doMonicCubic:
        varsDict['bound'] = bound
        varsDict['meanBoundPerGraph'] = meanBoundPerGraph
        varsDict['meanBound'] = meanBound
        varsDict['stdDevBound'] = stdDevBound
    #   Determine filename to save them into
    varsFilename = 'representationError.pkl'
    pathToFile = os.path.join(saveDirResults, varsFilename)
    with open(pathToFile, 'wb') as varsFile:
        pickle.dump(varsDict, varsFile)

#########
# PLOTS #
#########

if doFigs:
    # Create figure handle
    reprErrorFig = plt.figure(figsize = (1.61*figSize, 1*figSize))
    # For each model, plot the results
    for thisModel in modelList:
        plt.errorbar(perturbationEpsilon, meanReprError[thisModel],
                     yerr = stdDevReprError[thisModel],
                     linewidth = lineWidth, marker = markerShape,
                     markersize = markerSize)
    # If there's representation error of the GFT, plot it
    if doGFT:
        plt.errorbar(perturbationEpsilon, meanReprErrorGFT,
                     yerr = stdDevReprErrorGFT,
                     linewidth = lineWidth, marker = markerShape,
                     markersize = markerSize)
    # If there's a bound, plot it
    if computeBound and doMonicCubic:
        plt.errorbar(perturbationEpsilon, meanBound,
                     yerr = stdDevBound,
                     linewidth = lineWidth, linestyle = '--',
                     marker = markerShape, markerSize = markerSize)
    plt.yscale('log')
    plt.ylabel(r'Representation Error: ' + \
               r'$\| \Phi(\mathbf{S},\mathbf{x}) ' + \
               r'- \Phi(\hat{\mathbf{S}},\mathbf{x})\| / ' + \
               r'\| \Phi(\mathbf{S},\mathbf{x}) \|$')
    plt.xlabel(r'$\varepsilon$')
    # Add the names to the legends
    if doGFT:
        modelList.append(GFTname)
    if computeBound and doMonicCubic:
        modelList.append(boundName)
    plt.legend(modelList)
    reprErrorFig.savefig(os.path.join(saveDirResults, 'reprErrorFig.pdf'),
                         bbox_inches = 'tight')