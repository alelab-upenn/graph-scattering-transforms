# 2019/10/27
# Fernando Gama, fgama@seas.upenn.edu
"""
dataTools.py Data management module

Several tools to manage data

FacebookEgo (class): loads the Facebook adjacency matrix of EgoNets
SourceLocalization (class): creates the datasets for a source localization
    problem
Authorship (class): loads and splits the dataset for the authorship attribution
    problem
"""

import os
import pickle
import hdf5storage # This is required to import old Matlab(R) files.
import urllib.request # To download from the internet
import gzip # To handle gz files
import shutil # Command line utilities

import numpy as np
import torch

import Utils.graphTools as graph

zeroTolerance = 1e-9 # Values below this number are considered zero.

def normalizeData(x, ax):

    # Normalize data along axis ax. The normalization is minus mean, divided
    # by std

    thisShape = x.shape # get the shape
    assert ax < len(thisShape) # check that the axis that we want to normalize
        # is there
    dataType = type(x) # get data type so that we don't have to convert

    if 'numpy' in repr(dataType):

        # Compute the statistics
        xMean = np.mean(x, axis = ax)
        xDev = np.std(x, axis = ax)
        # Add back the dimension we just took out
        xMean = np.expand_dims(xMean, ax)
        xDev = np.expand_dims(xDev, ax)

    elif 'torch' in repr(dataType):

        # Compute the statistics
        xMean = torch.mean(x, dim = ax)
        xDev = torch.std(x, dim = ax)
        # Add back the dimension we just took out
        xMean = xMean.unsqueeze(ax)
        xDev = xDev.unsqueeze(ax)

    # Subtract mean and divide by standard deviation
    x = (x - xMean) / xDev

    return x

def changeDataType(x, dataType):

    # So this is the thing: To change data type it depends on both, what dtype
    # the variable already is, and what dtype we want to make it.
    # Torch changes type by .type(), but numpy by .astype()
    # If we have already a torch defined, and we apply a torch.tensor() to it,
    # then there will be warnings because of gradient accounting.

    # All of these facts make changing types considerably cumbersome. So we
    # create a function that just changes type and handles all this issues
    # inside.

    # If we can't recognize the type, we just make everything numpy.

    # Check if the variable has an argument called 'dtype' so that we can now
    # what type of data type the variable is
    if 'dtype' in dir(x):
        varType = x.dtype

    # So, let's start assuming we want to convert to numpy
    if 'numpy' in repr(dataType):
        # Then, the variable con be torch, in which case we move it to cpu, to
        # numpy, and convert it to the right file.
        if 'torch' in repr(varType):
            x = x.cpu().numpy().astype(dataType)
        # Or it could be numpy, in which case we just use .astype
        elif 'numpy' in repr(type(x)):
            x = x.astype(dataType)
    # Now, we want to convert to torch
    elif 'torch' in repr(dataType):
        # If the variable is torch in itself
        if 'torch' in repr(varType):
            x = x.type(dataType)
        # But, if it's numpy
        elif 'numpy' in repr(type(x)):
            x = torch.tensor(x, dtype = dataType)

    # This only converts between numpy and torch. Any other thing is ignored
    return x

class _data:
    # Internal supraclass from which all data sets will inherit.
    # There are certain methods that all Data classes must have:
    #   getSamples(), to() and astype().
    # To avoid coding this methods over and over again, we create a class from
    # which the data can inherit this basic methods.

    # All the inputs are always assumed to be graph signals that are written
    #   nDataPoints (x nFeatures) x nNodes
    # If we have one feature, we have the expandDims() that adds a x1 so that
    # it can be readily processed by architectures/functions that always assume
    # a 3-dimensional input.

    # The output can be anything, so there's no point on doing an expandDims()
    # there. It could be a label, or it could be another graph signal, or a
    # one-hot vector. In any case, if there's a specific expand dims for the
    # output, it has to be dealt with within that specific class.
    def __init__(self):
        # Minimal set of attributes that all data classes should have
        self.dataType = None
        self.device = None
        self.nTrain = None
        self.nValid = None
        self.nTest = None
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = None
        self.samples['train']['targets'] = None
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = None
        self.samples['valid']['targets'] = None
        self.samples['test'] = {}
        self.samples['test']['signals'] = None
        self.samples['test']['targets'] = None

    def getSamples(self, samplesType, *args):
        # type: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.samples[samplesType]['signals']
        y = self.samples[samplesType]['targets']
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # Select the corresponding samples
                xSelected = x[selectedIndices]
                y = y[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                xSelected = x[args[0]]
                # And assign the labels
                y = y[args[0]]

            # If we only selected a single element, then the nDataPoints dim
            # has been left out. So if we have less dimensions, we have to
            # put it back
            if len(xSelected.shape) < len(x.shape):
                if 'torch' in self.dataType:
                    x = xSelected.unsqueeze(0)
                else:
                    x = np.expand_dims(xSelected, axis = 0)
            else:
                x = xSelected

        return x, y

    def expandDims(self):

        # For each data set partition
        for key in self.samples.keys():
            # If there's something in them
            if self.samples[key]['signals'] is not None:
                # And if it has only two dimensions
                #   (shape: nDataPoints x nNodes)
                if len(self.samples[key]['signals'].shape) == 2:
                    # Then add a third dimension in between so that it ends
                    # up with shape
                    #   nDataPoints x 1 x nNodes
                    # and it respects the 3-dimensional format that is taken
                    # by many of the processing functions
                    if 'torch' in repr(self.dataType):
                        self.samples[key]['signals'] = \
                                       self.samples[key]['signals'].unsqueeze(1)
                    else:
                        self.samples[key]['signals'] = np.expand_dims(
                                                   self.samples[key]['signals'],
                                                   axis = 1)

    def astype(self, dataType):
        # This changes the type for the minimal attributes (samples). This
        # methods should still be initialized within the data classes, if more
        # attributes are used.

        # The labels could be integers as created from the dataset, so if they
        # are, we need to be sure they are integers also after conversion.
        # To do this we need to match the desired dataType to its int
        # counterpart. Typical examples are:
        #   numpy.float64 -> numpy.int64
        #   numpy.float32 -> numpy.int32
        #   torch.float64 -> torch.int64
        #   torch.float32 -> torch.int32

        targetType = str(self.samples['train']['targets'].dtype)
        if 'int' in targetType:
            if 'numpy' in repr(dataType):
                if '64' in targetType:
                    targetType = np.int64
                elif '32' in targetType:
                    targetType = np.int32
            elif 'torch' in repr(dataType):
                if '64' in targetType:
                    targetType = torch.int64
                elif '32' in targetType:
                    targetType = torch.int32
        else: # If there is no int, just stick with the given dataType
            targetType = dataType

        # Now that we have selected the dataType, and the corresponding
        # labelType, we can proceed to convert the data into the corresponding
        # type
        for key in self.samples.keys():
            self.samples[key]['signals'] = changeDataType(
                                                   self.samples[key]['signals'],
                                                   dataType)
            self.samples[key]['targets'] = changeDataType(
                                                   self.samples[key]['targets'],
                                                   targetType)

        # Update attribute
        if dataType is not self.dataType:
            self.dataType = dataType

    def to(self, device):
        # This changes the type for the minimal attributes (samples). This
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        # This can only be done if they are torch tensors
        if 'torch' in repr(self.dataType):
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                      = self.samples[key][secondKey].to(device)

            # If the device changed, save it.
            if device is not self.device:
                self.device = device

class _dataForClassification(_data):
    # Internal supraclass from which data classes inherit when they are used
    # for classification. This renders the .evaluate() method the same in all
    # cases (how many examples are correctly labels) so justifies the use of
    # another internal class.

    def __init__(self):

        super().__init__()


    def evaluate(self, yHat, y, tol = 1e-9):
        """
        Return the accuracy (ratio of yHat = y)
        """
        N = len(y)
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
            yHat = torch.argmax(yHat, dim = 1)
            #   And compute the error
            totalErrors = torch.sum(torch.abs(yHat - y) > tol)
            accuracy = 1 - totalErrors.type(self.dataType)/N
        else:
            yHat = np.array(yHat)
            y = np.array(y)
            #   We compute the target label (hardmax)
            yHat = np.argmax(yHat, axis = 1)
            #   And compute the error
            totalErrors = np.sum(np.abs(yHat - y) > tol)
            accuracy = 1 - totalErrors.astype(self.dataType)/N
        #   And from that, compute the accuracy
        return accuracy

class FacebookEgo:
    """
    FacebookEgo: Loads the adjacency matrix of the Facebook Egonets available
        in https://snap.stanford.edu/data/ego-Facebook.html by
        J. McAuley and J. Leskovec. Learning to Discover Social Circles in Ego
        Networks. NIPS, 2012.
    Initialization:
    Input:
        dataDir (string): path for the directory in where to look for the data
            (if the data is not found, it will be downloaded to this directory)
        use234 (bool): if True, load a smaller subnetwork of 234 users with two
            communities (one big, and one small)

    Methods:

    .loadData(filename, use234): load the data in self.dataDir/filename, if it
        does not exist, then download it and save it as filename in self.dataDir
        If use234 is True, load the 234-user subnetwork as well.

    adjacencyMatrix = .getAdjacencyMatrix([use234]): return the nNodes x nNodes
        np.array with the adjacency matrix. If use234 is True, then return the
        smaller nNodes = 234 user subnetwork (default: use234 = False).
    """

    def __init__(self, dataDir, use234 = False):

        # Dataset directory
        self.dataDir = dataDir
        # Empty attributes
        self.adjacencyMatrix = None
        self.adjacencyMatrix234 = None

        # Load data
        self.loadData('facebookEgo.pkl', use234)

    def loadData(self, filename, use234):
        # Check if the dataDir exists, and if not, create it
        if not os.path.exists(self.dataDir):
            os.makedirs(self.dataDir)
        # Create the filename to save/load
        datasetFilename = os.path.join(self.dataDir, filename)
        if use234:
            datasetFilename234 = os.path.join(self.dataDir,'facebookEgo234.pkl')
            if os.path.isfile(datasetFilename234):
                with open(datasetFilename234, 'rb') as datasetFile234:
                    datasetDict = pickle.load(datasetFile234)
                    self.adjacencyMatrix234 = datasetDict['adjacencyMatrix']
        # Check if the file does exist, load it
        if os.path.isfile(datasetFilename):
            # If it exists, load it
            with open(datasetFilename, 'rb') as datasetFile:
                datasetDict = pickle.load(datasetFile)
                # And save the corresponding variable
                self.adjacencyMatrix = datasetDict['adjacencyMatrix']
        else: # If it doesn't exist, load it
            # There could be three options here: that we have the raw data
            # already there, that we have the zip file and need to unzip it,
            # or that we do not have nothing and we need to download it.
            existsRawData = \
                   os.path.isfile(os.path.join(self.dataDir,
                                               'facebook_combined.txt'))
           # And the zip file
            existsZipFile = os.path.isfile(os.path.join(
                                       self.dataDir,'facebook_combined.txt.gz'))
            if not existsRawData and not existsZipFile: # We have to download it
                fbURL='https://snap.stanford.edu/data/facebook_combined.txt.gz'
                urllib.request.urlretrieve(fbURL,
                                 filename = os.path.join(
                                       self.dataDir,'facebook_combined.txt.gz'))
                existsZipFile = True
            if not existsRawData and existsZipFile: # Unzip it
                zipFile = os.path.join(self.dataDir, 'facebook_combined.txt.gz')
                txtFile = os.path.join(self.dataDir, 'facebook_combined.txt')
                with gzip.open(zipFile, 'rb') as f_in:
                    with open(txtFile, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            # Now that we have the data, we can get their filenames
            rawDataFilename = os.path.join(self.dataDir,'facebook_combined.txt')
            assert os.path.isfile(rawDataFilename)
            # And we can load it and store it.
            adjacencyMatrix = np.empty([0, 0]) # Start with an empty matrix and
            # then we slowly add the number of nodes, which we do not assume
            # to be known beforehand.
            # Let's start with the data.
            # Open it.
            with open(rawDataFilename, 'r') as rawData:
                # The file consists of a succession of lines, each line
                # corresponds to an edge
                for dataLine in rawData:
                    # For each line, we split it in the different fields
                    dataLineSplit = dataLine.rstrip('\n').split(' ')
                    # Keep the ones we care about here
                    node_i = int(dataLineSplit[0])
                    node_j = int(dataLineSplit[1])
                    node_max = max(node_i, node_j) # Get the largest node
                    # Now we have to add this information to the adjacency
                    # matrix.
                    #   We need to check whether we need to add more elements
                    if node_max+1 > max(adjacencyMatrix.shape):
                        colDiff = node_max+1 - adjacencyMatrix.shape[1]
                        zeroPadCols = np.zeros([adjacencyMatrix.shape[0],\
                                                colDiff])
                        adjacencyMatrix = np.concatenate((adjacencyMatrix,
                                                          zeroPadCols),
                                                         axis = 1)
                        rowDiff = node_max+1 - adjacencyMatrix.shape[0]
                        zeroPadRows = np.zeros([rowDiff,\
                                                adjacencyMatrix.shape[1]])
                        adjacencyMatrix = np.concatenate((adjacencyMatrix,
                                                          zeroPadRows),
                                                         axis = 0)
                    # Now that we have assured appropriate dimensions
                    adjacencyMatrix[node_i, node_j] = 1.
                    # And because it is undirected by construction
                    adjacencyMatrix[node_j, node_i] = 1.
            # Now that it is loaded, let's store it
            self.adjacencyMatrix = adjacencyMatrix
            # And save it in a pickle file for posterity
            with open(datasetFilename, 'wb') as datasetFile:
                pickle.dump(
                        {'adjacencyMatrix': self.adjacencyMatrix},
                        datasetFile
                        )

    def getAdjacencyMatrix(self, use234 = False):

        return self.adjacencyMatrix234 if use234 else self.adjacencyMatrix

class SourceLocalization(_dataForClassification):
    """
    SourceLocalization: Creates the dataset for a source localization problem

    Initialization:

    Input:
        G (class): Graph on which to diffuse the process, needs an attribute
            .N with the number of nodes (int) and attribute .W with the
            adjacency matrix (np.array)
        nTrain (int): number of training samples
        nValid (int): number of validation samples
        nTest (int): number of testing samples
        sourceNodes (list of int): list of indices of nodes to be used as
            sources of the diffusion process
        tMax (int): maximum diffusion time, if None, the maximum diffusion time
            is the size of the graph (default: None)
        dataType (dtype): datatype for the samples created (default: np.float64)
        device (device): if torch.Tensor datatype is selected, this is on what
            device the data is saved.

    Methods:

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .expandDims(): Adds the feature dimension to the graph signals (i.e. for
        graph signals of shape nSamples x nNodes, turns them into shape
        nSamples x 1 x nNodes, so that they can be handled by general graph
        signal processing techniques that take into account a feature dimension
        by default)

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): unnormalized probability of each label (shape:
                nDataPoints x nClasses)
            y (dtype.array): correct labels (1-D binary vector, shape:
                nDataPoints)
            tol (float, default = 1e-9): numerical tolerance to consider two
                numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """

    def __init__(self, G, nTrain, nValid, nTest, sourceNodes, tMax = None,
                 dataType = np.float64, device = 'cpu'):
        # Initialize parent
        super().__init__()
        # store attributes
        self.dataType = dataType
        self.device = device
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        # If no tMax is specified, set it the maximum possible.
        if tMax == None:
            tMax = G.N
        #\\\ Generate the samples
        # Get the largest eigenvalue of the weighted adjacency matrix
        EW, VW = graph.computeGFT(G.W, order = 'totalVariation')
        eMax = np.max(EW)
        # Normalize the matrix so that it doesn't explode
        Wnorm = G.W / eMax
        # total number of samples
        nTotal = nTrain + nValid + nTest
        # sample source nodes
        sampledSources = np.random.choice(sourceNodes, size = nTotal)
        # sample diffusion times
        sampledTimes = np.random.choice(tMax, size = nTotal)
        # Since the signals are generated as W^t * delta, this reduces to the
        # selection of a column of W^t (the column corresponding to the source
        # node). Therefore, we generate an array of size tMax x N x N with all
        # the powers of the matrix, and then we just simply select the
        # corresponding column for the corresponding time
        lastWt = np.eye(G.N, G.N)
        Wt = lastWt.reshape([1, G.N, G.N])
        for t in range(1,tMax):
            lastWt = lastWt @ Wnorm
            Wt = np.concatenate((Wt, lastWt.reshape([1, G.N, G.N])), axis = 0)
        x = Wt[sampledTimes, :, sampledSources]
        # Now, we have the signals and the labels
        signals = x # nTotal x N (CS notation)
        # Finally, we have to match the source nodes to the corresponding labels
        # which start at 0 and increase in integers.
        nodesToLabels = {}
        for it in range(len(sourceNodes)):
            nodesToLabels[sourceNodes[it]] = it
        labels = [nodesToLabels[x] for x in sampledSources] # nTotal
        # Split and save them
        self.samples['train']['signals'] = signals[0:nTrain, :]
        self.samples['train']['targets'] = np.array(labels[0:nTrain])
        self.samples['valid']['signals'] = signals[nTrain:nTrain+nValid, :]
        self.samples['valid']['targets'] =np.array(labels[nTrain:nTrain+nValid])
        self.samples['test']['signals'] = signals[nTrain+nValid:nTotal, :]
        self.samples['test']['targets'] =np.array(labels[nTrain+nValid:nTotal])
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)

class Authorship(_dataForClassification):
    """
    Authorship: Loads the dataset of 19th century writers for the authorship
        attribution problem

    Initialization:

    Input:
        authorName (string): which is the selected author to attribute plays to
        ratioTrain (float): ratio of the total texts to be part of the training
            set
        ratioValid (float): ratio of the train texts to be part of the
            validation set
        dataPath (string): path to where the authorship data is located
        graphNormalizationType ('rows' or 'cols'): how to normalize the created
            graph from combining all the selected author WANs
        keepIsolatedNodes (bool): If False, get rid of isolated nodes
        forceUndirected (bool): If True, create an undirected graph
        forceConnected (bool): If True, ensure that the resulting graph is
            connected
        dataType (dtype): type of loaded data (default: np.float64)
        device (device): where to store the data (e.g., 'cpu', 'cuda:0', etc.)

    Methods:

    .loadData(dataPath): load the data found in dataPath and store it in
        attributes .authorData and .functionWords

    authorData = .getAuthorData(samplesType, selectData, [, optionalArguments])

        Input:
            samplesType (string): 'train', 'valid', 'test' or 'all' to determine
                from which dataset to get the raw author data from
            selectData (string): 'WAN' or 'wordFreq' to decide if we want to
                retrieve either the WAN of each excerpt or the word frequency
                count of each excerpt
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get

        Output:
            Either the WANs or the word frequency count of all the excerpts of
            the selected author

    .createGraph(): creates a graph from the WANs of the excerpt written by the
        selected author available in the training set. The fusion of this WANs
        is done in accordance with the input options following
        graphTools.createGraph().
        The resulting adjacency matrix is stored.

    .getGraph(): fetches the stored adjacency matrix and returns it

    .getFunctionWords(): fetches the list of functional words. Returns a tuple
        where the first element correspond to all the functional words in use,
        and the second element consists of all the functional words available.
        Obs.: When we created the graph, some of the functional words might have
        been dropped in order to make it connected, for example.

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .expandDims(): Adds the feature dimension to the graph signals (i.e. for
        graph signals of shape nSamples x nNodes, turns them into shape
        nSamples x 1 x nNodes, so that they can be handled by general graph
        signal processing techniques that take into account a feature dimension
        by default)

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """

    def __init__(self, authorName, ratioTrain, ratioValid, dataPath,
                 graphNormalizationType, keepIsolatedNodes,
                 forceUndirected, forceConnected,
                 dataType = np.float64, device = 'cpu'):
        # Initialize parent
        super().__init__()
        # Store
        self.authorName = authorName
        self.ratioTrain = ratioTrain
        self.ratioValid = ratioValid
        self.dataPath = dataPath
        self.dataType = dataType
        self.device = device
        # Store characteristics of the graph to be created
        self.graphNormalizationType = graphNormalizationType
        self.keepIsolatedNodes = keepIsolatedNodes
        self.forceUndirected = forceUndirected
        self.forceConnected = forceConnected
        self.adjacencyMatrix = None
        # Other data to save
        self.authorData = None
        self.selectedAuthor = None
        self.allFunctionWords = None
        self.functionWords = None
        # Load data
        self.loadData(dataPath)
        # Check that the authorName is a valid name
        assert authorName in self.authorData.keys()
        # Get the selected author's data
        thisAuthorData = self.authorData[authorName].copy()
        nExcerpts = thisAuthorData['wordFreq'].shape[0] # Number of excerpts
            # by the selected author
        nTrainAuthor = int(round(ratioTrain * nExcerpts))
        nValidAuthor = int(round(ratioValid * nTrainAuthor))
        nTestAuthor = nExcerpts - nTrainAuthor
        nTrainAuthor = nTrainAuthor - nValidAuthor
        # Now, we know how many training, validation and testing samples from
        # the required author. But we will also include an equal amount of
        # other authors, therefore
        self.nTrain = round(2 * nTrainAuthor)
        self.nValid = round(2 * nValidAuthor)
        self.nTest = round(2 * nTestAuthor)

        # Now, let's get the corresponding signals for the author
        xAuthor = thisAuthorData['wordFreq']
        # Get a random permutation of these works, and split them accordingly
        randPerm = np.random.permutation(nExcerpts)
        # Save the indices corresponding to each split
        randPermTrain = randPerm[0:nTrainAuthor]
        randPermValid = randPerm[nTrainAuthor:nTrainAuthor+nValidAuthor]
        randPermTest = randPerm[nTrainAuthor+nValidAuthor:nExcerpts]
        xAuthorTrain = xAuthor[randPermTrain, :]
        xAuthorValid = xAuthor[randPermValid, :]
        xAuthorTest = xAuthor[randPermTest, :]
        # And we will store this split
        self.selectedAuthor = {}
        # Copy all data
        self.selectedAuthor['all'] = thisAuthorData.copy()
        # Copy word frequencies
        self.selectedAuthor['train'] = {}
        self.selectedAuthor['train']['wordFreq'] = xAuthorTrain.copy()
        self.selectedAuthor['valid'] = {}
        self.selectedAuthor['valid']['wordFreq'] = xAuthorValid.copy()
        self.selectedAuthor['test'] = {}
        self.selectedAuthor['test']['wordFreq'] = xAuthorTest.copy()
        # Copy WANs
        self.selectedAuthor['train']['WAN'] = \
                              thisAuthorData['WAN'][randPermTrain, :, :].copy()
        self.selectedAuthor['valid']['WAN'] = \
                              thisAuthorData['WAN'][randPermValid, :, :].copy()
        self.selectedAuthor['test']['WAN'] = \
                               thisAuthorData['WAN'][randPermTest, :, :].copy()
        # Now we need to get an equal amount of works from the rest of the
        # authors.
        xRest = np.empty([0, xAuthorTrain.shape[1]]) # Create an empty matrix
        # to store all the works by the rest of the authors.
        # Now go author by author gathering all works
        for key in self.authorData.keys():
            # Only for authors that are not the selected author
            if key is not authorName:
                thisAuthorTexts = self.authorData[key]['wordFreq']
                xRest = np.concatenate((xRest, thisAuthorTexts), axis = 0)
        # After obtaining all works, xRest is of shape nRestOfData x nWords
        # We now need to select at random from this other data, but only up
        # to nExcerpts. Therefore, we will randperm all the indices, but keep
        # only the first nExcerpts indices.
        randPerm = np.random.permutation(xRest.shape[0])
        randPerm = randPerm[0:nExcerpts] # nExcerpts x nWords
        # And now we should just get the appropriate number of texts from these
        # other authors.
        # Compute how many samples for each case
        nTrainRest = self.nTrain - nTrainAuthor
        nValidRest = self.nValid - nValidAuthor
        nTestRest = self.nTest - nTestAuthor
        # And obtain those
        xRestTrain = xRest[randPerm[0:nTrainRest], :]
        xRestValid = xRest[randPerm[nTrainRest:nTrainRest + nValidRest], :]
        xRestTest = xRest[randPerm[nTrainRest+nValidRest:nExcerpts], :]
        # Now construct the signals and labels. Signals is just the
        # concatenation of each of these excerpts. Labels is just a bunch of
        # 1s followed by a bunch of 0s
        # Obs.: The fact that the dataset is ordered now, it doesn't matter,
        # since it will be shuffled at each epoch.
        xTrain = np.concatenate((xAuthorTrain, xRestTrain), axis = 0)
        labelsTrain = np.concatenate((np.ones(nTrainAuthor),
                                      np.zeros(nTrainRest)), axis = 0)
        xValid = np.concatenate((xAuthorValid, xRestValid), axis = 0)
        labelsValid = np.concatenate((np.ones(nValidAuthor),
                                      np.zeros(nValidRest)), axis = 0)
        xTest = np.concatenate((xAuthorTest, xRestTest), axis = 0)
        labelsTest = np.concatenate((np.ones(nTestAuthor),
                                     np.zeros(nTestRest)), axis = 0)
        # And assign them to the required attribute samples
        self.samples['train']['signals'] = xTrain
        self.samples['train']['targets'] = labelsTrain.astype(np.int)
        self.samples['valid']['signals'] = xValid
        self.samples['valid']['targets'] = labelsValid.astype(np.int)
        self.samples['test']['signals'] = xTest
        self.samples['test']['targets'] = labelsTest.astype(np.int)
        # Create graph
        self.createGraph()
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)

    def loadData(self, dataPath):
        # TODO: Analyze if it's worth it to create a .pkl and load that
        # directly once the data has been appropriately parsed. It's just
        # that loading with hdf5storage takes a couple of second that
        # could be saved if the .pkl file is faster.
        rawData = hdf5storage.loadmat(dataPath)
        # rawData is a dictionary with four keys:
        #   'all_authors': contains the author list
        #   'all_freqs': contains the word frequency count for each excerpt
        #   'all_wans': contains the WANS for each excerpt
        #   'function_words': a list of the functional words
        # The issue is that hdf5storage, while necessary to load old
        # Matlab(R) files, gives the data in a weird format, that we need
        # to adapt and convert.
        # The data will be structured as follows. We will have an
        # authorData dictionary of dictionaries: the first key will be the
        # author name, the second key will be either freqs or wans to
        # access either one or another.
        # We will also clean up and save the functional word list, although
        # we do not need to use it.
        authorData = {} # Create dictionary
        for it in range(len(rawData['all_authors'])):
            thisAuthor = str(rawData['all_authors'][it][0][0][0])
            # Each element in rawData['all_authors'] is nested in a couple
            # of lists, so that's why we need the three indices [0][0][0]
            # to reach the string with the actual author name.
            # Get the word frequency
            thisWordFreq = rawData['all_freqs'][0][it] # 1 x nWords x nData
            # Again, the [0] is due to the structure of the data
            # Let us get rid of that extra 1, and then transpose this to be
            # stored as nData x nWords (since nWords is the dimension of
            # the number of nodes the network will have; CS notation)
            thisWordFreq = thisWordFreq.squeeze(0).T # nData x nWords
            # Finally, get the WANs
            thisWAN = rawData['all_wans'][0][it] # nWords x nWords x nData
            thisWAN = thisWAN.transpose(2, 0, 1) # nData x nWords x nWords
            # Obs.: thisWAN is likely not symmetric, so the way this is
            # transposed matters. In this case, since thisWAN was intended
            # to be a tensor in matlab (where the last index is the
            # collection of matrices), we just throw that last dimension to
            # the front (since numpy consider the first index as the
            # collection index).
            # Now we can create the dictionary and save the corresopnding
            # data.
            authorData[thisAuthor] = {}
            authorData[thisAuthor]['wordFreq'] = thisWordFreq
            authorData[thisAuthor]['WAN'] = thisWAN
        # And at last, gather the list of functional words
        functionWords = [] # empty list to store the functional words
        for word in rawData['function_words']:
            functionWords.append(str(word[0][0][0]))
        # Store all the data recently collected
        self.authorData = authorData
        self.allFunctionWords = functionWords
        self.functionWords = functionWords.copy()

    def getAuthorData(self, samplesType, dataType, *args):
        # type: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test' or samplesType == 'all'
        # Check that the dataType is either wordFreq or WAN
        assert dataType == 'WAN' or dataType == 'wordFreq'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.selectedAuthor[samplesType][dataType]
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # The reshape is to avoid squeezing if only one sample is
                # requested (because x can have two or three dimension, we
                # need to take a longer path here, so we will only do it
                # if args[0] is equal to 1.)
                if args[0] == 1:
                    newShape = [1]
                    newShape.extend(list(x.shape[1:]))
                    x = x[selectedIndices].reshape(newShape)
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                xNew = x[args[0]]
                # If only one element is selected, avoid squeezing. Given that
                # the element can be a list (which has property len) or an
                # np.array (which doesn't have len, but shape), then we can
                # only avoid squeezing if we check that it has been sequeezed
                # (or not)
                if len(xNew.shape) <= len(x.shape):
                    newShape = [1]
                    newShape.extend(list(x.shape[1:]))
                    x = xNew.reshape(newShape)

        return x

    def createGraph(self):

        # Save list of nodes to keep to later update the datasets with the
        # appropriate words
        nodesToKeep = []
        # Number of nodes (so far) = Number of functional words
        N = self.selectedAuthor['all']['wordFreq'].shape[1]
        # Create graph
        graphOptions = {}
        graphOptions['adjacencyMatrices'] = self.selectedAuthor['train']['WAN']
        graphOptions['nodeList'] = nodesToKeep
        graphOptions['aggregationType'] = 'sum'
        graphOptions['normalizationType'] = self.graphNormalizationType
        graphOptions['isolatedNodes'] = self.keepIsolatedNodes
        graphOptions['forceUndirected'] = self.forceUndirected
        graphOptions['forceConnected'] = self.forceConnected
        W = graph.createGraph('fuseEdges', N, graphOptions)
        # Obs.: We do not need to recall graphOptions['nodeList'] as nodesToKeep
        # since these are all passed as pointers that point to the same list, so
        # modifying graphOptions also modifies nodesToKeep.
        # Store adjacency matrix
        self.adjacencyMatrix = W.astype(np.float64)
        # Update data
        #   For each dataset split
        for key in self.samples.keys():
            #   Check the signals have been loaded
            if self.samples[key]['signals'] is not None:
                #   And check which is the dimension of the nodes (i.e. whether
                #   it was expanded or not, since we always need to keep the
                #   entries of the last dimension)
                if len(self.samples[key]['signals'].shape) == 2:
                    self.samples[key]['signals'] = \
                                   self.samples[key]['signals'][: , nodesToKeep]
                elif len(self.samples[key]['signals'].shape) == 2:
                    self.samples[key]['signals'] = \
                                   self.samples[key]['signals'][:,:,nodesToKeep]

        if self.allFunctionWords is not None:
            self.functionWords = [self.allFunctionWords[w] for w in nodesToKeep]

    def getGraph(self):

        return self.adjacencyMatrix

    def getFunctionWords(self):

        return self.functionWords, self.allFunctionWords

    def astype(self, dataType):
        # This changes the type for the selected author as well as the samples
        for key in self.selectedAuthor.keys():
            for secondKey in self.selectedAuthor[key].keys():
                self.selectedAuthor[key][secondKey] = changeDataType(
                                            self.selectedAuthor[key][secondKey],
                                            dataType)
        self.adjacencyMatrix = changeDataType(self.adjacencyMatrix, dataType)

        # And now, initialize to change the samples as well (and also save the
        # data type)
        super().astype(dataType)


    def to(self, device):
        # If the dataType is 'torch'
        if 'torch' in repr(self.dataType):
            # Change the selected author ('test', 'train', 'valid', 'all';
            # 'WANs', 'wordFreq')
            for key in self.selectedAuthor.keys():
                for secondKey in self.selectedAuthor[key].keys():
                    self.selectedAuthor[key][secondKey] \
                                = self.selectedAuthor[key][secondKey].to(device)
            self.adjacencyMatrix.to(device)
            # And call the inherit method to initialize samples (and save to
            # device)
            super().to(device)