# 2019/10/27
# Fernando Gama, fgama@seas.upenn.edu
"""
architectures.py Architectures module

Definition of GNN architectures.

SelectionGNN: implements the selection GNN architecture
"""

import numpy as np
import scipy
import torch
import torch.nn as nn

import Utils.graphML as gml
import Utils.graphTools

zeroTolerance = 1e-9 # Absolute values below this number are considered zero.

class SelectionGNN(nn.Module):
    """
    SelectionGNN: implement the selection GNN architecture

    Initialization:

        SelectionGNN(dimNodeSignals, nFilterTaps, bias, # Graph Filtering
                     nonlinearity, # Nonlinearity
                     nSelectedNodes, poolingFunction, poolingSize, # Pooling
                     dimLayersMLP, # MLP in the end
                     GSO, # Structure
                     coarsening = False)

        Input:
            dimNodeSignals (list of int): dimension of the signals at each layer
            nFilterTaps (list of int): number of filter taps on each layer
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nFilterTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nFilterTaps) = L.
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            >> Obs.: If coarsening = True, this variable is ignored since the
                number of nodes in each layer is given by the graph coarsening
                algorithm.
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            >> Obs.: If coarsening = True, then the pooling function is one of
                the regular 1-d pooling functions available in torch.nn (instead
                of one of the summarizing functions in Utils.graphML).
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            >> Obs.: If coarsening = True, then the pooling size is ignored
                since, due to the binary tree nature of the graph coarsening
                algorithm, it always has to be 2.
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
            GSO (np.array): graph shift operator of choice.
            coarsening (bool, default = False): if True uses graph coarsening
                instead of zero-padding to reduce the number of nodes.
            >> Obs.: [i] Graph coarsening only works when the number
                 of edge features is 1 -scalar weights-. [ii] The graph
                 coarsening forces a given order of the nodes, and this order
                 has to be used to reordering the GSO as well as the samples
                 during training; as such, this order is internally saved and
                 applied to the incoming samples in the forward call -it is
                 thus advised to use the identity ordering in the model class
                 when using the coarsening method-.

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.

    Forward call:

        SelectionGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]

    Other methods:

        .changeGSO(S, nSelectedNodes = [], poolingSize = []): takes as input a
        new graph shift operator S as a tensor of shape
            numberNodes x numberNodes (x dimEdgeFeatures)
        Then, next time the SelectionGNN is run, it will run over the graph
        with GSO S, instead of running over the original GSO S. This is
        particularly useful when training on one graph, and testing on another
        one. The number of selected nodes and the pooling size will not change
        unless specifically consider those as input. Those lists need to have
        the same length as the number of layers. There is no need to define
        both, unless they change.
        >> Obs.: The number of nodes in the GSOs need not be the same, but
            unless we want to risk zero-padding beyond the original number
            of nodes (which just results in disconnected nodes), then we might
            want to update the nSelectedNodes and poolingSize accordingly, if
            the size of the new GSO is different.

        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimLayersMLP[-1], as well as the output
        of all the GNN layers (i.e. before the MLP layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1].
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO,
                 # Coarsening
                 coarsening = False):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nFilterTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nFilterTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nFilterTaps # Filter taps
        self.E = GSO.shape[0] # Number of edge features
        self.coarsening = coarsening # Whether to do coarsening or not
        # If we have to do coarsening, then note that it can only be done if
        # we have a single edge feature, otherwise, each edge feature could be
        # coarsed (and thus, ordered) in a different way, and there is no s
        # sensible way of merging back this different orderings. So, we will
        # only do coarsening if we have a single edge feature; otherwise, we
        # will default to selection sampling (therefore, always specify
        # nSelectedNodes)
        if self.coarsening and self.E == 1:
            GSO = scipy.sparse.csr_matrix(GSO[0])
            GSO, self.order = Utils.graphTools.coarsen(GSO, levels=self.L,
                                                       self_connections=False)
            # Now, GSO is a list of csr_matrix with self.L+1 coarsened GSOs,
            # we need to torch.tensor them and put them in a list.
            # order is just a list of indices to reorder the nodes.
            self.S = []
            self.N = [] # It has to be reset, because now the number of
                # nodes is determined by the coarsening scheme
            for S in GSO:
                S = S.todense().A.reshape([self.E, S.shape[0], S.shape[1]])
                    # So, S.todense() returns a numpy.matrix object; a numpy
                    # matrix cannot be converted into a tensor (i.e., added
                    # the third dimension), therefore we need to convert it to
                    # a numpy.array. According to the documentation, the
                    # attribute .A in a numpy.matrix returns self as an ndarray
                    # object. So that's why the .A is there.
                self.S.append(torch.tensor(S))
                self.N.append(S.shape[1])
            # Finally, because the graph coarsening algorithm is a binary tree
            # pooling, we always need to force a pooling size of 2
            self.alpha = [2] * self.L
        else:
            # If there's not coarsening, just save the GSO as a torch.tensor
            self.S = torch.tensor(GSO)
            self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
            self.alpha = poolingSize
            self.coarsening = False # If it failed because there are more than
                # one edge feature, then just set this to false, so we do not
                # need to keep checking whether self.E == 1 or not, just this
                # one
            self.order = None # No internal order, the order is given externally
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.dimLayersMLP = dimLayersMLP
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            gfl.append(gml.GraphFilter(self.F[l], self.F[l+1], self.K[l],
                                              self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            if self.coarsening:
                gfl[3*l].addGSO(self.S[l])
            else:
                gfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            gfl.append(self.sigma())
            #\\ Pooling
            if self.coarsening:
                gfl.append(self.rho(self.alpha[l]))
            else:
                gfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
                # Same as before, this is 3*l+2
                gfl[3*l+2].addGSO(self.S)
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def changeGSO(self, GSO, nSelectedNodes = [], poolingSize = []):

        # We use this to change the GSO, using the same graph filters.

        # Check that the new GSO has the correct
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N

        # Before making decisions, check if there is a new poolingSize list
        if len(poolingSize) > 0 and not self.coarsening:
            # (If it's coarsening, then the pooling size cannot change)
            # Check it has the right length
            assert len(poolingSize) == self.L
            # And update it
            self.alpha = poolingSize

        # Save the GSO in the right type and device
        if not self.coarsening:
            device = self.S.device
            # If there's not coarsening, just save the GSO as a torch.tensor
            if 'torch' in repr(GSO.dtype):
                self.S = GSO.to(device)
            else:
                self.S = torch.tensor(GSO).to(device)
        # Note that if it is coarsening, then we need to leave S as it is,
        # since we need to convert it into csr_matrix to compute the
        # corresponding clustering strategy

        # Now, check if we have a new list of nodes (this only makes sense
        # if there is no coarsening, because if it is coarsening, the list with
        # the number of nodes to be considered is ignored.)
        if len(nSelectedNodes) > 0 and not self.coarsening:
            # If we do, then we need to change the pooling functions to select
            # less nodes. This would allow to use graphs of different size.
            # Note that the pooling function, there is nothing learnable, so
            # they can easily be re-made, re-initialized.
            # The first thing we need to check, is that the length of the
            # number of nodes is equal to the number of layers (this list
            # indicates the number of nodes selected at the output of each
            # layer)
            assert len(nSelectedNodes) == self.L
            # Then, update the N that we have stored
            self.N = [GSO.shape[1]] + nSelectedNodes
            # And get the new pooling functions
            for l in range(self.L):
                # For each layer, add the pooling function
                self.GFL[3*l+2] = self.rho(self.N[l], self.N[l+1],
                                           self.alpha[l])
                self.GFL[3*l+2].addGSO(self.S)
        elif len(nSelectedNodes) == 0 and not self.coarsening:
            # Just update the GSO
            for l in range(self.L):
                self.GFL[3*l+2].addGSO(self.S)

        # If it's coarsening, then we need to compute the new coarsening
        # scheme
        if self.coarsening and self.E == 1:
            device = self.S[0].device
            GSO = scipy.sparse.csr_matrix(GSO[0])
            GSO, self.order = Utils.graphTools.coarsen(GSO, levels=self.L,
                                                       self_connections=False)
            # Now, GSO is a list of csr_matrix with self.L+1 coarsened GSOs,
            # we need to torch.tensor them and put them in a list.
            # order is just a list of indices to reorder the nodes.
            self.S = []
            self.N = [] # It has to be reset, because now the number of
                # nodes is determined by the coarsening scheme
            for S in GSO:
                S = S.todense().A.reshape([self.E, S.shape[0], S.shape[1]])
                    # So, S.todense() returns a numpy.matrix object; a numpy
                    # matrix cannot be converted into a tensor (i.e., added
                    # the third dimension), therefore we need to convert it to
                    # a numpy.array. According to the documentation, the
                    # attribute .A in a numpy.matrix returns self as an ndarray
                    # object. So that's why the .A is there.
                self.S.append(torch.tensor(S).to(device))
                self.N.append(S.shape[1])
            # And we need to update the GSO in all the places.
            #   Note that we do not need to change the pooling function, because
            #   it is the standard pooling function that doesn't care about the
            #   number of nodes: it still takes one every two of them.
            for l in range(self.L):
                self.GFL[3*l].addGSO(self.S[l]) # Graph convolutional layer
        else:
            # And update in the LSIGF that is still missing
            for l in range(self.L):
                self.GFL[3*l].addGSO(self.S) # Graph convolutional layer

    def splitForward(self, x):
        # Check if we need to reorder it (due to the internal ordering stemming
        # from the coarsening procedure)
        if self.coarsening:
            # If they have the same number of nodes (i.e. no dummy nodes where
            # added in the coarsening step) just re order them
            if x.shape[2] == self.N[0]:
                x = x[:, :, self.order]
            # If dummy nodes where added, then we need to add them to the data.
            # This is achieved by a function perm_data, but that operates on
            # np.arrays(), so we need to reconvert them back to np.arrays
            else:
                thisDevice = x.device # Save the device we where operating on
                x = x.cpu().numpy() # Convert to numpy
                x = Utils.graphTools.permCoarsening(x, self.order)
                    # Re order and add dummy values
                x = torch.tensor(x).to(thisDevice)

        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph filtering layer
        y = self.GFL(x)
        # Flatten the output
        yFlat = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(yFlat), y
        # If self.MLP is a sequential on an empty list it just does nothing.

    def forward(self, x):

        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)

        return output

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        if self.coarsening:
            for l in range(self.L):
                self.S[l] = self.S[l].to(device)
                self.GFL[3*l].addGSO(self.S[l])
        else:
            self.S = self.S.to(device)
            # And all the other variables derived from it.
            for l in range(self.L):
                self.GFL[3*l].addGSO(self.S)
                self.GFL[3*l+2].addGSO(self.S)