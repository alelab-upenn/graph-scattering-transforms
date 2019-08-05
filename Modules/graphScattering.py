#2019/05/20~2019/05/23
# Fernando Gama, fgama@seas.upenn.edu
"""
graphScattering.py  Graph scattering transform models

Functions:
    
monicCubicFunction: computes a function with a monic polynomial with cubic 
    interpolation
HannKernel: computes the values of the Hann kernel function

Wavelets (filter banks):

monicCubicWavelet: computes the filter bank for the wavelets constructed
    of two monic polynomials interpolated with a cubic one
tightHannWavelet: computes the filter bank for the tight wavelets constructed
    using a Hann kernel function and warping
diffusionWavelets: computes the filter bank for the diffusion wavelets
    
Models:

graphScatteringTransform: base class for the computation of the graph scattering
    transform coefficients
DiffusionScattering: diffusion scattering transform
MonicCubic: graph scattering transform using monic cubic polynomial wavelets
TightHann: graph scattering transform using a tight frame with Hann wavelets
"""

import numpy as np
import Utils.graphTools as graphTools

zeroTolerance = 1e-9 # Values below this number are considered zero.
infiniteNumber = 1e12 # infinity equals this number

def monicCubicFunction(x, alpha, beta, x1, x2, computeDerivative):
    """
    monicCubicFunction: computes the output of a function with monic polynomials
        with cubic interpolation
        
    See eq. (65) of D. K. Hammond, P. Vandergheynst, and R. Gribonval, "Wavelets
        on graphs via spectral graph theory," Appl. Comput. Harmonic Anal.,
        vol. 30, no. 2, pp. 129–150, March 2011.
        
    Input:
        x (np.array): contains the values of the input to the function (shape N)
        alpha (float): exponent of the lower interval monic polynomial
        beta (float): exponent of the upper interval monic polynomial
        x1 (float): higher end of the lower interval
        x2 (float): lower end of the higher interval
        computeDerivative (bool): compute the derivative if True
        
    Output:
        y (np.array): contains the output of the function (same shape as x)
        dy (np.array): if computeDerivative is True, constains the value of
            the derivative; otherwise, contains None
        
    """
    # Empty function
    y = np.zeros(x.shape[0])
    # Coefficients for the interpolating cubic polynomial
    A = np.array(
            [[1., x1, x1 ** 2,    x1 ** 3],
             [1., x2, x2 ** 2,    x2 ** 3],
             [0., 1.,  2 * x1, 3* x1 ** 2],
             [0., 1.,  2 * x2, 3* x2 ** 2]]
            )
    b = np.array([1., 1., alpha/x1, -beta/x2])
    c = np.linalg.solve(A, b)
    y[x < x1] = (x1 ** (-alpha)) * (x[x < x1] ** alpha)
    indicesMiddle = np.intersect1d(np.nonzero(x >= x1), np.nonzero(x <= x2))
    y[indicesMiddle] = c[0] \
                        + c[1] *  x[indicesMiddle] \
                        + c[2] * (x[indicesMiddle] ** 2) \
                        + c[3] * (x[indicesMiddle] ** 3)
    y[x > x2] = (x2 ** beta) * (x[x > x2] ** (-beta))
    
    # Compute the derivative
    if computeDerivative:
        dy = np.zeros(x.shape[0])
        dy[x < x1] = alpha * (x1 ** (-alpha)) * (x[x < x1] ** (alpha-1))
        dy[indicesMiddle] = c[1] \
                                + 2 * c[2] *  x[indicesMiddle] \
                                + 3 * c[3] * (x[indicesMiddle] ** 2)
        dy[x > x2] = -beta * (x2 ** beta) * (x[x > x2] ** (-beta-1))
    else:
        dy = None
    
    return y, dy
    
def monicCubicWavelet(V, E, J, alpha, beta, x1, x2, K, eMax, computeBound):
    """
    monicCubicWavelet: computes the filter bank for the wavelets constructed
        of two monic polynomials interpolated with a cubic one
        
    See eq. (24) of D. K. Hammond, P. Vandergheynst, and R. Gribonval, "Wavelets
        on graphs via spectral graph theory," Appl. Comput. Harmonic Anal.,
        vol. 30, no. 2, pp. 129–150, March 2011.
    Also, see discussion below eq. (65) for wavelet construction.
        
    Input:
        V (np.array): eigenvector matrix (shape N x N)
        E (np.array): eigenvalue matrix (diagonal matrix of shape N x N)
        J (int): number of scales
        alpha (float): exponent of the lower interval monic polynomial
        beta (float): exponent of the upper interval monic polynomial
        x1 (float): higher end of the lower interval
        x2 (float): lower end of the higher interval
        K (int): factor to determine minimum eigenvalue resolution as eMax/K
        eMax (float): maximum eigenvalue resolution
        computeBound (bool): if True, compute the integral Lipschitz constant 
            and the filter bank frame upper bound
        
    Output:
        H (np.array): of shape J x N x N contains the J matrices corresponding
            to all the filter scales
        B (float): frame upper bound (if computeBound is True; None otherwise)
        C (float): integral Lipschitz constant (if computeBound is True;
          None otherwise)
    """
    e = np.diag(E) # eigenvalues
    VH = V.conj().T # V hermitian
    N = V.shape[0] # Number of nodes
    eMin = eMax / K # lambda_min
    maxScale = x2 / eMin # t_1
    minScale = x2 / eMax # t_J
    t = np.logspace(np.log10(minScale), np.log10(maxScale), J-1) # scales
    # Create wavelets
    H = np.empty([0, N, N])
    # Compute integral Lipschitz constant
    if computeBound:
        C = np.empty(0)
        B = np.empty(0)
    else:
        C = None
        B = None
    for j in range(0, J-1):
        psi, dpsi =  monicCubicFunction(t[j] * e, alpha,beta,x1,x2,computeBound)
        if computeBound:
            thisB = np.max(np.abs(psi))
            B = np.append(B, thisB)
            thisC = np.max(np.abs(t[j] * e * dpsi))
            C = np.append(C, thisC)
        thisH = (V @ np.diag(psi) @ VH).reshape(1, N, N)
        H = np.concatenate((H, thisH), axis = 0)
    # Create h (low pass)
    h = np.exp(-(e/(0.6*eMin)) ** 4)
    if computeBound:
        thisB = np.max(np.abs(h))
        B = np.insert(B, 0, thisB)
        dh = h * (-4/(0.6*eMin) * (e/(0.6*eMin)) ** 3)
        thisC = np.max(np.abs(e * dh))
        C = np.insert(C, 0, thisC)
    h = (V @ np.diag(np.max(np.abs(H))*h) @ VH).reshape(1, N, N)
    H = np.concatenate((h, H), axis = 0) # J x N x N
    return H, B, C

def HannKernel(x, J, R, eMax):
    """
    HannKernel: computes the value sof the Hann kernel function
    
    See eq. (9) of D. Shuman, C. Wiesmeyr, N. Holighaus, and P. Vandergheynst,
        "Spectrum-adapted tight graph wavelet and vertex-frequency frames,"
        IEEE Trans. Signal Process., vol. 63, no. 16, pp. 4223–4235, Aug. 2015.
    
    Input:
        x (np.array): input to the function (of shape N)
        J (int): number of scales (M in the paper cited above)
        R (int): scaling factor (R in the paper cided above)
        eMax (float): upper bound on the spectrum (gamma in the citation above)
        
    Output:
        y (np.array): value of the Hann kernel function evaluated on x
    """
    y = 0.5 + 0.5 * np.cos(2. * np.pi * (J + 1. - R)/(R * eMax) * x  + 0.5)
    y[x >= 0] = 0.
    y[x < (-R * eMax/(J + 1. - R))] = 0.
    return y

def tightHannWavelet(V, E, J, R, eMax, doWarping = True):
    """
    tightHannWavelet: computes the filter bank for the tight wavelets 
        constructed using a Hann kernel function and warping
    
    See eq. (9) of D. Shuman, C. Wiesmeyr, N. Holighaus, and P. Vandergheynst,
        "Spectrum-adapted tight graph wavelet and vertex-frequency frames,"
        IEEE Trans. Signal Process., vol. 63, no. 16, pp. 4223–4235, Aug. 2015.
    Also see eq. (12) and (13) for warping.
    
    Input:
        V (np.array): eigenvector matrix (shape N x N)
        E (np.array): eigenvalue matrix (diagonal matrix of shape N x N)
        J (int): number of scales
        R (int): scaling factor (R in eq. (9) of the paper cited above)
        eMax (float): upper bound on the eigenvalues
        doWarping (float): do a log(x) warping if True (default: True)
        
    Output:
        H (np.array): of shape J x N x N contains the J matrices corresponding
            to all the filter scales
    """
    e = np.diag(E) # eigenvalues
    VH = V.conj().T # V hermitian
    N = V.shape[0] # Number of nodes
    # Create wavelets
    H = np.empty([0, N, N])
    if doWarping:
        e = np.log(e) # Warping
        e[np.isnan(e)] = -infiniteNumber
        sumPsiSquared = np.zeros(N) # If there's warping, I have to add all the
            # kernels to build the scaling function (eq. 13)
        eMax = np.log(eMax)
    t = np.arange(1,J+1) * eMax / (J + 1 - R) # translations
    for j in range(0, J-1): # If there is no warping, then we should go all the
        # way to J, but if there's warping, we have to add the scaling function
        # at the beginning
        psi =  HannKernel(e - t[j], J, R, eMax)
        if doWarping:
            sumPsiSquared += np.abs(psi) ** 2
        thisH = (V @ np.diag(psi) @ VH).reshape(1, N, N)
        H = np.concatenate((H, thisH), axis = 0)
    if doWarping:
        psi = R * 0.25 + R/2 * 0.25 - sumPsiSquared
        psi[np.abs(psi) < zeroTolerance] = 0
        psi = np.sqrt(psi)
        # Once we built the scaling function, we have to add it at the beginning
        # instead of at the end
        thisH = (V @ np.diag(psi) @ VH).reshape(1, N, N)
        H = np.concatenate((thisH, H), axis = 0)
    else:
        psi =  HannKernel(e - t[J], J, R, eMax)
        thisH = (V @ np.diag(psi) @ VH).reshape(1, N, N)
        H = np.concatenate((H, thisH), axis = 0)
    return H

def diffusionWavelets(J, T):
    """
    diffusionWavelets: computes the filter bank for the diffusion wavelets
    
    See R. R. Coifman and M. Maggioni, “Diffusion wavelets,” Appl. Comput.
        Harmonic Anal., vol. 21, no. 1, pp. 53–94, July 2006.
    Alternatively, see eq. (6) of F. Gama, A. Ribeiro, and J. Bruna, “Diffusion
        Scattering Transforms on Graphs,” in 7th Int. Conf. Learning 
        Representations. New Orleans, LA: Assoc. Comput. Linguistics, 
        6-9 May 2019, pp. 1–12.
    
    Input:
        J (int): number of scales
        T (np.array): lazy diffusion operator
        
    Output:
        H (np.array): of shape J x N x N contains the J matrices corresponding
            to all the filter scales
    """
    # J is the number of scales, and we do waveletgs from 0 to J-1, so it always
    # needs to be at least 1: I need at last one scale
    assert J > 0
    N = T.shape[0] # Number of nodes
    assert T.shape[1] == N # Check it's a square matrix
    I = np.eye(N) # Identity matrix
    H = (I - T).reshape(1, N, N) # 1 x N x N
    for j in range(1,J):
        thisPower = 2 ** (j-1) # 2^(j-1)
        powerT = np.linalg.matrix_power(T, thisPower) # T^{2^{j-1}}
        thisH = powerT @ (I - powerT) # T^{2^{j-1}} * (I - T^{2^{j-1}})
        H = np.concatenate((H, thisH.reshape(1,N,N)), axis = 0)
    return H

class GraphScatteringTransform:
    """
    graphScatteringTransform: base class for the computation of the graph 
        scattering transform coefficients
        
    Initialization:
        
    Input:
        numScales (int): number of wavelet scales (size of the filter bank)
        numLayers (int): number of layers
        adjacencyMatrix (np.array): of shape N x N
        
    Output:
        Creates graph scattering transform base handler
        
    Methods:
        
        Phi = .computeTransform(x): computes the graph scattering coefficients
            of input x (where x is a np.array of shape B x F x N, with B the
            batch size, F the number of node features, and N the number of 
            nodes)
    """
    
    # We use this as base class to then specify the wavelet and the self.U
    # All of them use np.abs() as noinlinearity. I could make this generic
    # afterward as well, but not for now.
    
    def __init__(self, numScales, numLayers, adjacencyMatrix):
        
        self.J = numScales
        self.L = numLayers
        self.W = adjacencyMatrix.copy()
        self.N = self.W.shape[0]
        assert self.W.shape[1] == self.N
        self.U = None
        self.H = None
        
    def computeTransform(self, x):
        # Check the dimension of x: batchSize x numberFeatures x numberNodes
        assert len(x.shape) == 3
        B = x.shape[0] # batchSize
        F = x.shape[1] # numberFeatures
        assert x.shape[2] == self.N
        # Start creating the output
        #   Add the dimensions for B and F in low-pass operator U
        U = self.U.reshape([1, self.N, 1]) # 1 x N x 1
        #   Compute the first coefficient
        Phi = x @ U # B x F x 1
        rhoHx = x.reshape(B, 1, F, self.N) # B x 1 x F x N
        # Reshape U once again, because from now on, it will have to multiply
        # J elements (we want it to be 1 x J x N x 1)
        U = U.reshape(1, 1, self.N, 1) # 1 x 1 x N x 1
        U = np.tile(U, [1, self.J, 1, 1])
        # Now, we move to the rest of the layers
        for l in range(1,self.L): # l = 1,2,...,L
            nextRhoHx = np.empty([B, 0, F, self.N])
            for j in range(self.J ** (l-1)): # j = 0,...,l-1
                thisX = rhoHx[:,j,:,:] # B x J x F x N
                thisHx = thisX.reshape(B, 1, F, self.N) \
                            @ self.H.reshape(1, self.J, self.N, self.N)
                    # B x J x F x N
                thisRhoHx = np.abs(thisHx) # B x J x F x N
                nextRhoHx = np.concatenate((nextRhoHx, thisRhoHx), axis = 1)
                
                phi_j = thisRhoHx @ U # B x J x F x 1
                phi_j = phi_j.squeeze(3) # B x J x F
                phi_j = phi_j.transpose(0, 2, 1) # B x F x J
                Phi = np.concatenate((Phi, phi_j), axis = 2) # Keeps adding the
                    # coefficients
            rhoHx = nextRhoHx.copy()
        
        return Phi

class DiffusionScattering(GraphScatteringTransform):
    """
    DiffusionScattering: diffusion scattering transform
    
    Initialization:
    
    Input:
        numScales (int): number of wavelet scales (size of the filter bank)
        numLayers (int): number of layers
        adjacencyMatrix (np.array): of shape N x N
        
    Output:
        Instantiates the class for the diffusion scattering transform
        
    Methods:
        
        Phi = .computeTransform(x): computes the diffusion scattering
            coefficients of input x (np.array of shape B x F x N, with B the
            batch size, F the number of node features, and N the number of 
            nodes)
    """
    
    def __init__(self, numScales, numLayers, adjacencyMatrix):
        super().__init__(numScales, numLayers, adjacencyMatrix)
        d = np.sum(self.W, axis = 1)
        killIndices = np.nonzero(d < zeroTolerance)[0] # Nodes with zero
            # degree or negative degree (there shouldn't be any since (i) the
            # graph is connected -all nonzero degrees- and (ii) all edge
            # weights are supposed to be positive)
        dReady = d.copy()
        dReady[killIndices] = 1.
        # Apply sqrt and invert without fear of getting nans or stuff
        dSqrtInv = 1./np.sqrt(dReady)
        # Put back zeros in those numbers that had been failing
        dSqrtInv[killIndices] = 0.
        # Inverse diagonal squareroot matrix
        DsqrtInv = np.diag(dSqrtInv)
        # Normalized adjacency
        A = DsqrtInv.dot(self.W.dot(DsqrtInv))
        # Lazy diffusion
        self.T = 1/2*(np.eye(self.N) + A)
        # Low-pass average operator
        self.U = d/np.linalg.norm(d, 1)
        # Construct wavelets
        self.H = diffusionWavelets(self.J, self.T)    
    
class MonicCubic(GraphScatteringTransform):
    """
    MonicCubic: graph scattering transform using monic cubic polynomial wavelets
    
    Initialization:
    
    Input:
        numScales (int): number of wavelet scales (size of the filter bank)
        numLayers (int): number of layers
        adjacencyMatrix (np.array): of shape N x N
        computeBound (bool, default: True): computes the frame bound and the
            integral Lipschitz constant
        normalize (bool, default: True): use normalized Laplacian as the graph
            shift operator on which to build the wavelets (if False, use the
            graph "combinatorial" Laplacian)
        alpha (float, default: 2): exponent of the lower interval monic
            polynomial
        beta (float, default: 2): exponent of the upper interval monic
            polynomial
        K (int, default: 20): factor to determine minimum eigenvalue resolution
        
    Output:
        Instantiates the class for the graph scattering transform using monic
        cubic polynomials
        
    Methods:
        
        Phi = .computeTransform(x): computes the graph scattering coefficients
            of input x using monic cubic polynomials (and where x is a np.array
            of shape B x F x N, with B the batch size, F the number of node 
            features, and N the number of nodes)
            
        B = .getFilterBound(): returns the scalar (float) with the upper bound
            of the filter frame
        
        C = .getIntegralLipschitzConstant(): returns the scalar (float) with the
            largest integral Lipschitz constant of all the filters in the bank
    """
    
    def __init__(self, numScales, numLayers, adjacencyMatrix,
                 computeBound = True, normalize = True,
                 alpha=2, beta=2, K = 20):
        super().__init__(numScales, numLayers, adjacencyMatrix)
        S = graphTools.adjacencyToLaplacian(self.W) # Laplacian
        if normalize:
            self.S = graphTools.normalizeLaplacian(S)
        else:
            self.S = S
        self.E, self.V = graphTools.computeGFT(self.S, order = 'increasing')
        eMax = np.max(np.diag(self.E))
        x1 = np.diag(self.E)[np.floor(self.N/4).astype(np.int)]
        x2 = np.diag(self.E)[np.ceil(3*self.N/4).astype(np.int)]
        # Low-pass average operator
        #self.U = self.V[:, 0] # v1
        self.U = (1/self.N) * np.ones(self.N)
        # Construct wavelets
        self.H, self.B, self.C = monicCubicWavelet(self.V, self.E, self.J,
                                                   alpha, beta, x1, x2, K, eMax,
                                                   computeBound)
        # self.H is the J x N x N matrix with all the filters
        # self.B is the bound on the filters, a vector of size J
        # self.C is the integral Lipschitz constant of the filter for each of 
        # the filters (vector of size J)
        # Obs.: If computeBound is false, then both self.B and self.C are None
    
    def getIntegralLipschitzConstant(self):
        
        return self.C
    
    def getFilterBound(self):
        
        return self.B
        
class TightHann(GraphScatteringTransform):
    """
    TightHann: graph scattering transform using a tight frame with Hann wavelets
    
    Initialization:
    
    Input:
        numScales (int): number of wavelet scales (size of the filter bank)
        numLayers (int): number of layers
        adjacencyMatrix (np.array): of shape N x N
        normalize (bool, default: True): use normalized Laplacian as the graph
            shift operator on which to build the wavelets (if False, use the
            graph "combinatorial" Laplacian)
        R (int, default: 3): scaling factor
        doWarping (bool, default: True): do a log(x) warping
        
    Output:
        Instantiates the class for the graph scattering transform using a tight
        frame of Hann wavelets
        
    Methods:
        
        Phi = .computeTransform(x): computes the graph scattering coefficients
            of input x using a tight frame of Hann wavelets (and where x is a
            np.array of shape B x F x N, with B the batch size, F the number of
            node features, and N the number of nodes)
    """
    
    def __init__(self, numScales, numLayers, adjacencyMatrix,
                 normalize = True, R = 3, doWarping = True):
        super().__init__(numScales, numLayers, adjacencyMatrix)
        S = graphTools.adjacencyToLaplacian(self.W) # Laplacian
        if normalize:
            self.S = graphTools.normalizeLaplacian(S)
        else:
            self.S = S
        self.E, self.V = graphTools.computeGFT(self.S, order = 'increasing')
        eMax = np.max(np.diag(self.E))
        if R > self.J:
            R = self.J - 1
        # Low-pass average operator
        #self.U = self.V[:, 0] # v1
        self.U = (1/self.N) * np.ones(self.N)
        # Construct wavelets
        self.H = tightHannWavelet(self.V, self.E, self.J, R, eMax, doWarping)
