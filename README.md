# Graph Scattering Transforms
Code for experimentation on graph scattering transforms. If any part of this code is used, the following paper must be cited:

Any questions, comments or suggestions, please e-mail Fernando Gama at fgama@seas.upenn.edu. The specific random seeds used to get the results that appear in the paper can be obtained by request.

## Wavelets
Graph scattering transform depend on a multi-resolution graph wavelet filter bank. This code implements three different wavelet filter banks, namely:

1. <i>Diffusion wavelets:</i>

F. Gama, A. Ribeiro, and J. Bruna, "<a href="https://openreview.net/forum?id=BygqBiRcFQ">Diffusion Scattering Transforms on Graphs</a>," in 7th Int. Conf. Learning Representations. New Orleans, LA: Assoc. Comput. Linguistics, 6-9 May 2019, pp. 1–12.

R. R. Coifman and M. Maggioni, "<a href="https://www.sciencedirect.com/science/article/pii/S106352030600056X">Diffusion wavelets</a>," Appl. Comput. Harmonic Anal., vol. 21, no. 1, pp. 53–94, July 2006.

2. <i>Monic polynomials interpolated by a cubic polynomial:</i>

D. K. Hammond, P. Vandergheynst, and R. Gribonval, "<a href="https://www.sciencedirect.com/science/article/pii/S1063520310000552">Wavelets on graphs via spectral graph theory</a>," Appl. Comput. Harmonic Anal., vol. 30, no. 2, pp. 129–150, March 2011.

3. <i>Tight frames of Hann wavelets:</i>

D. I. Shuman, C. Wiesmeyr, N. Holighaus, and P. Vandergheynst, "<a href="https://ieeexplore.ieee.org/document/7088640">Spectrum-adapted tight graph wavelet and vertex-frequency frames</a>," IEEE Trans. Signal Process., vol. 63, no. 16, pp. 4223–4235, Aug. 2015.

## Datasets
Three experiments are run.

<p>1. The first one, under the name <code>GSTsmallWorld.py</code> is a synthetic experiment on a small world graph. The objective of this experiment is to showcase the effect of the stability bound on graph scattering transform under a controlled environment.</p>

D. J. Watts and S. H. Strogatz, "<a href="https://www.nature.com/articles/30918">Collective dynamics of 'small-world' networks</a>," Nature, vol. 393, no. 6684, pp. 440–442, June 1998.

<p>2. The second one, on the file <code>GSTauthorship.py</code> considers the problem of authorship attribution. It demonstrates the richness of the graph scattering representation on a real dataset. The dataset is available under <code>datasets/authorshipData/</code> and the following paper must be cited whenever such dataset is used</p>

S. Segarra, M. Eisen, and A. Ribeiro, "<a href="https://ieeexplore.ieee.org/document/6638728">Authorship attribution through function word adjacency networks</a>," IEEE Trans. Signal Process., vol. 63, no. 20, pp. 5464–5478, Oct. 2015.

<p>3. The third and last one, on the file <code>GSTfacebook.py</code> studies the problem of source localization on a subnetwork of a Facebook graph. This experiment is meant to also show the richness of the graph scattering representation under a real graph perturbation, with a synthetic signal defined on top of it. The original dataset is available <a href="https://snap.stanford.edu/data/ego-Facebook.html">here</a>, and use of this dataset must cite the following paper</p>

J. McAuley and J. Leskovec, "<a href="https://papers.nips.cc/paper/4532-learning-to-discover-social-circles-in-ego-networks">Learning to discover social circles in Ego networks</a>," in 26th Neural Inform. Process. Syst. Stateline, TX: NIPS Foundation, 3-8 Dec. 2012.

## Code
The code is written in Python3. Details as follows.

### Dependencies
The following Python libraries are required: <code>os</code>, <code>numpy</code>, <code>matplotlib</code>, <code>pickle</code>, <code>datetime</code>, <code>sklearn</code>, <code>hdf5storage</code>, <code>urllib</code>, <code>gzip</code>, <code>shutil</code> and <code>scipy</code>, as well as a LaTeX installation.

### Concept
The three main files <code>GSTauthorship.py</code>, <code>GSTfacebook.py</code> and <code>GSTsmallWorld.py</code> consists of the three main experiments. The first lines of code on each of those files (after importing the corresponding libraries) have the main parameters defined, which could be edited for running experiment under a different set of hyperparameters.

The directory <code>Modules/</code> contains the implemented graph scattering transforms. In most cases, it has a function that just compute the corresponding equation, a wavelet function that computes the graph wavelet filter bank, and a graph scattering transform class that computes the entire representation.

Under <code>Utils/</code> there are three modules containing utilitary classes and functions, such as those handling the graph <code>graphTools.py</code>, handling the data <code>dataTools.py</code> and some miscellaneous tools <code>miscTools.py</code>.

### Run
The code runs by simply executing <code>python GSTsmallWorld.py</code> or any of the two other experiments (assuming <code>python</code> is the command for Python3). For running the authorship attribution experiment, the .rar files on <code>datasets/</code> have to be uncompressed. For obtaining the entire Facebook graph, a connection to the internet is required since the dataset is download from its <a href="https://snap.stanford.edu/data/ego-Facebook.html">source</a>.
