 # -*- coding: utf-8 -*-

"""
    Module that defines functions to sample a parameter space with MCMC
    @author: Navid Hedjazian
"""


import numpy as np
import arviz
import pickle


class MCMCBase:
    """
    Abstract class for MCMC methods
    """

    def __init__(self, n_vars=None, verbose=0, show_stats=10000):

        # utils
        self.verbose = verbose
        # MCMC method
#        self.hierarchical = 0
#        self.n_hyperparameters = 0
#        self.hierarchical_proposal_probability = 0.25
        self.proposal = None
        self.logprior = None
        self.loglikelihood = None

        # MCMC params
        self.n_samples = None
        self.show_stats = show_stats
        self._tune_counter = None

        # Parameters
        # self.n_model_vars = None  # Number of normal parameters # ONLY
        # required for hierarchical or
        self.n_vars = n_vars  # Number of parameters
        self.prop_S = None

        # MCMC results
        self.sample_dtype = np.float64  # Default to float
        self.samples = None

        self._save_stats = []
        self.stats = {}

        self.duration = None

    def write(self, filename, format="pickle"):
        """
        Write the search result object with python pickle.

        Parameters
        ----------
        filename :  str
            Name of the file / path where the object is written to

        format : str
            Format type.
            "pickle" : python pickle objects
        """
        if format == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(self, f)  # Use default python 3.0 protocol

    def write_samples(self, filename, format='arviz'):
        """
        Write the search result object with python pickle.

        Parameters
        ----------
        filename :  str
            Name of the file / path where the object is written to


        format : str
            "xarray": saves as an xarray/netCDF format
        """
        obj = self.get_results(format=format)
        return

    @classmethod
    def read(cls, filename, format="pickle"):
        """
        Read the MCMC object.

        Parameters
        ----------
        filename : str
            File name / path of the object to be loaded.

        Returns
        -------
        obj : MCMC object
        """
        obj = None
        if format == 'pickle':
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            assert(isinstance(obj, MCMCBase)), \
                "File does not contain the correct class object."
        return obj

    def initialize_arrays(self, *args):
        raise NotImplementedError("Must be implemented by subclass.")

    def save_sample(self, x, i):
        """
        Store vector x as the ith sample
        Parameters
        ----------
        x : 1D numpy array
        i: integer
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def tune(self):
        """
        Function to update the mcmc parameters. Currently implements tuning
        the proposal distribution
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def reset(self):
        """
        Function to reset the parameters learned during the sampling, such as
        proposal parameters.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    # def acceptance(self, *args):
    #     raise NotImplementedError("Must be implemented by subclass.")
    #
    # def select(self, *args):
    #     raise NotImplementedError("Must be implemented by subclass.")

    def run(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by subclass.")

    def get_results(self, format="xarray"):
        """
        Returns the results of the mcmc algorithm to a specific format.
        It contains the samples and the stats of the chain.

        Parameters
        ----------
        format : str
            'xarray' : Returns a xarray dataset
            'arviz' : Returns a arviz InrefenceData object

        Returns
        -------
        out : object
            The object type is given in 'format' parameter.
        """
        if format == 'xarray' or format == 'arviz':
            data = dict(self.stats)
            # Add a first dimension for chain number
            for key in data:
                data[key] = np.expand_dims(data[key], axis=0)
            attrs = {'n_vars': self.n_vars, 'n_samples': self.n_samples,
                     'duration': self.duration}
        if format == 'xarray':
            data['x'] = np.expand_dims(self.samples, axis=0)
            dataset = arviz.dict_to_dataset(data, attrs=attrs)
        elif format == 'arviz':
            post = {'x': np.expand_dims(self.samples, axis=0)}
            if 'loglikelihood' in data.keys():
                ll = {'log_likelihood': data['loglikelihood']}
            else:
                ll = {}
            dataset = arviz.from_dict(posterior=post, log_likelihood=ll)
        return dataset

    ############################################################################
    # Post-processing tools
    ############################################################################
    def cut_chain(self, i=0):
        """
        Remove the few first samples of a chain. Use it to remove a burn-in
        period for example, and keep only the part of the chain that has
        reached equilibrium.
        Note that all diagnostics quantities such as acceptance and others
        will not be valid anymore. They need to be recalculated.
        Parameters
        ----------
        i : integer
            The number of samples to remove.

        Returns
        -------

        """
        self.samples = self.samples[i:, :]
        # self._stats = self._stats[i:] # implement in subclass
        self.n_samples = self.samples.shape[0]


# def merge_mcmc(mcmc_list):
#     """
#     Merge MCMC objects into one
#     :param mcmc_list: list of MCMC objects
#     :return: a mcmc object containing the samples of each chain
#     """
#     assert(isinstance(mcmc_list[0], MCMCBase))
#     merged_mcmc = copy.deepcopy(mcmc_list[0])
#     n_chains = len(mcmc_list)
#
#     # initialize MCMC params of interest
#     merged_mcmc.n_iter = 0
#     merged_mcmc._starting_sample = list()
#     n_accept = 0
#     for i in range(n_chains):
#         merged_mcmc.n_iter += mcmc_list[i].n_iter
#         merged_mcmc._starting_sample.append(mcmc_list[i]._starting_sample)
#         n_accept += int(mcmc_list[i].chain_accept_ratio * mcmc_list[i].n_iter)
#
#     merged_mcmc.chain_accept_ratio = n_accept / merged_mcmc.n_iter
#
#     merged_mcmc.samples = np.concatenate([mcmc_list[i].samples for i in
#                                           range(n_chains)])
#
#     merged_mcmc._stats = np.concatenate(
#         [mcmc_list[i]._stats for i in range(n_chains)])
#
#     return merged_mcmc


def gelman_diagnostic(mcmc_list):
    """
    Compute the Gelman-Rubin diagnostic for MCMC convergence.
    :param mcmc_list: a list of markov chains
    :return: R, that should be < 1.1
    """
    n_chains = len(mcmc_list)
    n_params = mcmc_list[0].n_vars
    n = mcmc_list[0].n_samples
    chains_means = np.zeros((n_chains, n_params))
    chains_variances = np.zeros((n_chains, n_params))
    # mean and variance of each parameter in each chain
    for i in range(n_chains):
        for j in range(n_params):
            chains_means[i, j] = np.mean(mcmc_list[i].samples[:, j])
            chains_variances[i, j] = np.var(mcmc_list[i].samples[:, j])
    # Calculate ratio
    ratio = np.zeros(n_params)
    for j in range(n_params):
        pooled_mean = np.mean(chains_means[:, j])
        within_var = np.mean(chains_variances[:, j])
        between_var = n/(n_chains - 1) * np.sum(
            (chains_means[:, j] - pooled_mean)**2)
        v = (n - 1) / n * within_var + (n_chains + 1)/(n_chains*n)*between_var
        ratio[j] = np.sqrt(v/within_var)
    return ratio
