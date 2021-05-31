# -*- coding: utf-8 -*-

"""
    Module for the metropolis-based samplers
    @author: Navid Hedjazian
"""
import numpy as np
from datetime import datetime
from numpy.random import default_rng


from .mcmc_base import MCMCBase

STATS = {
    "loglikelihood": ["_x_loglike", 1],
    "accept_ratio": ["_general_accept_ratio", 1],
    "prop_S": ["prop_S", "n_vars"],
    "parameter_accept_ratio": ["_accept_ratios", "n_vars"]
}


class Metropolis1dStep(MCMCBase):
    """
    Class Metropolis1dStep.

    Run a Metropolis MCMC algorithm using 1D steps only.

    Description of attributes:

    MCMC parameters:
    ----------------
        # Information for MCMC
        n_vars : integer
            total number of variables to sample.

        n_samples : integer
            number of saved iterations in the chain.

        show_stats : integer
            interval n° of samples to show stats.

        # Functions

        loglikelihood : function
            Log-likelihood function.

        logprior : function
            Prior probability function.

        proposal : function
            Proposal function.

        prop_S : 1d numpy array, shape (n_vars)
            Standard deviation for the proposal function.

    MCMC results:
    -------------
        samples : ndarray shape (n_samples, n_vars)
            The samples in the parameter space.

        save_stats : dictionary
            Container for the stats to save during simulation.

        duration : datetime
            Duration of the last run.

    """

    # How to define prior, proposal and loglikelihood functions
    # Proposal
    # def proposal(x, prop_S)
    #
    # Prior
    # def logprior(x)
    #
    # Loglikelihood
    # def loglikelihood(x)
    #

    def __init__(self, n_vars=None, verbose=0, show_stats=10000):
        super(Metropolis1dStep, self).__init__(n_vars=n_vars,
                                               verbose=verbose,
                                               show_stats=show_stats)

        # MCMC state
        self._current_iter = 1
        self._recent_n_prop = None
        self._recent_n_accept = None
        self._prop_mat = None
        self._accept_mat = None
        self._accept_ratios = None
        self._general_accept_ratio = None
        self._starting_sample = None
        self._init_prop_S = None  # Saves the initial proposal std
        self._x_loglike = None  # Saves the current loglikelihood

    # TODO: define a default for the getattr method

    def add_stat(self, name):
        if name in STATS.keys():
            self._save_stats.append(name)
        else:
            print("Invalid stat, chose from:")
            print(STATS.keys())

    def print_stats(self, i_sample):
        """
        Print the stats that are recorded
        """
        print()
        print("---------------------------------")
        print("        -- MCMC stats --         ")
        print("---------------------------------")
        print("At sample number {}".format(self._current_iter))
        for stat in self._save_stats:
            print(stat)
            print(self.stats[stat][i_sample, ...])

    def initialize(self):
        """
        Run all functions required before starting or restarting the MCMC
        algorithm
        1. Re-initialize results arrays and set state to 0
        2. Reset adaptive parameters
        """
        self.reset()
        pass

    def initialize_arrays(self, x0, n, thin, tune, tune_interval,
                          discard_tuning):
        self.n_vars = x0.size
        # Results arrays
        self.n_samples = n
        if discard_tuning:
            self.n_samples -= tune
        self.n_samples = n // thin

        self.samples = np.zeros((self.n_samples, self.n_vars),
                                dtype=self.sample_dtype)

        # Set self.stats as a dictionary of numpy arrays for each stat
        for stat in self._save_stats:
            # get the stats attributes, the second element of the list is the
            # number of variables
            attr = STATS[stat][1]
            if isinstance(attr, str):
                n_params = int(getattr(self, attr))
            else:
                n_params = int(attr)
            self.stats[stat] = np.zeros((self.n_samples, n_params))

        # Private arrays
        self._current_iter = 1
        self._recent_n_prop = np.zeros(self.n_vars, dtype=np.int32)
        self._recent_n_accept = np.zeros(self.n_vars, dtype=np.int32)
        self._prop_mat = np.zeros((tune_interval, self.n_vars),
                                  dtype=np.bool_)
        self._accept_mat = np.zeros((tune_interval, self.n_vars),
                                    dtype=np.bool_)
        self._accept_ratios = np.zeros(self.n_vars)

    def reset(self):
        """
        Function to reset the parameters learned during the sampling, such as
        proposal parameters.
        """
        self.prop_S = self._init_prop_S

    def save_sample(self, x, i):
        """
        Store vector x as the ith sample
        Parameters
        ----------
        x : 1D numpy array
        i: integer
        """
        self.samples[i, :] = x  # Add point to history of accepted models

    def tune(self):
        """
        Tune the proposal standard deviation according to the previous
        samples acceptance.

        Taken from pymc3 documentation:
            Rate    Variance adaptation
            ----    -------------------
            <0.001        x 0.1
            <0.05         x 0.5
            <0.2          x 0.9
            >0.5          x 1.1
            >0.75         x 2
            >0.95         x 10
        Returns
        -------

        """
        for i in range(self.n_vars):
            if self._accept_ratios[i] < 0.001:
                self.prop_S[i] *= 0.1
                continue
            elif self._accept_ratios[i] < 0.05:
                self.prop_S[i] *= 0.5
                continue
            elif self._accept_ratios[i] < 0.2:
                self.prop_S[i] *= 0.9
                continue
            elif self._accept_ratios[i] > 0.95:
                self.prop_S[i] *= 10.
                continue
            elif self._accept_ratios[i] > 0.75:
                self.prop_S[i] *= 2.
                continue
            elif self._accept_ratios[i] > 0.5:
                self.prop_S[i] *= 1.1
                continue

    def run(self, x0, n, tune=0, tune_interval=1000,
            discard_tuned_samples=False, thin=1):
        """
        Runs the algorithm.

        Parameters
        ----------
        x0 : 1D numpy array
            Starting sample.

        n : int
            Number of iterations.

        tune : int
            Number of samples where proposal tuning is allowed.

        tune_interval : int
            Interval of samples for tuning the proposal.

        discard_tuned_samples : bool
            Do or do not save tuned samples. Default to False.

        thin : int
            Thining of the chain, save the sample every number of iterations.

        """
        start_time = datetime.now()
        # Initialize arrays
        self.initialize_arrays(x0, n, thin, tune, tune_interval,
                               discard_tuned_samples)
        # Initialize numpy random generator
        rng = default_rng()
        print("Start Metropolis1dStep")
        print("----------------------")
        print("number of saved samples")
        print(self.n_samples)

        x = x0

        # Initialize likelihood and prior
        self._x_loglike = self.loglikelihood(x)
        x_logprior = self.logprior(x)
        if self.verbose > 1:
            print("Starting log-likelihood")
            print(self._x_loglike)
            print()

        # Initialize other parameters
        n_accept = 0
        i_sample = 0

        # Start MCMC sampling
        # -------------------
        for i in range(n):

            if self.verbose > 1:
                print("ITER", i)
                print(" ------------------------------------")

            # perturb the proposal point x to xp
            xp = self.proposal(x, self.prop_S)
            modified_parameters_list = np.nonzero(xp - x)

            # Roll prop and accept matrix
            self._prop_mat = np.roll(self._prop_mat, 1, axis=0)
            self._prop_mat[0, :] = False
            self._accept_mat = np.roll(self._accept_mat, 1, axis=0)
            self._accept_mat[0, :] = False

            # Saves which parameter has been proposed
            for param_i in modified_parameters_list:
                self._prop_mat[0, param_i] = True

            if self.verbose > 1:
                print("Modified parameter")
                print(modified_parameters_list)
                print("Proposal x")
                print(xp - x)
                print(xp)

            # Get the prior at xp
            xp_logprior = self.logprior(xp)

            if xp_logprior == -np.inf:
                compute_likelihood = False
                if self.verbose > 1:
                    print("Prior is {}".format(xp_logprior))
                    print("Do not compute log-likelihood")
            else:
                compute_likelihood = True

            if compute_likelihood:
                xp_loglike = self.loglikelihood(xp)
            else:
                xp_loglike = - np.inf

            u = rng.random()
            if self.verbose > 1:
                print("       LOGLIKE       ")
                print("---------------------")
                print(self._x_loglike, xp_loglike)

                print("")
                print("       PRIOR       ")
                print("-------------------")
                print(x_logprior, xp_logprior)

                print("rand u is ", u)
                print("Accept if LOG(PRIOR) + LOGLIKE diff > log(u)")
                print("log u ", np.log(u))
                print("LOGLIKE difference", xp_loglike - self._x_loglike)
                print("")

            # Draw a random number between 0 and 1
            # def accept():

            if np.log(u) < xp_logprior + xp_loglike - self._x_loglike - \
                    x_logprior:
                # case accept
                np.copyto(x, xp)  # copy xp into x
                self._x_loglike = xp_loglike  # update likelihood value
                x_logprior = xp_logprior  # update log prior
                n_accept += 1  # Update accepted moves

                if self.verbose > 1:
                    print(" Accept iter {} \n".format(i))

                # Saves which parameter has been accepted
                for param_i in modified_parameters_list:
                    self._accept_mat[0, param_i] = True

            # Update acceptance ratios
            self._recent_n_prop = np.sum(self._prop_mat, axis=0)
            self._recent_n_accept = np.sum(self._accept_mat, axis=0)
            self._accept_ratios = \
                np.true_divide(self._recent_n_accept, self._recent_n_prop,
                               where=(self._recent_n_prop != 0))

            self._general_accept_ratio = float(n_accept) / float(
                self._current_iter)

            if self._current_iter % tune_interval == 0 and \
                    self._current_iter < tune:
                self.tune()
                if self.verbose > 0:
                    print()
                    print("----------------------")
                    print(" Tune MCMC parameters ")
                    print("----------------------")
                    print("New standard deviation of proposal distribution at "
                          "iter {}:".format(self._current_iter))
                    with np.printoptions(precision=2, suppress=True):
                        print(self.prop_S)
                    print()

            if self._current_iter % self.show_stats == 0:
                self.print_stats(i_sample)
                if self.verbose > 1:
                    print("Position x: {}".format(x))
                    print("Acceptance rate for each parameter:")
                    print(self._accept_ratios)
                print()

            # Records the proposal point
            if n % thin == 0:
                i_sample = int(i / thin)
                self.save_sample(x, i_sample)
                for stat in self._save_stats:
                    # get the attribute name
                    attr = STATS[stat][0]
                    # save it to dictionnary
                    self.stats[stat][i_sample, ...] = getattr(self, attr)

            # Update iter
            self._current_iter += 1

        # Saves the final quantities
        self.duration = datetime.now() - start_time
