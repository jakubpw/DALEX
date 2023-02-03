import numpy as np
from tqdm import tqdm
import copy
import itertools

from scipy.special import binom
from shap.utils import safe_isinstance

from . import utils
from shap import Explanation


class Shapr(Explanation):
    """Uses the Kernel SHAPR method to explain the output of any function.
        Kernel SHAPR is a method that uses a special weighted linear regression
        to compute the importance of each feature. The computed importance values
        are Shapley values from game theory and also coefficents from a local linear
        regression.
        Parameters
        ----------
        model : function
            User supplied function that takes a matrix of samples (# samples x # features) and
            computes a the output of the model for those samples. The output can be a vector
            (# samples) or a matrix (# samples x # model outputs).
        data : numpy.array
            The background dataset to use for integrating out features. To determine the
            impact of a feature, that feature is set to "missing" and the change in the model output
            is observed. Since most models aren't designed to handle arbitrary missing data at test
            time, we simulate "missing" by taking average of model outputs on samples replacing the feature
            with all the values it takes in the background dataset. For small problems
            this background dataset can be the whole training set, but for larger problems consider
            using a single reference value or using the kmeans function to summarize the dataset.
        mask_opt : bool
             To or not to limit the number of feature subsets that will be replaced, speed up operation.
             """

    def __init__(self, model, data, masks_opt=False):
        self.masks_opt = masks_opt
        self.model = model
        self.X = data
        self.M = data.shape[1]
        self.sigma = 0.4
        from shap import KernelExplainer
        self.expected_value = KernelExplainer(model, data).expected_value
        if masks_opt:
            self.nsamples = 2 * self.M + 2 ** 8
            self.max_samples = 2 ** 30
            if self.M <= 30:
                self.max_samples = 2 ** self.M - 2
                if self.nsamples > self.max_samples:
                    self.nsamples = self.max_samples

            # reserve space for some of our computations
            self.maskMatrix = np.zeros((self.nsamples, self.M))
            self.lastMask = np.zeros(self.nsamples)
            self.nsamplesAdded = 0

            # weight the different subset sizes
            num_subset_sizes = np.int(np.ceil((self.M - 1) / 2.0))
            num_paired_subset_sizes = np.int(np.floor((self.M - 1) / 2.0))
            weight_vector = np.array([(self.M - 1.0) / (i * (self.M - i)) for i in range(1, num_subset_sizes + 1)])
            weight_vector[:num_paired_subset_sizes] *= 2
            weight_vector /= np.sum(weight_vector)

            # fill out all the subset sizes we can completely enumerate
            # given nsamples*remaining_weight_vector[subset_size]
            num_full_subsets = 0
            num_samples_left = self.nsamples
            group_inds = np.arange(self.M, dtype='int64')
            mask = np.zeros(self.M)
            remaining_weight_vector = copy.copy(weight_vector)
            for subset_size in range(1, num_subset_sizes + 1):

                # determine how many subsets (and their complements) are of the current size
                nsubsets = binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes: nsubsets *= 2

                # see if we have enough samples to enumerate all subsets of this size
                if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                    num_full_subsets += 1
                    num_samples_left -= nsubsets

                    # rescale what's left of the remaining weight vector to sum to 1
                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

                    # add all the samples of the current subset size
                    w = weight_vector[subset_size - 1] / binom(self.M, subset_size)
                    if subset_size <= num_paired_subset_sizes: w /= 2.0
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype='int64')] = 1.0
                        self.addsample(mask)
                        if subset_size <= num_paired_subset_sizes:
                            mask[:] = np.abs(mask - 1)
                            self.addsample(mask)
                else:
                    break
            samples_left = self.nsamples - self.nsamplesAdded

            if num_full_subsets != num_subset_sizes:
                remaining_weight_vector = copy.copy(weight_vector)
                remaining_weight_vector[:num_paired_subset_sizes] /= 2  # because we draw two samples each below
                remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
                remaining_weight_vector /= np.sum(remaining_weight_vector)
                ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left, p=remaining_weight_vector)
                ind_set_pos = 0
                used_masks = {}
                while samples_left > 0 and ind_set_pos < len(ind_set):
                    mask.fill(0.0)
                    ind = ind_set[ind_set_pos]  # we call np.random.choice once to save time and then just read it here
                    ind_set_pos += 1
                    subset_size = ind + num_full_subsets + 1
                    mask[np.random.permutation(self.M)[:subset_size]] = 1.0

                    # only add the sample if we have not seen it before, otherwise just
                    # increment a previous sample's weight
                    mask_tuple = tuple(mask)
                    new_sample = False
                    if mask_tuple not in used_masks:
                        new_sample = True
                        used_masks[mask_tuple] = self.nsamplesAdded
                        samples_left -= 1
                        self.addsample(mask)

                    # add the compliment sample
                    if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)

                        # only add the sample if we have not seen it before, otherwise just
                        # increment a previous sample's weight
                        if new_sample:
                            samples_left -= 1
                            self.addsample(mask)

    def addsample(self, m):
        self.maskMatrix[self.nsamplesAdded, :] = m
        self.nsamplesAdded += 1

    def __call__(self, X):
        if safe_isinstance(X, "pandas.core.frame.DataFrame"):
            feature_names = list(X.columns)
            X = X
        else:
            feature_names = getattr(self, "data_feature_names", None)
        shap_values = self.shap_values(X)
        return Explanation(values=shap_values, data=X, feature_names=feature_names)

    def shap_values(self, X):
        phi = np.zeros((X.shape[0], self.M + 1))
        for idx, x in tqdm(enumerate(X)):
            if self.masks_opt:
                phi[idx] = utils.kernel_shapr_opt(self.model, x, self.X, self.M, self.sigma, self.maskMatrix)
            else:
                phi[idx] = utils.kernel_shapr(self.model, x, self.X, self.M, self.sigma)

        result = phi[:, :-1]
        return result