import numpy as np
import matplotlib.pyplot as plt


class Remapping:
    """
    Class heavily inspired by Yinrui Liu's implementation
    https://github.com/Yinrui-Liu/hadron-Ar_XS/blob/main/hadron-Ar_XS.ipynb
    I generalized the concept from 3d to Nd with more vectorized operations
    """
    def __init__(self, var_names):

        self.true_map = None
        self.reco_map = None
        self.true_total_bins, self.reco_total_bins = 0, 0
        self.var_names = var_names
        self.debug = False

    def remap_training_events(self, true_list, reco_list, bin_list, ndim):

        # 3D (l,m,n) -> (l*m*n) + (m*n) + n
        # total_bins = sum([nbin_list[d-1]**d for d in range(1, ndim+1)])
        # self.true_total_bins = sum([(len(bin_list[d - 1]) - 1) ** d for d in range(1, ndim + 1)])
        # self.true_total_bins = sum([(len(bin_list[d - 1]) - 1) ** (1 + ndim - d) for d in range(1, ndim + 1)])
        # self.true_total_bins = sum(np.flip(np.cumprod(np.flip([len(d)-1 for d in bin_list]))))
        self.true_total_bins = np.prod([len(d) - 1 for d in bin_list])

        true_event_weights = [np.ones(arr.shape) for arr in true_list]
        reco_event_weights = [np.ones(arr.shape) for arr in reco_list]

        true_nd_binned, true_nd_hist, true_nd_hist_err, true_nd_hist_cov = self.map_meas_to_bin_space(corr_var_list=true_list,
                                                                                               bin_list=bin_list,
                                                                                               total_bins=self.true_total_bins,
                                                                                               evt_weights=true_event_weights,
                                                                                               debug=self.debug)

        reco_nd_binned, reco_nd_hist, reco_nd_hist_err, reco_nd_hist_cov = self.map_meas_to_bin_space(corr_var_list=reco_list,
                                                                                               bin_list=bin_list,
                                                                                               total_bins=self.true_total_bins,
                                                                                               evt_weights=reco_event_weights,
                                                                                               debug=self.debug)

        print("Total Bins:", self.true_total_bins)
        print("True number Nd:", np.unique(true_nd_hist).shape)
        print("Meas number Nd:", np.unique(reco_nd_hist).shape)

        # Create map between 3D and 1D
        self.true_map, true_hist_sparse, true_hist_err_sparse = self.map_nd_to_1d(num_nd=true_nd_hist,
                                                                                  num_nd_err=true_nd_hist_err,
                                                                                  total_bins=self.true_total_bins)

        self.reco_map, reco_hist_sparse, reco_hist_err_sparse = self.map_nd_to_1d(num_nd=reco_nd_hist,
                                                                                  num_nd_err=reco_nd_hist_err,
                                                                                  total_bins=self.true_total_bins)

        print("True Map:", np.count_nonzero(self.true_map))
        print("Meas Map:", np.count_nonzero(self.reco_map))

        return (true_nd_binned, reco_nd_binned), (true_nd_hist, reco_nd_hist), (true_nd_hist_cov, reco_nd_hist_cov), \
               (true_hist_sparse, reco_hist_err_sparse)

    def remap_data_events(self, data_list, bin_list, ndim):

        # self.reco_total_bins = sum([nbin_list[d - 1]**d for d in range(1, ndim + 1)])
        # self.reco_total_bins = sum([(len(bin_list[d - 1]) - 1) ** (1+ndim - d) for d in range(1, ndim + 1)])
        #self.reco_total_bins = sum(np.flip(np.cumprod(np.flip([len(d)-1 for d in bin_list])))) # NOTE change from len() -1
        self.reco_total_bins = np.prod([len(d)-1 for d in bin_list])

        data_event_weights = [np.ones(arr.shape) for arr in data_list]

        data_nd_binned, data_nd_hist, data_nd_hist_err, data_nd_hist_cov = self.map_meas_to_bin_space(corr_var_list=data_list,
                                                                                               bin_list=bin_list,
                                                                                               total_bins=self.reco_total_bins,
                                                                                               evt_weights=data_event_weights,
                                                                                               debug=self.debug)

        # Data mapping to 1D
        data_hist_sparse, data_hist_err_sparse = self.map_data_to_1d_bins(num_nd=data_nd_hist, num_nd_err=data_nd_hist_err,
                                                                          map_nd1d=self.reco_map)
        print("Sparse Data nbins:", len(data_hist_sparse))

        return data_nd_binned, data_nd_hist, data_nd_hist_cov, data_hist_sparse, data_hist_err_sparse

    @staticmethod
    def map_meas_to_bin_space(corr_var_list, bin_list, total_bins, evt_weights, debug=False):
        """
        Convert list of correlated varibles to single ndim variable
        """
        # Mask out events that fall in underflow or overflow bins
        mask_events = np.ones(shape=corr_var_list[0].shape, dtype=bool)
        for cvar, bins in zip(corr_var_list, bin_list): # NOTE remove masking
            mask_events &= (cvar >= bins[0]) & (cvar < bins[-1])

        # Bin shifts, the last element is only added so multiply by 1
        # bin_shift = np.flip(np.cumprod([len(d) - 1 for d in bin_list]))
        # bin_shift = np.flip(np.cumprod(np.flip([len(d) - 1 for d in bin_list]))).astype('int')
        # bin_shift[-1] = 1
        bin_lens = [len(d)-1 for d in bin_list] # NOTE changed from len() -1
        bin_shift = [np.prod(bin_lens[d + 1:]).astype('int') for d, l in enumerate(bin_lens)]

        weight_array = None
        # nnd_bin_array = np.zeros(np.count_nonzero(mask_events)).astype('int')
        digitized_vars = []
        for i, arr_bin in enumerate(zip(corr_var_list, bin_list, bin_shift, evt_weights)):
            arr, bins, shift, weight = arr_bin
            nbins = len(bins) - 1
            # Mapping from measured space (usually energy) to bin space
            n_binned = np.digitize(x=arr[mask_events], bins=bins, right=False) - 1
            if debug: print("Unique bins:", np.unique(n_binned))
            if debug: print("(n_binned >= 0) & (n_binned <= (", nbins-1, ")") # was >= 1 and <= 0 # NOTE remove cut
            n_binned = n_binned[(n_binned >= 0) & (n_binned <= nbins-1)]  # ignore under/over flow bins, 0/n+1 respectively
            digitized_vars.append(n_binned)
            if debug: print("Unique Bins post under/over -flow cut:", np.unique(n_binned))
            if debug: print("Shift:", shift)
            # nnd_bin_array += shift * n_binned

        # Now ravel (unwrap) the indices
        stacked_vars = np.vstack(digitized_vars)
        ravelled_idx = np.ravel_multi_index(stacked_vars, dims=bin_lens)

        print("Max bin:", np.max(ravelled_idx))
        if debug: print("Unique Bins:", np.unique(ravelled_idx))

        # Create the histogram with event weighting and calculate errors
        evt_weight = np.ones(ravelled_idx.shape)
        num_nd, _ = np.histogram(ravelled_idx, bins=total_bins, range=(0, total_bins-1), weights=evt_weight)
        num_nd_err, _ = np.histogram(ravelled_idx, bins=total_bins, range=(0, total_bins-1), weights=evt_weight * evt_weight)
        num_nd_vcov = np.diag(num_nd_err)

        return ravelled_idx, num_nd, np.sqrt(num_nd_err), num_nd_vcov

    @staticmethod
    def map_nd_to_1d(num_nd, num_nd_err, total_bins):
        """
        Convert truth and reco 3D to 1D
        """
        # Create the maps for true and measured
        nd_to_1d_map = np.zeros(total_bins, dtype=np.int32)
        tmp_idx = 0
        for b in range(total_bins):
            if num_nd[b] > 0:
                tmp_idx += 1
                nd_to_1d_map[b] = tmp_idx

        n1d_sparse = num_nd[num_nd > 0]
        n1d_err_sparse = num_nd_err[num_nd > 0]

        return nd_to_1d_map, n1d_sparse, n1d_err_sparse

    @staticmethod
    def map_data_to_1d_bins(num_nd, num_nd_err, map_nd1d):
        """
        Convert data from ND to 1D
        """
        for b in range(len(map_nd1d)):
            if num_nd[b] > 0 and map_nd1d[b] == 0:
                print("Not empty in data but empty in MC.")

        n1d_sparse = num_nd[map_nd1d > 0]
        n1d_err_sparse = num_nd_err[map_nd1d > 0]

        return n1d_sparse, n1d_err_sparse

    def map_1d_to_nd(self, unfolded_hist_np, unfolded_cov_np, nbins):
        """
        Convert 1D back to ND
        """
        unfold_nd_hist = np.zeros(nbins)
        unfold_nd_cov = np.zeros([nbins, nbins])
        eff_1d = np.ones(unfolded_hist_np.shape)
        data_mc_scale = 1

        for i in range(nbins):
            if self.true_map[i] <= 0:
                continue
            if unfolded_hist_np[self.true_map[i] - 1] > 0:
                unfold_nd_hist[i] = unfolded_hist_np[self.true_map[i] - 1] / eff_1d[self.true_map[i] - 1]
                for j in range(nbins):
                    if self.true_map[j] > 0 and unfolded_hist_np[self.true_map[j] - 1] > 0:
                        eff_denom = eff_1d[self.true_map[i] - 1] * eff_1d[self.true_map[j] - 1]
                        unfold_nd_cov[i, j] = unfolded_cov_np[self.true_map[i] - 1, self.true_map[j] - 1] / eff_denom
            elif eff_1d[self.true_map[i] - 1] == 0: # FIXME do something better here
                print("Warning: Efficiency is 0 here!")
                raise ValueError
                # unfold_nd_hist[i] = true_nd_hist[i] * data_mc_scale
                # unfold_nd_cov[i, i] = true_cov_nd[i, i] * data_mc_scale * data_mc_scale

        unfold_nd_err = np.sqrt(np.diag(unfold_nd_cov))

        return unfold_nd_hist, unfold_nd_cov, unfold_nd_err

    @staticmethod
    def propagate_unfolded_1d_errors(unfolded_cov, bin_list):

        cov_num_bins = np.prod(bin_list).astype(int)
        var_1d_num_bins = np.sum(bin_list).astype(int)
        jacobian = np.zeros((var_1d_num_bins, cov_num_bins))
        print("Nbins Cov/Var 1D:", cov_num_bins, "/", var_1d_num_bins)

        cov_bin_idx = np.arange(cov_num_bins)
        bin_sum = 0
        for dim_idx, nbins in enumerate(bin_list):
            idx = np.unravel_index(cov_bin_idx, bin_list)[dim_idx] + bin_sum
            jacobian[idx, cov_bin_idx] = 1
            bin_sum += nbins

        # Propagate the errors with the Jacobian
        unfolded_1d_err_cov = (jacobian @ unfolded_cov) @ jacobian.T

        # Remove under/over flow bins (first/last bins idx=0/-1)
        selected_idx = []
        bin_shift = 0
        for nbins in bin_list:
            selected_idx += list(np.arange(1, nbins - 1) + bin_shift)
            bin_shift += nbins

        # Remove x-axis under/over flow bins, then y-axis
        temp_cov = unfolded_1d_err_cov[selected_idx, :]
        no_under_over_flow_cov = temp_cov[:, selected_idx]

        # Return covariance with and without over_under flow bins
        return unfolded_1d_err_cov, no_under_over_flow_cov

    @staticmethod
    def propagate_unfolded_errors_with_incident(unfolded_1d_cov, bin_list):
        """
        This assumes no under/over flow bins
        Call the bins within the range , ROI range/region of interest
        """
        roi_num_bins = sum(bin_list)
        over_flow_bins = 2 * len(bin_list)
        roi_num_bins -= over_flow_bins
        num_bins = bin_list[0] - 2
        jacobian = np.zeros([roi_num_bins, roi_num_bins])

        # Incident covariance
        for ibin in range(num_bins - 1):
            for itmp in range(ibin, num_bins):
                jacobian[ibin, itmp] = 1
            for itmp in range(ibin + num_bins, num_bins + num_bins):
                jacobian[ibin, itmp] = -1

        # End covariance
        for ibin in range(num_bins, num_bins + num_bins):
            jacobian[ibin, ibin] = 1

        # Interacting (signal) covariance
        for ibin in range(num_bins + num_bins, num_bins + num_bins + num_bins - 1):
            jacobian[ibin, ibin] = 1

        unfolded_1d_with_inc_cov = (jacobian @ unfolded_1d_cov) @ jacobian.T

        return unfolded_1d_with_inc_cov

    @staticmethod
    def map_bin_to_variable_space(unfold_nd_hist_np, unfold_nd_cov_np, truth_bin_list):
        """
        Only works for bins since we can sum over the unwanted axes, not covariance
        """
        bin_lens = [len(b) - 1 for b in truth_bin_list]

        # Cast into nth-order tensor containing bins corresponding to the original measured physics variable
        # the first bin is underflow and last is overflow
        unfold_var_hist = unfold_nd_hist_np.reshape(bin_lens)
        # unfold_var_err = np.diag(unfold_nd_cov_np).reshape(bin_lens)

        return unfold_var_hist

    def plot_variables(self, var_list, nbin_list):

        num_vars = len(var_list)
        _, axes = plt.subplots(1, num_vars, figsize=(16, 4))

        for p in range(num_vars):
            ax = axes[p] if type(axes) is list else axes
            ax.hist(var_list[p], bins=nbin_list[p], edgecolor='black', range=[np.min(var_list[p]), np.max(var_list[p])])
            ax.set_title("Variable: " + self.var_names[p])

        plt.legend()
        plt.show()
