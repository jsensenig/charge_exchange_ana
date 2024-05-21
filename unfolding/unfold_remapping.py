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

        # total_bins = sum([nbin_list[d-1]**d for d in range(1, ndim+1)])
        self.true_total_bins = sum([(len(bin_list[d - 1]) - 1) ** d for d in range(1, ndim + 1)])

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

    def remap_data_events(self, data_list, bin_list, nbin_list, ndim):

        self.reco_total_bins = sum([nbin_list[d - 1]**d for d in range(1, ndim + 1)])
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
        bin_cnt = 1
        nnd_bin_array = None
        weight_array = None
        for i, arr_bin in enumerate(zip(corr_var_list, bin_list, evt_weights)):
            arr, bins, weight = arr_bin
            nbins = len(bins) - 1
            # Mapping from measured space (usually energy) to bin space
            n_binned = np.digitize(x=arr, bins=bins, right=False)
            if debug: print("Unique bins:", np.unique(n_binned))
            if debug: print("(n_binned >= 1) & (n_binned <= (", nbins, ")")
            n_binned = n_binned[(n_binned >= 1) & (n_binned <= nbins)]  # ignore under/over flow bins, 0/n+1 respectively
            if debug: print("Unique Bins post under/over -flow cut:", np.unique(n_binned))
            if i == 0:
                nnd_bin_array = n_binned
                continue
            bin_cnt *= nbins
            nnd_bin_array += bin_cnt * n_binned

        print("Max bin:", np.max(nnd_bin_array))
        nnd_bin_array -= 1

        # Create the histogram with event weighting and calculate errors
        evt_weight = np.ones(nnd_bin_array.shape)
        num_nd, _ = np.histogram(nnd_bin_array, bins=total_bins, range=(0, total_bins), weights=evt_weight)
        num_nd_err, _ = np.histogram(nnd_bin_array, bins=total_bins, range=(0, total_bins), weights=evt_weight * evt_weight)
        num_nd_vcov = np.diag(num_nd_err)

        return nnd_bin_array, num_nd, np.sqrt(num_nd_err), num_nd_vcov

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

    def map_1d_to_nd(self, unfolded_hist_np, unfolded_cov_np, true_nd_hist, true_cov_nd, true_nbins_1d_sparse, nbins):
        """
        Convert 1D back to ND
        """
        unfold_nd_hist = np.zeros(nbins)
        unfold_nd_cov = np.zeros([nbins, nbins])
        eff_1d = np.ones(true_nbins_1d_sparse)
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
            elif eff_1d[self.true_map[i] - 1] == 0:
                unfold_nd_hist[i] = true_nd_hist[i] * data_mc_scale
                unfold_nd_cov[i, i] = true_cov_nd[i, i] * data_mc_scale * data_mc_scale

        unfold_nd_err = np.sqrt(np.diag(unfold_nd_cov))

        return unfold_nd_hist, unfold_nd_cov, unfold_nd_err

    def plot_variables(self, var_list, nbin_list):

        num_vars = len(var_list)
        _, axes = plt.subplots(1, num_vars, figsize=(16, 4))

        for p in range(num_vars):
            ax = axes[p] if type(axes) is list else axes
            ax.hist(var_list[p], bins=nbin_list[p], edgecolor='black', range=[np.min(var_list[p]), np.max(var_list[p])])
            ax.set_title("Variable: " + self.var_names[p])

        plt.legend()
        plt.show()
