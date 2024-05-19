import numpy as np


class Remapping:
    """
    Class heavily inspired by Yinrui Liu's implementation
    https://github.com/Yinrui-Liu/hadron-Ar_XS/blob/main/hadron-Ar_XS.ipynb
    I generalized the concept from 3d to Nd with more vectorized operations
    """
    def __init__(self):
        self.true_nd_to_1d_map = None
        self.meas_nd_to_1d_map = None

    def remap_training_events(self, true_list, reco_list, bin_list, nbin_list, ndim):

        total_bins = sum([nbin_list[d-1]**d for d in range(1, ndim+1)])

        true_event_weights = [np.ones(arr.shape) for arr in true_list]
        reco_event_weights = [np.ones(arr.shape) for arr in reco_list]

        true_num_nd_bin, true_num_nd, true_num_nd_err, true_num_nd_cov = self.map_meas_to_bin_space(corr_var_list=true_list,
                                                                                               nbin_list=bin_list,
                                                                                               total_bins=total_bins,
                                                                                               evt_weights=true_event_weights)

        reco_num_nd_bin, reco_num_nd, reco_num_nd_err, reco_num_nd_cov = self.map_meas_to_bin_space(corr_var_list=reco_list,
                                                                                               nbin_list=bin_list,
                                                                                               total_bins=total_bins,
                                                                                               evt_weights=reco_event_weights)

        print("Total Bins:", total_bins)
        print("True number Nd:", np.unique(true_num_nd).shape)
        print("Meas number Nd:", np.unique(reco_num_nd).shape)

        # Create map between 3D and 1D
        self.true_nd_to_1d_map, true_n1d_sparse, true_n1d_err_sparse = self.map_nd_to_1d(num_nd=true_num_nd,
                                                                                         num_nd_err=true_num_nd_err,
                                                                                         total_bins=total_bins)
        self.meas_nd_to_1d_map, meas_n1d_sparse, meas_n1d_err_sparse = self.map_nd_to_1d(num_nd=reco_num_nd,
                                                                                         num_nd_err=reco_num_nd_err,
                                                                                         total_bins=total_bins)

        print("True Map:", np.count_nonzero(self.true_nd_to_1d_map))
        print("Meas Map:", np.count_nonzero(self.meas_nd_to_1d_map))

        return true_num_nd_bin, reco_num_nd_bin #, true_nd_to_1d_map, meas_nd_to_1d_map

    def remap_data_events(self, data_list, reco_3d_to_1d_map, bin_list, nbin_list, ndim):

        total_bins = sum([nbin_list[d-1]**d for d in range(1, ndim+1)])
        data_event_weights = [np.ones(arr.shape) for arr in data_list]

        data_num_nd_bin, data_num_nd, data_num_nd_err, data_num_nd_cov = self.map_meas_to_bin_space(corr_var_list=data_list,
                                                                                               nbin_list=bin_list,
                                                                                               total_bins=total_bins,
                                                                                               evt_weights=data_event_weights)

        # Data mapping to 1D
        data_n1d_sparse, data_n1d_err_sparse = self.map_data_to_1d_bins(num_nd=data_num_nd, num_nd_err=data_num_nd_err,
                                                                   map_nd1d=reco_3d_to_1d_map)
        print("Sparse Data nbins:", len(data_n1d_sparse))

        return data_n1d_sparse, data_n1d_err_sparse

    @staticmethod
    def map_meas_to_bin_space(self, corr_var_list, nbin_list, total_bins, evt_weights, debug=False):
        """
        Convert list of correlated varibles to single ndim variable
        """
        bin_cnt = 1
        nnd_bin_array = None
        weight_array = None
        for i, arr_bin in enumerate(zip(corr_var_list, nbin_list, evt_weights)):
            arr, bins, weight = arr_bin
            nbins = len(bins) - 1
            # Mapping from measured space (usually energy) to bin space
            n_binned = np.digitize(x=arr, bins=bins, right=False)
            if debug: print("Unique bins:", np.unique(n_binned))
            if debug: print("(n_binned >= 1) & (n_binned <= (", total_bins, ")")
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
    def to_1d_map(self, n_nd1d, nbins):
        """
        Convert ND to 1D
        """
        map_1d = np.zeros(nbins, dtype=np.int32)
        tmp_idx = 0
        for b in range(nbins):
            if n_nd1d[b] > 0:
                tmp_idx += 1
                map_1d[b] = tmp_idx

        return map_1d

    def map_nd_to_1d(self, num_nd, num_nd_err, total_bins):
        """
        Convert truth and reco 3D to 1D
        """
        # Create the maps for true and measured
        nd_to_1d_map = self.to_1d_map(n_nd1d=num_nd, nbins=total_bins)

        n1d_sparse = num_nd[num_nd > 0]
        n1d_err_sparse = num_nd_err[num_nd > 0]

        return nd_to_1d_map, n1d_sparse, n1d_err_sparse

    @staticmethod
    def map_data_to_1d_bins(self, num_nd, num_nd_err, map_nd1d):
        """
        Convert data from ND to 1D
        """
        for b in range(len(map_nd1d)):
            if num_nd[b] > 0 and map_nd1d[b] == 0:
                print("Not empty in data but empty in MC.")

        n1d_sparse = num_nd[map_nd1d > 0]
        n1d_err_sparse = num_nd_err[map_nd1d > 0]

        return n1d_sparse, n1d_err_sparse

    def plot(self):
        pass
