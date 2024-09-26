import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import ROOT
import RooUnfold
import awkward as ak

from cex_analysis.plot_utils import histogram_constructor, bin_centers_np, bin_width_np
from cex_analysis.true_process import TrueProcess
from unfolding.unfold_remapping import Remapping
from unfolding.unfolding_interface import XSecVariablesBase


class Unfold:

    def __init__(self, config_file, response_file=None, efficiency_file=None):
        self.config = self.configure(config_file=config_file)
        self.show_plots = self.config["show_plots"]
        self.figs_path = self.config["figure_path"]
        self.bayes_niter = self.config["bayes_niter"]
        self.is_training = self.config["is_training"]
        self.single_var = self.config["single_var"]
        self.truth_ndim, self.reco_ndim = 0, 0
        self.truth_hist, self.reco_hist = None, None
        self.truth_nbins_sparse, self.reco_nbins_sparse = 0, 0
        self.true_bin_array, self.reco_bin_array = self.get_bin_config()
        self.compile_cpp_helpers()

        self.remap_evts = Remapping(var_names=self.config["var_names"])
        self.true_process = TrueProcess()

        self.remap_evts.remove_overflow = self.config["remove_overflow"]
        self.remap_evts.subtract_bkgd = self.config["subtract_background"]
        self.calc_bkgds = self.config["calculate_background"]
        self.bkgd_scale_cls = self.config["bkgd_scale_cls"]
        self.bkgd_file_name = self.config["bkgd_file_name"]
        if self.remap_evts.subtract_bkgd:
            self.remap_evts.background_dict = self.load_background()

        self.true_record_var = self.config["true_record_var"]
        self.reco_record_var = self.config["reco_record_var"]
        self.reco_weight_var = self.config["reco_weight_var"]
        self.true_weight_var = self.config["true_weight_var"]
        self.data_weight_var = self.config["data_weight_var"]

        # Get the classes to interface the data to the unfolding
        var_cls = {cls.__name__: cls for cls in XSecVariablesBase.__subclasses__()}
        self.vars = var_cls[self.config["xsec_vars"]](config_file=self.config,
                                                      is_training=self.is_training,
                                                      energy_slices=self.reco_bin_array[0][1:-1])
        self.beam_vars = self.config["xsec_vars"] == "BeamPionVariables"

        self.response = None
        self.efficiency, self.eff_err = None, None
        self.apply_eff_correction = self.config["apply_eff_correction"]
        self.eff_cut = self.config["efficiency_thresh"]

        if not self.is_training:
            self.load_response(response_file=response_file)
        if self.apply_eff_correction:
            self.load_efficiency(efficiency_file=efficiency_file)

    @staticmethod
    def compile_cpp_helpers():

        # Fill a 3D histogram
        ROOT.gInterpreter.ProcessLine("""
            void fill_hist_th3d( int len, double *T_pi0, double *theta_pi0, double *T_piplus, TH3D *hist ) {
                for( int i = 0; i < len; i++ ) hist->Fill( T_pi0[i], theta_pi0[i], T_piplus[i] );
           }
        """)

        # Fill RooUnfold response matrix with events
        ROOT.gInterpreter.ProcessLine("""
            void fill_response_1d( int len, double *reco, double *truth, double *weights, RooUnfoldResponse *resp ) {
                for( int i = 0; i < len; i++ ) {
                    resp->Fill(reco[i], truth[i], weights[i]);
                }
            }
        """)

    def run_unfold(self, event_record=None, data_mask=None, true_var_list=None, reco_var_list=None, true_weight_list=None,
                   reco_weight_list=None,  data_weight_list=None, evt_categories=None, external_vars=False, calculate_eff=False):

        if not external_vars:
            true_var_list, reco_var_list, true_weight_list, reco_weight_list, data_weight_list = (
                self.get_unfold_variables(event_record=event_record, reco_int_mask=data_mask))
        else:
            if self.is_training:
                for i, (tk, rk) in enumerate(zip(self.true_record_var, self.reco_record_var)):
                    self.vars.xsec_vars[tk] = true_var_list[i]
                    self.vars.xsec_vars[rk] = reco_var_list[i]
            else:
                for i, rk in enumerate(self.reco_record_var):
                    self.vars.xsec_vars[rk] = reco_var_list[i]

        if self.is_training:
            print("Start Training")
            true_nd_binned, true_weights, true_nd_hist, true_nd_hist_cov, true_hist_sparse, self.remap_evts.true_map, true_evt_mask \
                = self.remap_evts.remap_events(var_list=true_var_list,
                                               bin_list=self.true_bin_array,
                                               event_weights=true_weight_list)

            reco_nd_binned, reco_weights, reco_nd_hist, reco_nd_hist_cov, reco_hist_sparse, self.remap_evts.reco_map, reco_evt_mask \
                = self.remap_evts.remap_events(var_list=reco_var_list,
                                               bin_list=self.true_bin_array,
                                               event_weights=reco_weight_list)

            # Only used for efficiency calculation
            # _, _, signal_nd_hist, _, signal_hist_sparse, _, _ = self.remap_evts.remap_events(var_list=signal_var_list,
            #                                                                               bin_list=self.true_bin_array,
            #                                                                               event_weights=signal_weight_list,
            #                                                                               is_true_reco=False, is_data=False)
            # sel_signal = [var[selected_signal_mask] for var in signal_var_list]
            # sel_weight = signal_weight_list[selected_signal_mask]
            # _, _, selected_signal_nd_hist, _, _, _, _ = self.remap_evts.remap_events(var_list=sel_signal,
            #                                                                       bin_list=self.true_bin_array,
            #                                                                       event_weights=sel_weight,
            #                                                                       is_true_reco=False, is_data=False)
            #
            # self.efficiency, self.eff_err = self.remap_evts.calculate_efficiency(full_signal=signal_nd_hist,
            #                                                                      full_selected=selected_signal_nd_hist)
            #
            if self.calc_bkgds:
                bkgd_dict = self.vars.get_backgrounds(pts=reco_var_list, process=evt_categories, bin_array=self.true_bin_array,
                                                      weights=reco_weight_list, scale_cls=self.bkgd_scale_cls)
                self.save_backgrounds(bkgd_dict=bkgd_dict)

            self.truth_nbins_sparse, self.reco_nbins_sparse = len(true_hist_sparse), len(reco_hist_sparse)

            response_mask = true_evt_mask & reco_evt_mask
            self.create_response_matrix(reco_events=self.remap_evts.reco_map[reco_nd_binned[response_mask]].astype('d'),
                                        true_events=self.remap_evts.true_map[true_nd_binned[response_mask]].astype('d'),
                                        reco_weights=reco_weights[response_mask].astype('d'))

        if self.show_plots:
            self.plot_response_matrix(response_matrix=self.response)

        # Set data
        _, _, _, _, data_hist_sparse, _, _ = self.remap_evts.remap_events(var_list=reco_var_list,
                                                                       bin_list=self.true_bin_array,
                                                                       event_weights=data_weight_list,
                                                                       is_true_reco=False, is_data=True)

        self.fill_data_hist(sparse_data_hist=data_hist_sparse)

        # Unfold data
        unfolded_data_hist, unfolded_data_cov = self.unfold_bayes()

        # Convert to numpy
        self.truth_hist = ROOT.TH1D("truth", "Truth", self.truth_nbins_sparse, 0, self.truth_nbins_sparse)
        unfolded_data_hist_np, unfolded_data_cov_np = self.root_to_numpy(unfolded_hist=unfolded_data_hist,
                                                                         cov_matrix=unfolded_data_cov)

        # Plot sparse unfolded results
        if self.show_plots and self.is_training:
            self.truth_hist = ROOT.TH1D("truth", "Truth", int(len(unfolded_data_hist_np)), 0, float(len(unfolded_data_hist_np)))
            self.plot_unfolded_results(unfolded_data_hist_np=unfolded_data_hist_np, unfolded_cov_np=unfolded_data_cov_np,
                                       true_hist_np=true_hist_sparse)

        # Map 1D back to ND space
        total_bins = self.remap_evts.true_total_bins #if self.is_training else self.remap_evts.reco_total_bins
        raw_unfold_nd_hist_np, raw_unfold_nd_cov_np, _ = self.remap_evts.map_1d_to_nd(unfolded_hist_np=unfolded_data_hist_np,
                                                                                    unfolded_cov_np=unfolded_data_cov_np,
                                                                                    nbins=total_bins)

        # Efficiency correct unfolded data after its mapped back to non-sparse form
        if calculate_eff:
            return raw_unfold_nd_hist_np, raw_unfold_nd_cov_np

        if self.apply_eff_correction:
            unfold_nd_hist_np, unfold_nd_cov_np = self.remap_evts.correct_for_efficiency(unfolded_data=raw_unfold_nd_hist_np,
                                                                                         unfolded_data_cov=raw_unfold_nd_cov_np,
                                                                                         efficiency=self.efficiency,
                                                                                         eff_cut=self.eff_cut)
        else:
            unfold_nd_hist_np, unfold_nd_cov_np = raw_unfold_nd_hist_np, raw_unfold_nd_cov_np

        if self.show_plots and self.is_training:
            self.truth_hist = ROOT.TH1D("truth", "Truth", int(len(unfold_nd_hist_np)), 0, float(len(unfold_nd_hist_np)))
            self.plot_unfolded_results(unfolded_data_hist_np=unfold_nd_hist_np, unfolded_cov_np=unfold_nd_cov_np,
                                       true_hist_np=true_nd_hist)

        unfolded_corr_np = self.correlation_from_covariance(unfolded_cov=unfold_nd_cov_np)

        unfold_var_hist = self.remap_evts.map_bin_to_variable_space(unfold_nd_hist_np=unfold_nd_hist_np,
                                                                    unfold_nd_cov_np=unfold_nd_cov_np,
                                                                    truth_bin_list=self.true_bin_array)

        # Errors, one with the unfolded variables and one with the incident histogram
        #bin_lens = np.ma.count(self.reco_bin_array, axis=1) - 1 
        bin_lens = [len(b) - 1 for b in self.reco_bin_array]
        unfolded_1d_err_cov, no_under_over_flow_cov = self.remap_evts.propagate_unfolded_1d_errors(unfolded_cov=unfold_nd_cov_np,
                                                                                                   bin_list=bin_lens)
        unfolded_1d_with_inc_cov = None
        if self.beam_vars and not self.single_var:
            unfolded_1d_with_inc_cov = self.remap_evts.propagate_unfolded_errors_with_incident(unfolded_1d_cov=no_under_over_flow_cov,
                                                                                               bin_list=bin_lens)

        # FIXME disable for now
        if self.show_plots and self.is_training and False:
            unfold_var_err = np.ones_like(unfold_var_hist) * np.sqrt(unfold_var_hist)
            bin_lens = [len(b) - 1 for b in self.true_bin_array]
            self.plot_unfolded_results_var_space(unfold_var_hist=unfold_var_hist, unfold_var_err=unfold_var_err,
                                                 true_var_list=true_var_list, bin_lens=bin_lens,
                                                 var_label_list=self.config["var_names"])

        return unfold_nd_hist_np, unfold_nd_cov_np, unfolded_corr_np, unfold_var_hist, unfolded_1d_err_cov, no_under_over_flow_cov, unfolded_1d_with_inc_cov, 

    def fill_data_hist(self, sparse_data_hist):
        if self.reco_hist is not None:
            self.reco_hist.Delete()

        self.reco_hist = ROOT.TH1D("data", "Data", self.reco_nbins_sparse, 0, self.reco_nbins_sparse)
        _ = [self.reco_hist.SetBinContent(i + 1, sparse_data_hist[i]) for i in range(self.reco_nbins_sparse)]

    @staticmethod
    def get_weights(var_dict, weight_list, mask, nevts):
        if len(weight_list) < 1:
            return np.ones(nevts)

        return np.prod([var_dict[w][mask] for w in weight_list], axis=0)

    def get_unfold_variables(self, event_record, reco_int_mask):

        # Get the variables of interest
        var_dict = self.vars.get_xsec_variable(event_record=event_record, reco_int_mask=reco_int_mask)
        print("Loaded variables:", list(var_dict))
        is_beam_var = self.config["xsec_vars"] == "BeamPionVariables"

        reco_var_list = [var_dict[var] for var in self.reco_record_var]
        reco_mask = var_dict["full_len_reco_mask"] if is_beam_var else np.ones(len(reco_var_list[0])).astype(bool)

        true_var_list = None
        if self.is_training:
            true_var_list = [var_dict[var] for var in self.true_record_var]
            true_mask = var_dict["full_len_true_mask"] if is_beam_var else np.ones(len(true_var_list[0])).astype(bool)
            true_weight_list = self.get_weights(var_dict=var_dict, weight_list=self.true_weight_var,
                                                mask=true_mask, nevts=len(true_var_list[0]))
            reco_weight_list = self.get_weights(var_dict=var_dict, weight_list=self.reco_weight_var,
                                                mask=reco_mask, nevts=len(reco_var_list[0]))
        else:
            true_weight_list = np.ones(len(reco_var_list[0]))
            reco_weight_list = np.ones(len(reco_var_list[0]))
            
        data_weight_list = self.get_weights(var_dict=var_dict, weight_list=self.data_weight_var,
                                            mask=reco_mask, nevts=len(reco_var_list[0]))            

        return true_var_list, reco_var_list, true_weight_list, reco_weight_list, data_weight_list

    def create_response_matrix(self, reco_events, true_events, reco_weights):

        self.response = RooUnfold.RooUnfoldResponse(self.reco_nbins_sparse, 1, self.reco_nbins_sparse+1,
                                                    self.truth_nbins_sparse, 1, self.truth_nbins_sparse+1)

        ROOT.fill_response_1d(len(reco_events), reco_events, true_events, reco_weights, self.response)

    def unfold_bayes(self):

        unfold = RooUnfold.RooUnfoldBayes(self.response, self.reco_hist, self.bayes_niter)

        # Get statistical uncorrelated bin errors
        # since setting the diagonal of the Cov matrix it should be sigma^2
        data_error = np.diag([self.reco_hist.GetBinError(b+1)**2 for b in range(self.reco_hist.GetNbinsX())])
        data_diag_error = ROOT.TMatrix(data_error.shape[0], data_error.shape[1])
        for i in range(data_error.shape[0]):
            for j in range(data_error.shape[1]):
                data_diag_error[i, j] = data_error[i, j]

        unfold.SetMeasuredCov(data_diag_error)

        unfolded_data_hist = unfold.Hunfold() # returns Hist
        unfolded_data_cov = unfold.Eunfold()  # returns TVectorD
        print("unfolded_data_hist.GetNbinsX()", unfolded_data_hist.GetNbinsX())
        print("NRows:", unfolded_data_cov.GetNrows(), " NCols:", unfolded_data_cov.GetNcols())
        return unfolded_data_hist, unfolded_data_cov

    def create_hists_numpy(self, data_events, reco_events=None, true_events=None):
        if self.is_training:
            self.truth_hist, _ = np.histogramdd(true_events, self.true_bin_array)
            self.reco_hist, _ = np.histogramdd(reco_events, self.true_bin_array)
        bin_array = self.true_bin_array if self.is_training else self.reco_bin_array
        self.reco_hist, _ = np.histogramdd(data_events, bin_array)

    def create_hists(self, data_events, reco_events, true_events=None):
        """
        The events should have shape (nsample, ndim)
        """
        if self.is_training:
            self.truth_hist = histogram_constructor("truth", bin_arrays=self.true_bin_array, ndim=self.truth_ndim)
        bin_array = self.reco_bin_array
        self.reco_hist = histogram_constructor("data", bin_arrays=bin_array, ndim=self.reco_ndim)

        if self.is_training:
            self.truth_hist.FillN(len(true_events), true_events[:, 0], ROOT.nullptr)
            self.reco_hist.FillN(len(reco_events), reco_events[:, 0], ROOT.nullptr)
        self.reco_hist.FillN(len(data_events), data_events[:, 0], ROOT.nullptr)

    @staticmethod
    def correlation_from_covariance(unfolded_cov):
        v = np.sqrt(np.diag(unfolded_cov))
        unfolded_corr = unfolded_cov / np.outer(v, v)
        unfolded_corr[unfolded_cov == 0] = 0
        return unfolded_corr

    @staticmethod
    def root_to_numpy(unfolded_hist, cov_matrix):
        cov_matrix_np = np.empty(shape=(cov_matrix.GetNrows(), cov_matrix.GetNcols()))
        unfolded_data_hist_np = np.empty(shape=(unfolded_hist.GetNbinsX()))

        for i in range(cov_matrix.GetNrows()):
            unfolded_data_hist_np[i] = unfolded_hist.GetBinContent(i + 1)
            for j in range(cov_matrix.GetNcols()):
                cov_matrix_np[i, j] = cov_matrix[i, j]

        return unfolded_data_hist_np, cov_matrix_np

    def plot_response_matrix(self, response_matrix):
        response_matrix_np = np.empty(shape=(response_matrix.Hresponse().GetNbinsX(), response_matrix.Hresponse().GetNbinsY()))
        for i in range(response_matrix.Hresponse().GetNbinsX()):
            for j in range(response_matrix.Hresponse().GetNbinsY()):
                response_matrix_np[i, j] = response_matrix.Hresponse().GetBinContent(i + 1, j + 1)

        norm_factor = 1. / response_matrix_np.sum(axis=0)

        # Transpose response so we have truth vs reco
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        f = ax1.imshow(response_matrix_np.T, origin='lower', cmap=plt.cm.jet, norm=LogNorm())
        ax1.set_title("Response Matrix", fontsize=14)
        ax1.set_xlabel("Reco Bins", fontsize=12)
        ax1.set_ylabel("True Bins", fontsize=12)
        plt.colorbar(f)
        f = ax2.imshow((response_matrix_np * norm_factor).T, origin='lower', cmap=plt.cm.jet, norm=LogNorm())
        ax2.set_title("Response Matrix: $P(Reco | True)$", fontsize=14)
        ax2.set_xlabel("Reco Bins", fontsize=12)
        ax2.set_ylabel("True Bins", fontsize=12)
        plt.colorbar(f)
        plt.savefig(self.figs_path + "/response_matrix.pdf")
        plt.show()

    def plot_unfolded_results(self, unfolded_data_hist_np, unfolded_cov_np, true_hist_np):

        corr_matrix_np = self.correlation_from_covariance(unfolded_cov=unfolded_cov_np)

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        f = ax1.imshow(corr_matrix_np, origin='lower', cmap=plt.cm.RdBu_r, vmin=-1, vmax=1)
        plt.colorbar(f)

        # Get the truth histogram edges
        true_bin_edges = np.asarray([self.truth_hist.GetBinLowEdge(b + 1) for b in range(self.truth_hist.GetNbinsX()+1)])
        true_bin_centers = (true_bin_edges[1:] + true_bin_edges[:-1]) / 2.

        _,bx,_=ax2.hist(true_bin_centers, bins=true_bin_edges, weights=true_hist_np, edgecolor='black', color='indianred',
                        alpha=0.9, label='Truth')
        ax2.errorbar(true_bin_centers, unfolded_data_hist_np, np.sqrt(np.diag(unfolded_cov_np)), bin_width_np(bx)/2,
                     capsize=2, marker='s', markersize=3, color='black', linestyle='None', label='Unfolded Data')
        ax2.set_ylim(bottom=0)
        plt.legend()
        plt.savefig(self.figs_path + "/data_unfolded_hist_cov.pdf")
        plt.show()

    def plot_unfolded_results_var_space(self, unfold_var_hist, unfold_var_err, true_var_list, bin_lens, var_label_list=None):

        if var_label_list is None:
            var_label_list = ["Var_0", "Var_1"]

        if self.truth_ndim == 2:
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            binsx, binsy = bin_lens
            min_max_var0 = (np.min(self.true_bin_array[0]), np.max(self.true_bin_array[0]))
            min_max_var1 = (np.min(self.true_bin_array[1]), np.max(self.true_bin_array[1]))
            h, bx, by, f = ax1.hist2d(true_var_list[0], true_var_list[1], bins=bin_lens, range=[min_max_var0, min_max_var1])
            ax1.set_title('True Distribution')
            plt.colorbar(f)
            f = ax2.imshow(unfold_var_hist, origin='lower')
            ax2.set_title('Unfolded Distribution')
            plt.colorbar(f)
            f = ax3.imshow(np.sqrt(unfold_var_err), origin='lower')
            ax3.set_title('Unfolded Errors')
            plt.colorbar(f)
            plt.show()

            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            y_err = unfold_var_hist.sum(axis=0) / np.sqrt(unfold_var_err.sum(axis=0))
            ax1.hist(true_var_list[0], bins=binsx, range=min_max_var0, edgecolor='black', color='indianred', alpha=0.9, label='True')
            ax1.errorbar(bin_centers_np(bx), unfold_var_hist.sum(axis=0), y_err, bin_width_np(bx), capsize=2, marker='s',
                         markersize=3, color='black', linestyle='None', label='Unfolded')
            ax1.set_xlabel(var_label_list[0], fontsize=12)
            ax1.set_xticks(np.arange(200, 1900, 200))
            ax1.set_xlim(min_max_var0)
            ax1.set_ylim(bottom=0)
            ax1.legend()

            y_err = unfold_var_hist.sum(axis=1) / np.sqrt(unfold_var_err.sum(axis=1))
            ax2.hist(true_var_list[1], bins=binsy, range=min_max_var1, edgecolor='black', color='indianred', alpha=0.9, label='True')
            ax2.errorbar(bin_centers_np(by), unfold_var_hist.sum(axis=1), y_err, bin_width_np(by), capsize=2, marker='s',
                         markersize=3, color='black', linestyle='None', label='Unfolded')
            ax2.set_xlabel(var_label_list[1], fontsize=12)
            ax2.set_xlim(min_max_var1)
            ax2.set_ylim(bottom=0)
            ax2.legend()
            plt.show()
        elif self.truth_ndim == 1:
            plt.figure(figsize=(8, 5))
            binsx = bin_lens[0]
            min_max_var0 = (np.min(self.true_bin_array[0]), np.max(self.true_bin_array[0]))
            y_err = unfold_var_hist / np.sqrt(unfold_var_err)
            _, bx, _ = plt.hist(true_var_list[0], bins=binsx, range=min_max_var0, edgecolor='black', color='indianred',
                                alpha=0.9, label='True')
            plt.errorbar(bin_centers_np(bx), unfold_var_hist, y_err, bin_width_np(bx)/2., capsize=2, marker='s',
                         markersize=3, color='black', linestyle='None', label='Unfolded')
            plt.xlim(min_max_var0)
            plt.ylim(bottom=0)
            plt.xlabel(var_label_list[0], fontsize=12)
            plt.legend()
            plt.show()
        else:
            print(">2 dimensions not supported. Ndim=", self.truth_ndim)

    def get_bin_config(self):
        if self.config["use_bin_array"]:
            true_array = np.asarray(self.config["truth_bins"], dtype=np.dtype('d'))
            reco_array = np.asarray(self.config["reco_bins"], dtype=np.dtype('d'))
            self.truth_ndim = len(true_array)
            self.reco_ndim = len(reco_array)
        else: # 0th bins is underflow and n+1th bin is overflow
            nbins, bin_range = self.config["truth_bins"]["nbins"], self.config["truth_bins"]["limits"]
            self.truth_ndim = len(nbins)
            true_array = [np.linspace(limits[0], limits[1], bin + 1) for bin, limits in zip(nbins, bin_range)]
            true_array = [np.concatenate(([-1.e6], tarr, [limits[1] + 1.e6])) for tarr, limits in zip(true_array, bin_range)]
            nbins, bin_range = self.config["reco_bins"]["nbins"], self.config["reco_bins"]["limits"]
            self.reco_ndim = len(nbins)
            reco_array = [np.linspace(limits[0], limits[1], bin + 1) for bin, limits in zip(nbins, bin_range)]
            reco_array = [np.concatenate(([-1.e6], tarr, [limits[1] + 1.e6])) for tarr, limits in zip(reco_array, bin_range)]

        return true_array, reco_array

    def load_response(self, response_file):

        if response_file is None:
            response_file = self.config["response_file"]

        with open(response_file, 'rb') as f:
            unfold_param_dict = pickle.load(f)

        self.response = unfold_param_dict["response"]
        self.remap_evts.true_map = unfold_param_dict["true_bin_map"]
        self.remap_evts.reco_map = unfold_param_dict["reco_bin_map"]
        #self.true_bin_array = unfold_param_dict["true_bin_array"]
        #self.reco_bin_array = unfold_param_dict["reco_bin_array"]
        self.reco_nbins_sparse = unfold_param_dict["reco_nbins_sparse"]
        
        
        print("Loaded unfold param file:", response_file)

    def save_response(self, file_name):
        unfold_param_dict = {"response": self.response,
                             "true_bin_map": self.remap_evts.true_map,
                             "reco_bin_map": self.remap_evts.reco_map,
                             "true_bin_array": self.true_bin_array,
                             "reco_bin_array": self.reco_bin_array,
                             "reco_nbins_sparse": self.reco_nbins_sparse,
                             "efficiency": self.efficiency,
                             "efficiency_err": self.eff_err}

        with open(file_name, 'wb') as f:
            pickle.dump(unfold_param_dict, f)
        print("Wrote unfold param file:", file_name)

    def load_efficiency(self, efficiency_file):

        if efficiency_file is None:                               
            efficiency_file = self.config["efficiency_file"]
                                                         
        with open(efficiency_file, 'rb') as f:
            unfold_eff_dict = pickle.load(f)

        self.efficiency = unfold_eff_dict["efficiency"]
        self.eff_err = unfold_eff_dict["efficiency_err"]

        print("Loaded efficiency file:", efficiency_file)

    def save_efficiency(self, file_name, unfold_eff_dict=None):
        """
        Save the efficiency and its errors
        """
        if unfold_eff_dict is None:
            unfold_eff_dict = {"efficiency": self.efficiency,
                               "efficiency_err": self.eff_err}
                                                                          
        with open(file_name, 'wb') as f:
            pickle.dump(unfold_eff_dict, f)
        print("Wrote efficiency file:", file_name)

    def save_backgrounds(self, bkgd_dict=None):
        """
        Save the background and its (errors)
        """
        with open(self.bkgd_file_name, 'wb') as f:
            pickle.dump(bkgd_dict, f)
        print("Wrote background file:", self.bkgd_file_name)

    def load_background(self):

        background_file = self.config["background_file"]

        with open(background_file, 'rb') as f:
            background_dict = pickle.load(f)

        return background_dict

    @staticmethod
    def configure(config_file):
        """
        Implement the configuration for the concrete cut class here.
        """
        with open(config_file, "r") as cfg:
            config = json.load(cfg)

        return config

