import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import ROOT
import RooUnfold

from cex_analysis.plot_utils import histogram_constructor


class Unfold:

    def __init__(self, config_file):
        self.config = self.configure(config_file=config_file)
        self.bayes_niter = self.config["bayes_niter"]
        self.is_training = self.config["is_training"]
        self.train_ndim, self.data_ndim = 0, 0
        self.truth_hist, self.reco_hist, self.data_hist = None, None, None
        self.train_bin_array, self.data_bin_array = self.get_bin_config()
        self.compile_cpp_helpers()

        self.response = None
        if not self.is_training:
            self.load_response()

    @staticmethod
    def compile_cpp_helpers():

        # Fill a histogram with strings
        ROOT.gInterpreter.ProcessLine("""
            void fill_hist_th3d( int len, double *T_pi0, double *theta_pi0, double *T_piplus, TH3D *hist ) {
                for( int i = 0; i < len; i++ ) hist->Fill( T_pi0[i], theta_pi0[i], T_piplus[i] );
           }
        """)

        ROOT.gInterpreter.ProcessLine("""
            void fill_response_1d( int len, double *reco, double *true, double *weights, RooUnfoldResponse *resp ) {
                for( int i = 0; i < len; i++ ) {
                    resp.Fill(reco[i], true[i], weights[i]);
                }
            }
        """)

    def get_bin_config(self):
        if self.config["use_bin_array"]:
            train_array = self.config["train_bins"]
            data_array = self.config["data_bins"]
        else:
            nbins, bin_range = self.config["train_bins"]["nbins"], self.config["train_bins"]["limits"]
            self.train_ndim = len(nbins)
            train_array = [np.linspace(limits[0], limits[1], bin + 1) for bin, limits in zip(nbins, bin_range)]
            nbins, bin_range = self.config["data_bins"]["nbins"], self.config["data_bins"]["limits"]
            self.data_ndim = len(nbins)
            data_array = [np.linspace(limits[0], limits[1], bin + 1) for bin, limits in zip(nbins, bin_range)]

        return train_array, data_array

    def run_unfold(self, event_record, data_mask, train_mask):

        if self.is_training:
            self.create_hists_numpy(data_events=event_record[data_mask], reco_events=event_record[train_mask],
                                    true_events=event_record[train_mask])
        else:
            self.create_hists_numpy(data_events=event_record[data_mask], reco_events=None, true_events=None)

        if self.is_training:
            self.create_response_matrix()

        if self.is_training:
            self.create_response_matrix(reco_events=event_record[train_mask], true_events=event_record[train_mask])

        unfolded_data_hist, unfolded_data_cov = self.unfold_bayes()

    def create_response_matrix(self, reco_events, true_events):

        self.response = RooUnfold.RooUnfoldResponse(len(self.train_bin_array)-1, self.train_bin_array[0],
                                               self.train_bin_array[-1])
        ROOT.fill_response_1d(len(reco_events), reco_events, true_events, np.ones_like(reco_events), self.response)

    def unfold_bayes(self):

        unfold = RooUnfold.RooUnfoldBayes(self.response, self.data_hist, self.bayes_niter)

        # Get statistical uncorrelated bin errors
        data_error = np.diag([self.data_hist.GetBinError(b) for b in range(self.data_hist.GetNbinsX)])
        unfold.SetMeasuredCov(data_error)

        unfolded_data_hist = unfold.Hunfold() # returns Hist
        unfolded_data_cov = unfold.Eunfold()  # returns TVectorD

        return unfolded_data_hist, unfolded_data_cov

    def create_hists_numpy(self, data_events, reco_events=None, true_events=None):
        if self.is_training:
            self.truth_hist, _ = np.histogramdd(true_events, self.train_bin_array)
            self.reco_hist, _ = np.histogramdd(reco_events, self.train_bin_array)
        bin_array = self.train_bin_array if self.is_training else self.data_bin_array
        self.data_hist, _ = np.histogramdd(data_events, bin_array)

    def create_hists(self, data_events, reco_events, true_events=None):
        """
        The *events should have shape (nsample, ndim)
        """
        if self.is_training:
            self.truth_hist = histogram_constructor("truth", bin_arrays=self.train_bin_array, ndim=self.train_ndim)
            self.reco_hist = histogram_constructor("reco", bin_arrays=self.train_bin_array, ndim=self.train_ndim)
        bin_array = self.train_bin_array if self.is_training else self.data_bin_array
        self.data_hist = histogram_constructor("data", bin_arrays=bin_array, ndim=self.data_ndim)

        if self.train_ndim == 3:
            if self.is_training:
                ROOT.fill_hist_th3d(len(true_events), true_events[:, 0], true_events[:, 1], true_events[:, 2], self.truth_hist)
                ROOT.fill_hist_th3d(len(reco_events), reco_events[:, 0], reco_events[:, 1], reco_events[:, 2], self.reco_hist)
            ROOT.fill_hist_th3d(len(data_events), data_events[:, 0], data_events[:, 1], data_events[:, 2], self.data_hist)
        elif self.train_ndim == 2:
            if self.is_training:
                self.truth_hist.FillN(len(true_events), true_events[:, 0], true_events[:, 1], ROOT.nullptr)
                self.reco_hist.FillN(len(reco_events), reco_events[:, 0], reco_events[:, 1], ROOT.nullptr)
            self.data_hist.FillN(len(data_events), data_events[:, 0], data_events[:, 1], ROOT.nullptr)
        elif self.train_ndim == 1:
            if self.is_training:
                self.truth_hist.FillN(len(true_events), true_events, ROOT.nullptr)
                self.reco_hist.FillN(len(reco_events), reco_events, ROOT.nullptr)
            self.data_hist.FillN(len(data_events), data_events, ROOT.nullptr)
        else:
            print("Unsupported dim:", self.train_ndim)
            raise ValueError

    @staticmethod
    def plot_response_matrix(self, response_matrix):
        response_matrix_np = np.empty(shape=(response_matrix.Hresponse.GetNbinsX, response_matrix.Hresponse.GetNbinsY))
        for i in range(response_matrix.Hresponse.GetNbinsX):
            for j in range(response_matrix.Hresponse.GetNbinsY):
                response_matrix_np[i, j] = response_matrix.Hresponse().GetBinContent(i + 1, j + 1)

        plt.imshow(response_matrix_np, origin='lower', cmap=plt.cm.jet)
        plt.colorbar()
        plt.savefig("figs/response_matrix.pdf")
        plt.show()

    @staticmethod
    def plot_unfolded_results(self, meas_hist, cov_matrix):
        cov_matrix_np = np.empty(shape=(cov_matrix.GetNrows(), cov_matrix.Hresponse.GetNcols()))
        hist_np = np.empty(shape=(meas_hist.GetNbinsX))

        for i in range(cov_matrix.GetNrows()):
            hist_np[i] = meas_hist.GetBinContent(i + 1)
            for j in range(cov_matrix.Hresponse.GetNcols()):
                cov_matrix_np[i, j] = cov_matrix[i, j]

        plt.imshow(cov_matrix_np, origin='lower', cmap=plt.cm.RdBu_r)
        plt.colorbar()
        plt.savefig("figs/data_unfolded_cov_matrix.pdf")
        plt.show()

        x = np.arange(len(hist_np))
        xerr = (x[1] - x[0]) / 2.
        yerr = np.sqrt(hist_np)

        plt.errorbar(x, hist_np, yerr, xerr, marker='.', color='black', linestyle='None')
        plt.savefig("figs/data_unfolded_hist.pdf")
        plt.show()

    def load_response(self):
        with open(self.config["response_file"], 'rb') as f:
            self.response = pickle.load(f)

    @staticmethod
    def save_response(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.response, f)

    @staticmethod
    def configure(config_file):
        """
        Implement the configuration for the concrete cut class here.
        """
        with open(config_file, "r") as cfg:
            config = json.load(cfg)

        return config

