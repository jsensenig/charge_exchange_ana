import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import h5py
import cex_analysis.plot_utils as utils

plt.style.use(hep.style.ATLASAlt)


def create_plots(hists, data_hists):

    for idx in range(len(hists['hist_name'])):
        hname = hists['hist_name'][idx].decode('UTF-8')
        print(idx, " - ", hname)
        htype = hists['hist_type'][idx]
        bin_center = (hists['hist_bins'][idx][0:-1] + hists['hist_bins'][idx][1:]) / 2
        if htype == b'stack':
            mc_scale = np.sum(data_hists[hname]) / sum([h.sum() for h in hists['hist_stack'][idx] if len(h) > 0])
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,7), sharex=True, gridspec_kw={'height_ratios': [6, 1]})
            legend = [utils.code2string[int(leg.rsplit()[0])] + " " + b' '.join(leg.rsplit()[2:]).decode('UTF-8') for leg in hists['hist_legend'][idx]]
            stack_type = 'process' if int(hists['hist_legend'][idx][0].rsplit()[0]) > 5000 else 'pdg'
            hcol = [utils.colors[int(leg.rsplit()[0])] for leg in hists['hist_legend'][idx]]
            hlist = [h * mc_scale for h in hists['hist_stack'][idx] if len(h) > 0]
            hep.histplot(hlist, hists['hist_bins'][idx], stack=True, histtype='fill', ax=ax1, color=hcol)
            ax1.errorbar(bin_center, data_hists[hname], np.sqrt(data_hists[hname]), capsize=2, marker='s', markersize=3, color='black', linestyle='None')
            ax1.set_title(hname)
            ax1.legend(legend + ['Data'])
            ax1.set_ylabel(hists['hist_ylabel'][idx].decode('UTF-8'))
            ratio = data_hists[hname] / np.sum([h * mc_scale for h in hists['hist_stack'][idx] if len(h) > 0], axis=0)
            ratio[np.isinf(abs(ratio))] = np.nan
            ratio[(ratio >= 2) | (ratio <= 0)] = np.nan
            ratio = np.ones(len(ratio)) * 0. if np.all(np.isnan(ratio)) else ratio
            ax2.errorbar(bin_center, ratio, ratio * 0.1, capsize=2, marker='s', markersize=3, color='black', linestyle='None')
            plt.grid()
            ax2.set_xlabel(hists['hist_xlabel'][idx].decode('UTF-8'))
            ax2.set_ylabel('Data/MC', fontsize=14)
            ax2.set_ylim(0, 2)
            plt.savefig('figs/' + 'stack_' + stack_type + '_' + hname + '.pdf')
        elif htype == b'efficiency':
            plt.figure(figsize=(8,7))
            effic = hists['hist_passed'][idx] / hists['hist_total'][idx]
            effic[np.isnan(effic)] = 0.
            plt.errorbar(hists['hist_bins'][idx][0:-1], effic, effic*0.1, capsize=2, marker='s', markersize=3,
                         color='black', linestyle='None')
            plt.title(hname)
            leg = [l.decode('UTF-8') for l in hists['hist_legend'][idx]]
            plt.legend(leg)
            plt.ylabel(hists['hist_ylabel'][idx].decode('UTF-8'))
            plt.savefig('figs/' + 'effic_' + hname + '.pdf')
        elif htype == b'hist':
            mc_scale = np.sum(data_hists[hname]) / np.sum(hists['hist_1d'][idx])
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,7), sharex=True, gridspec_kw={'height_ratios': [6, 1]})
            hep.histplot(hists['hist_1d'][idx] * mc_scale, hists['hist_bins'][idx], histtype='fill', ax=ax1)
            ax1.errorbar(bin_center, data_hists[hname], np.sqrt(data_hists[hname]), capsize=2, marker='s', markersize=3, color='black', linestyle='None')
            ax1.set_title(hname)
            legend = [l.decode('UTF-8') for l in hists['hist_legend'][idx]]
            ax1.legend(legend + ['Data'])
            ax1.set_ylabel(hists['hist_ylabel'][idx].decode('UTF-8'))
            ratio = data_hists[hname] / (hists['hist_1d'][idx] * mc_scale)
            ratio[np.isinf(abs(ratio))] = np.nan
            ratio[(ratio >= 2) | (ratio <= 0)] = np.nan
            ratio = np.ones(len(ratio)) * 0 if np.all(np.isnan(ratio)) else ratio
            ax2.errorbar(bin_center, ratio, ratio * 0.1, capsize=2, marker='s', markersize=3, color='black', linestyle='None')
            plt.grid()
            ax2.set_xlabel(hists['hist_xlabel'][idx].decode('UTF-8'))
            ax2.set_ylabel('Data/MC', fontsize=14)
            ax2.set_ylim(0, 2)
            plt.savefig('figs/' + 'hist1d_' + hname + '.pdf')


def get_data_hists(hists):
    data_dict = {}
    for idx in range(len(hists['hist_name'])):
        if hists['hist_type'][idx] == b'hist':
            data_dict[hists['hist_name'][idx].decode('UTF-8')] = hists['hist_1d'][idx]

    return data_dict


if __name__ == '__main__':

    data_hist_file = 'test_hist_file.hdf5'
    with h5py.File(data_hist_file, 'r') as hists:
        data_hists = get_data_hists(hists=hists)

    hist_file = 'test_hist_file.hdf5'
    with h5py.File(hist_file, 'r') as hists:
        create_plots(hists=hists, data_hists=data_hists)
