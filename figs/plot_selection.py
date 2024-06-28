import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import h5py

plt.style.use(hep.style.ATLASAlt)


def create_plots(hists, data_hists):

    for idx in range(len(hists['hist_name'])):
        hname = hists['hist_name'][idx].decode('UTF-8')
        print(idx, " - ", hname)
        htype = hists['hist_type'][idx]
        if htype == b'stack':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,7), sharex=True, gridspec_kw={'height_ratios': [6, 1]})
            hlist = [h for h in hists['hist_stack'][idx] if len(h) > 0]
            hep.histplot(hlist, hists['hist_bins'][idx], stack=True, histtype='fill', ax=ax1)
            ax1.errorbar((hists['hist_bins'][idx][0:-1]+hists['hist_bins'][idx][1:])/2, data_hists[hname],
                         np.sqrt(data_hists[hname]), capsize=2, marker='s', markersize=3, color='black', linestyle='None')
            ax1.set_title(hname)
            leg = [l.decode('UTF-8') for l in hists['hist_legend'][idx]]
            ax1.legend(leg + ['Data'])
            ax1.set_ylabel(hists['hist_ylabel'][idx].decode('UTF-8'))
            ax2.plot(hists['hist_bins'][idx][0:-1], hlist[0]/hlist[0])
            plt.grid()
            ax2.set_xlabel('X [cm]')
            ax2.set_xlabel(hists['hist_xlabel'][idx].decode('UTF-8'))
            ax2.set_ylabel('Ratio')
            # plt.show()
        elif htype == b'efficiency':
            effic = hists['hist_passed'][idx] / hists['hist_total'][idx]
            effic[np.isnan(effic)] = 0.
            plt.errorbar(hists['hist_bins'][idx][0:-1], effic, effic*0.1, capsize=2, marker='s', markersize=3,
                         color='black', linestyle='None')
            plt.title(hname)
            leg = [l.decode('UTF-8') for l in hists['hist_legend'][idx]]
            plt.legend(leg)
            plt.ylabel(hists['hist_ylabel'][idx].decode('UTF-8'))
            # plt.show()
        elif htype == b'hist':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,7), sharex=True, gridspec_kw={'height_ratios': [6, 1]})
            hep.histplot(hists['hist_1d'][idx], hists['hist_bins'][idx], histtype='fill', ax=ax1)
            ax1.errorbar((hists['hist_bins'][idx][0:-1]+hists['hist_bins'][idx][1:])/2, data_hists[hname],
                         np.sqrt(data_hists[hname]), capsize=2, marker='s', markersize=3, color='black', linestyle='None')
            ax1.set_title(hname)
            leg = [l.decode('UTF-8') for l in hists['hist_legend'][idx]]
            ax1.legend(leg + ['Data'])
            ax1.set_ylabel(hists['hist_ylabel'][idx].decode('UTF-8'))
            ax2.plot(hists['hist_bins'][idx][0:-1], hists['hist_1d'][idx]/hists['hist_1d'][idx])
            plt.grid()
            ax2.set_xlabel('X [cm]')
            ax2.set_xlabel(hists['hist_xlabel'][idx].decode('UTF-8'))
            ax2.set_ylabel('Ratio')
            # plt.show()

        plt.savefig('figs/' + hists['hist_name'][idx].decode('UTF-8') + '.pdf')


def get_data_hists(hists):
    data_dict = {}
    for idx in range(len(hists['hist_name'])):
        if hists['hist_type'][idx] == b'hist':
            data_dict[hists['hist_name'][idx].decode('UTF-8')] = hists['hist_1d'][idx]

    return data_dict


if __name__ == 'main':

    data_hist_file = 'test_hist_file.hdf5'
    with h5py.File(data_hist_file, 'r') as hists:
        data_hists = get_data_hists(hists=hists)

    hist_file = 'test_hist_file.hdf5'
    with h5py.File(hist_file, 'r') as hists:
        create_plots(hists=hists, data_hists=data_hists)
