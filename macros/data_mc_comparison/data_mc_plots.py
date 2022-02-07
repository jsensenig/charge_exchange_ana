import ROOT
import sys


def scale_and_plot(mc_hist_dict, data_hist_dict, mc_evts, data_evts, outdir):
    """
    Scale the MC to data, i.e. num_data_events/num_mc_events
    :param mc_hist_dict: Dict of MC histograms to plot
    :param data_hist_dict: Dict of Data histograms to plot
    :param mc_evts: int Number of MC events
    :param data_evts: int Number of Data events
    :return:
    """
    for hmc in mc_hist_dict:
        c = ROOT.TCanvas()
        # Get each of the MC histograms in the THStack and scale them
        stack_hists = mc_hist_dict[hmc].GetHists()
        for h in stack_hists:
            h.Scale(data_evts / mc_evts)
        mc_hist_dict[hmc].Modified()
        # Draw and save everything
        mc_hist_dict[hmc].Draw("HIST")
        data_hist_dict[hmc].SetMarkerColor(1)
        data_hist_dict[hmc].SetMarkerStyle(20)
        data_hist_dict[hmc].SetMarkerSize(0.6)
        data_hist_dict[hmc].Draw("SAME;E0")
        c.Draw()
        c.Write(hmc)
        c.SaveAs(outdir + "/" + hmc + ".png")


def get_mc_data_histograms(mcfile, datafile):
    """
    Get all histograms we want to plot and stor in dictionary, one for MC and one for Data
    :param mcfile: str Path and file name of MC histogram files
    :param datafile: str Path and file name of Data histogram files
    :return: dict, dict
    """
    hist_list = mcfile.GetListOfKeys()
    print("Number of histograms:", len(hist_list))

    mc_hist_name_dict = {}
    data_hist_name_dict = {}

    for hist in hist_list:
        h_name = hist.GetName()
        # Only compare the stacked by PDG histograms
        if h_name[0:5] != "stack" or h_name[13:17] == "proc" or h_name[14:18] == "proc":
            continue
        mc_hist_name_dict[h_name] = mcfile.Get(h_name)
        # Obviously Data doesn't have stacked histograms as this relies on truth info
        data_name = "hist" + h_name[5:]
        data_hist_name_dict[h_name] = datafile.Get(data_name)

    return mc_hist_name_dict, data_hist_name_dict


def run_plot(mcfile, datafile, outdir):

    open_mc_file = ROOT.TFile.Open(mcfile, "READ")
    open_data_file = ROOT.TFile.Open(datafile, "READ")

    total_mc_events = open_mc_file.Get("hist_total_events").GetBinContent(1)
    total_data_events = open_data_file.Get("hist_total_events").GetBinContent(1)
    print("Total MC/Data", total_mc_events, "/", total_data_events, " events.")

    mc_hist_dict, data_hist_dict = get_mc_data_histograms(open_mc_file, open_data_file)

    open_plot_file = ROOT.TFile.Open(outdir + "/mc_data_plots.root", "RECREATE")
    scale_and_plot(mc_hist_dict, data_hist_dict, total_mc_events, total_data_events, outdir)
    open_plot_file.Close()

    open_mc_file.Close()
    open_data_file.Close()


if __name__ == "__main__":

    # Paths to MC and Data files
    mc_file = sys.argv[1]
    data_file = sys.argv[2]
    out_dir = sys.argv[3]

    print("Plotting for MC File:", mc_file)
    print("Data File:", data_file)

    run_plot(mc_file, data_file, out_dir)
