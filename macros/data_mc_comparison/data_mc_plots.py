import ROOT
import sys


def create_ratio(h1, h2):
    h3 = h1.Clone("h3")
    h3.SetLineColor(1)
    h3.SetMarkerStyle(20)
    h3.SetMarkerSize(0.6)
    h3.SetTitle("")
    h3.SetMinimum(0.5)
    h3.SetMaximum(1.5)
    # Set up plot for markers and errors
    h3.Sumw2()
    h3.SetStats(0)
    h3.Divide(h2)

    # Adjust y-axis settings
    y = h3.GetYaxis()
    y.SetTitle("Ratio MC/Data ")
    y.SetNdivisions(505)
    y.SetTitleSize(20)
    y.SetTitleFont(43)
    y.SetTitleOffset(1.55)
    y.SetLabelFont(43)
    y.SetLabelSize(15)

    # Adjust x-axis settings
    x = h3.GetXaxis()
    x.SetTitleSize(20)
    x.SetTitleFont(43)
    x.SetTitleOffset(0.8)
    x.SetLabelFont(43)
    x.SetLabelSize(15)
    x.SetTitle(h1.GetXaxis().GetTitle())

    return h3


def create_canvas_pads():
    c = ROOT.TCanvas("c", "canvas", 800, 800)

    # Upper histogram plot is pad1
    pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
    pad1.SetBottomMargin(0)  # joins upper and lower plot
    pad1.Draw()

    # Lower ratio plot is pad2
    c.cd()  # returns to main canvas before defining pad2
    pad2 = ROOT.TPad("pad2", "pad2", 0, 0.05, 1, 0.3)
    pad2.SetTopMargin(0)  # joins upper and lower plot
    pad2.SetBottomMargin(0.2)
    pad2.Draw()

    return c, pad1, pad2


def scale_and_plot(mc_hist_dict, data_hist_dict, legend_dict, mc_evts, data_evts, outdir):
    """
    Scale the MC to data, i.e. num_data_events/num_mc_events
    :param mc_hist_dict: Dict of MC histograms to plot
    :param data_hist_dict: Dict of Data histograms to plot
    :param mc_evts: int Number of MC events
    :param data_evts: int Number of Data events
    :return:
    """
    for hmc in mc_hist_dict:
        # Get each of the MC histograms in the THStack and scale them
        stack_hists = mc_hist_dict[hmc].GetHists()
        for h in stack_hists:
            h.Scale(data_evts / mc_evts)
        mc_hist_dict[hmc].Modified()

        # Create the ratio plot histogram
        mc_hist = ROOT.TH1D(mc_hist_dict[hmc].GetStack().Last())
        ratio_hist = create_ratio(mc_hist, data_hist_dict[hmc])
        c, pad1, pad2 = create_canvas_pads()

        # Draw and save everything
        pad1.cd()
        mc_hist_dict[hmc].Draw("HIST")
        data_hist_dict[hmc].SetMarkerColor(1)
        data_hist_dict[hmc].SetMarkerStyle(20)
        data_hist_dict[hmc].SetMarkerSize(0.6)
        data_hist_dict[hmc].SetLineColor(1)
        data_hist_dict[hmc].Draw("SAME;E0")

        # Draw the legend
        legend_dict[hmc].AddEntry(data_hist_dict[hmc], "Data")
        legend_dict[hmc].Draw("SAME")

        pad2.cd()
        ratio_hist.Draw("SAME;E1")

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
    legend_dict = {}

    for hist in hist_list:
        h_name = hist.GetName()
        # Only compare the stacked by PDG histograms
        if h_name[0:5] != "stack" or h_name[13:17] == "proc" or h_name[14:18] == "proc":
            continue
        mc_hist_name_dict[h_name] = mcfile.Get(h_name)
        # Obviously Data doesn't have stacked histograms as this relies on truth info
        data_name = "hist" + h_name[5:]
        data_hist_name_dict[h_name] = datafile.Get(data_name)
        legend_name = "legend_" + h_name
        legend_dict[h_name] = mcfile.Get(legend_name)

    return mc_hist_name_dict, data_hist_name_dict, legend_dict


def run_plot(mcfile, datafile, outdir):

    ROOT.gROOT.SetBatch(True)  # Turn off Root Tcanvas display

    open_mc_file = ROOT.TFile.Open(mcfile, "READ")
    open_data_file = ROOT.TFile.Open(datafile, "READ")

    total_mc_events = open_mc_file.Get("hist_total_events").GetBinContent(1)
    total_data_events = open_data_file.Get("hist_total_events").GetBinContent(1)
    print("Total MC/Data", total_mc_events, "/", total_data_events, " events.")

    mc_hist_dict, data_hist_dict, legend_dict = get_mc_data_histograms(open_mc_file, open_data_file)

    open_plot_file = ROOT.TFile.Open(outdir + "/mc_data_plots.root", "RECREATE")
    scale_and_plot(mc_hist_dict, data_hist_dict, legend_dict, total_mc_events, total_data_events, outdir)
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
