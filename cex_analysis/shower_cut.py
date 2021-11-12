from cex_analysis.event_selection_base import EventSelectionBase
import numpy as np
import tmp.shower_count_direction as sdir


class ShowerCut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "ShowerCut"
        self.config = config
        self.reco_beam_pdg = self.config["reco_daughter_pdg"]

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)

        # FIXME test shower counting, make local class object
        self.dir = sdir.ShowerDirection()

    def cnn_shower_cut(self, events):
        # Create a mask for all daughters with CNN EM-like score <0.5
        return events[self.local_config["track_like_cnn_var"]] < self.local_config["cnn_shower_cut"]

    def min_shower_energy_cut(self, events):
        return events[self.local_config["shower_energy_var"]] > self.local_config["small_energy_shower_cut"]

    def shower_count_cut(self, events):
        """
        1. CNN cut to select shower-like daughters
        2. Energy cut, eliminate small showers from e.g. de-excitation gammas
        :param events:
        :return:
        """
        # Perform a 2 step cut on showers, get a daughter mask from each
        cnn_shower_mask = self.cnn_shower_cut(events)
        min_shower_energy = self.min_shower_energy_cut(events)

        # Shower selection mask
        shower_mask = cnn_shower_mask & min_shower_energy

        # We want to count the number of potential showers in each event
        shower_count = np.count_nonzero(events[self.local_config["shower_energy_var"], shower_mask], axis=1)

        # Create the event mask, true if there are 2 candidate showers
        return shower_count == 2

    def selection(self, events, hists):
        # First we configure the histograms we want to make
        hists.configure_hists(self.local_hist_config)

        # The variable on which we cut
        cut_variable = self.local_config["cut_variable"]

        # Plot the variable before making cut
        self.plot_particles_base(events=events, pdg=events[self.reco_beam_pdg], precut=True, hists=hists)

        # Candidate shower count mask
        selected_mask = self.shower_count_cut(events)

        ###############################
        # Method which counts the peaks in theta-phi histogram as a proxy for
        # number of showers.
        use_new_method = True
        if use_new_method:
            peak_count_list = []
            # for evt in events:
            for i in range(0, len(events)):
                print("---> I", events["event"][i])
                if events[i] is None:
                   peak_count_list.append(0)
                   continue
                coord = self.dir.transform_to_spherical(events=events[i])
                if coord is None:
                   peak_count_list.append(0)
                   continue
                rmask = coord[:, 3] <= (14. * 3.)  # 3 * X_0
                shower_dir = []
                if np.count_nonzero(rmask) > 0:
                    shower_dir = self.dir.get_shower_direction_unit(coord[rmask])
                    # shower_dir = self.dir.get_shower_direction_unit(coord)
                peak_count_list.append(len(shower_dir))
                                                                  
            selected_mask = (np.asarray(peak_count_list) == 1) | (np.asarray(peak_count_list) == 2)
            print("Shower Count: 1/2 =", np.sum(np.asarray(peak_count_list) == 1), "/", np.sum(np.asarray(peak_count_list) == 2)) 
        ###############################

        # Plot the variable after cut
        self.plot_particles_base(events=events[selected_mask], pdg=events[self.reco_beam_pdg, selected_mask],
                                 precut=False, hists=hists)

        # Plot the efficiency
        self.efficiency(total_events=events, passed_events=events[selected_mask], cut=self.cut_name, hists=hists)

        # Return event selection mask
        return selected_mask

    def plot_particles_base(self, events, pdg, precut, hists):
        hists.plot_process(x=events, precut=precut)
        for idx, plot in enumerate(self.local_hist_config):
            hists.plot_process_stack(x=events, idx=idx, variable=plot, precut=precut)
            hists.plot_particles_stack(x=events[plot], x_pdg=pdg, idx=idx, precut=precut)
            hists.plot_particles(x=events[plot], idx=idx, precut=precut)

    def efficiency(self, total_events, passed_events, cut, hists):
        for idx, plot in enumerate(self.local_hist_config):
            hists.plot_efficiency(xtotal=total_events[plot], xpassed=passed_events[plot], idx=idx)

    def get_cut_doc(self):
        doc_string = "Cut on daughter showers"
        return doc_string
