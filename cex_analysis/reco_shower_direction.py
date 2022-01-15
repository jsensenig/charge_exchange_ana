from cex_analysis.event_selection_base import EventSelectionBase
from cex_analysis.true_process import TrueProcess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tmp.shower_count_direction as sdir
import tmp.shower_likelihood as sll
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

import awkward as ak
import numpy as np


class RecoShowerDirection(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "RecoShowerDirection"
        self.config = config
        self.reco_beam_pdf = self.config["reco_beam_pdg"]

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)

        self.dir = sdir.ShowerDirection()
        self.ll = sll.ShowerLikelihood()

        self.pi0_mass = 134.98 # MeV

        self.true_scex_only = False
        self.plot_2d = False
        self.cluster_and_plot = False
        self.use_event_selection = False

        self.dbscan_cluster = True
        self.epsilon = 5 #7
        self.min_points = 10 #15

        self.kmean_cluster = False
        self.nclusters = 2

        self.lower_rcut = 5.
        self.inv_mass_cut = 0.1

        self.interaction_process_list = TrueProcess.get_process_list()

    def selection(self, events, hists):
        """
        Start function so this class can be called and run from the main analysis.
        :param events:
        :param hists:
        :return:
        """
        # First we configure the histograms we want to make
        hists.configure_hists(self.local_hist_config)

        return self.run_shower_reco(events=events, hists=hists)

    @staticmethod
    def rotation(thetax, thetay, thetaz, point):
        """
        Rotate a vector by theta around the respective axis, using right-hand rule.
        Rotated by constructing the appropriate rotational matrix in x,y,z and applying them to the point.
        :param thetax: Rotation by theta around x axis
        :param thetay: Rotation by theta around y axis
        :param thetaz: Rotation by theta around z axis
        :return: Rotated point
        """
        Rx = np.array([[1., 0., 0.], [0., np.cos(thetax), -np.sin(thetax)], [0., np.sin(thetax), np.cos(thetax)]])
        Ry = np.array([[np.cos(thetay), 0., np.sin(thetay)], [0., 1., 0.], [-np.sin(thetay), 0., np.cos(thetay)]])
        Rz = np.array([[np.cos(thetaz), -np.sin(thetaz), 0.], [np.sin(thetaz), np.cos(thetaz), 0.], [0., 0., 1.]])

        return ((Rx @ Ry @ Rz) @ point.T).T

    @staticmethod
    def unit_spherical_to_cartesian(theta, phi):
        """
        Convert from spherical to Cartesian coordinates
        Assumes unit vector, i.e. R  = 1 so only need the two angles.
        :param theta:
        :param phi:
        :return:
        """
        return np.vstack((np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta))).T[0]

    def get_beam_angles(self, event):

        # Beam direction variables
        beam_end_px = "true_beam_endPx"
        beam_end_py = "true_beam_endPy"
        beam_end_pz = "true_beam_endPz"

        # Convert to numpy array and combine from (N,1) to (N,3) shape, i.e. each row is a 3D vector and normalize
        beam_dir = np.vstack((ak.to_numpy(event[beam_end_px]),
                              ak.to_numpy(event[beam_end_py]),
                              ak.to_numpy(event[beam_end_pz]))).T
        beam_norm = np.linalg.norm(beam_dir, axis=1)

        # If beam direction is all 0 then no rotation needed, return all 0 angles
        if np.sum(beam_dir) == 0.:
            return 0., 0., 0.

        beam_dir_unit = beam_dir / np.stack((beam_norm, beam_norm, beam_norm), axis=1)

        beam_dir_unit = beam_dir_unit[0]

        # z/sqrt(y*y + z*z)
        thetax = beam_dir_unit[2] / np.sqrt(beam_dir_unit[1] ** 2 + beam_dir_unit[2] ** 2)
        # z/sqrt(x*x + z*z)
        thetay = beam_dir_unit[2] / np.sqrt(beam_dir_unit[0] ** 2 + beam_dir_unit[2] ** 2)
        # x/sqrt(x*x + y*y)
        thetaz = beam_dir_unit[0] / np.sqrt(beam_dir_unit[0]**2 + beam_dir_unit[1]**2)

        return thetax, thetay, thetaz

    def get_true_gamma_direction(self, event):

        # Beam direction variables
        beam_end_px = "true_beam_endPx"
        beam_end_py = "true_beam_endPy"
        beam_end_pz = "true_beam_endPz"

        # Daughter pi0 direction variables
        daughter_pdg = "true_beam_Pi0_decay_PDG"
        gamma_start_px = "true_beam_Pi0_decay_startPx"
        gamma_start_py = "true_beam_Pi0_decay_startPy"
        gamma_start_pz = "true_beam_Pi0_decay_startPz"

        # Convert to numpy array and combine from (N,1) to (N,3) shape, i.e. each row is a 3D vector and normalize
        beam_dir = np.vstack((ak.to_numpy(event[beam_end_px]),
                              ak.to_numpy(event[beam_end_py]),
                              ak.to_numpy(event[beam_end_pz]))).T
        beam_norm = np.linalg.norm(beam_dir, axis=1)

        if np.sum(beam_dir) == 0.:
            beam_dir_unit = beam_dir
        else:
            beam_dir_unit = beam_dir / np.stack((beam_norm, beam_norm, beam_norm), axis=1)
        beam_dir_unit = beam_dir_unit[0]

        # Select only the pi0 daughter
        gamma_daughter_mask = event[daughter_pdg] == 22
        gamma_dir_px = ak.to_numpy(event[gamma_start_px, gamma_daughter_mask])
        gamma_dir_py = ak.to_numpy(event[gamma_start_py, gamma_daughter_mask])
        gamma_dir_pz = ak.to_numpy(event[gamma_start_pz, gamma_daughter_mask])

        # Convert to numpy array and combine from (N,1) to (N,3) shape, i.e. each row is a 3D vector
        # and normalize
        gamma_dir = np.vstack((gamma_dir_px, gamma_dir_py, gamma_dir_pz)).T
        gamma_norm = np.linalg.norm(gamma_dir, axis=1)
        gamma_dir_unit = gamma_dir / np.stack((gamma_norm, gamma_norm, gamma_norm), axis=1)

        # Each gamma angle wrt to beam direction
        gamma_angle = [np.degrees(np.arccos(angle @ beam_dir_unit.T)) for angle in gamma_dir_unit]
        print("TRUE GAMMA ANGLES", gamma_angle)

        gamma_id = ak.to_numpy(event["true_beam_Pi0_decay_ID", gamma_daughter_mask])
        gamma_energy = ak.to_numpy(event["true_beam_Pi0_decay_startP", gamma_daughter_mask])
        gamma_energy *= 1.e3

        # Calculate the cos angle between beam and pi0 direction by taking the dot product of their
        # respective direction unit vectors
        if len(gamma_dir_unit) > 1:
            theta_gg = np.degrees(np.arccos(gamma_dir_unit[0] @ gamma_dir_unit[1].T))
        else:
            theta_gg = 0.

        # Sort the gamma energy in ascending order
        energy_sorted_idx = np.argsort(gamma_energy)

        return theta_gg, gamma_id[energy_sorted_idx], gamma_energy[energy_sorted_idx], np.asarray(gamma_angle)[energy_sorted_idx]

    @staticmethod
    def true_scex_event(event):
        return event["true_beam_PDG"] == 211 and event["true_beam_endProcess"] == "pi+Inelastic" and \
                          event["true_daughter_nPi0"] == 1 and \
                          event["true_daughter_nPiPlus"] == 0 and event["true_daughter_nPiMinus"] == 0 and \
                          (event["true_daughter_nProton"] > 0 or event["true_daughter_nNeutron"] > 0)

    def get_spacepoints(self, event):

        spx = "reco_daughter_PFP_shower_spacePts_X"
        spy = "reco_daughter_PFP_shower_spacePts_Y"
        spz = "reco_daughter_PFP_shower_spacePts_Z"
        # TODO trying SCE uncorrected vertex since the SPs aren't corrected
        vtxx = "reco_beam_endX"
        vtxy = "reco_beam_endY"
        vtxz = "reco_beam_endZ"

        if event[vtxx] is None:
            return None

        vertex = np.vstack((ak.to_numpy(event[vtxx]), ak.to_numpy(event[vtxy]), ak.to_numpy(event[vtxz]))).T
        xyz = np.vstack((ak.to_numpy(event[spx]), ak.to_numpy(event[spy]), ak.to_numpy(event[spz]))).T

        if len(xyz) < 1:
            return None

        # Shift origin to beam vertex
        xyz -= vertex

        thetax, thetay, thetaz = self.get_beam_angles(event=event)
        rotated_xyz = xyz #self.rotation(thetax, thetay, thetaz, xyz)

        return self.dir.transform_to_spherical_numpy(xyz=rotated_xyz)

    def select_reco_scex_event(self, event):

        # Reject event which is empty or not enough spacepoints
        if event is None or len(event["reco_daughter_PFP_shower_spacePts_X"]) < 50:
            return False, None

        # Reject event which has no spacepoints after conversion to spherical coordinates
        #coord = self.dir.transform_to_spherical(events=event)
        coord = self.get_spacepoints(event)

        if coord is None:
            print("COORD NONE!")
            return False, None

        # Reject event which has no spacepoints after R cut
        rmask = coord[:, 3] <= (3.5 * 14.)  # 3.5 * gamma_X_0
        shower_dir = []
        if np.count_nonzero(rmask) > 0:
            shower_dir = self.dir.get_shower_direction_unit(coord[rmask])

        # Accept event if it passes likelihood and theta-phi cut, otherwise reject it
        peak_count = len(shower_dir)
        pred_npi0 = self.ll.classify_npi0(coord[:, 3])

        if pred_npi0 == 1 or (pred_npi0 != 1 and peak_count == 2) or (pred_npi0 == 2 and peak_count == 1):
            print("GOOOD!:")
            return True, coord

        return False, None

    @staticmethod
    def check_wrap_around(phi_coord):
        # Check to see if Spacepoints wrap-around (from -180 to +180)
        # if they do, shift all points by pi so the shower isn't separated
        num_edge_points = np.count_nonzero((phi_coord < -170.) | (phi_coord > 170.))
        shifted_points = False

        if num_edge_points > 50:
            positive_mask = phi_coord >= 0
            phi_coord[positive_mask] -= 180.
            phi_coord[~positive_mask] += 180.
            shifted_points = True

        return shifted_points

    def calculate_reco_open_angle(self, event, angle_list):
        """
        Given a list of variables for each cluster, calculate Opening Angle, (Sub)Leading Shower Energy.
        Leading shower is defined as having the most SPs while Subleading the second most SPs.
        If there are more than 2 clusters, merge them into the nearest (in theta,phi space) (Sub)Leading cluster.
        :param event: ith event record
        :param angle_list: Each cluster in list contains, [SP Count, Theta, Phi, Energy]
        :return: [Opening Angle (deg), Leading Shower Energy (MeV), SubLeading Shower Energy (MeV)]
        """
        angle_array = np.asarray(angle_list)
        sorted_counts = np.sort(angle_array[:, 0], axis=0)
        print("SORTED SP COUNTS", sorted_counts)

        beam_end_dirx = "reco_beam_trackEndDirX"
        beam_end_diry = "reco_beam_trackEndDirY"
        beam_end_dirz = "reco_beam_trackEndDirZ"

        # Convert to numpy array and combine from (N,1) to (N,3) shape, i.e. each row is a 3D vector and normalize
        beam_dir = np.vstack((ak.to_numpy(event[beam_end_dirx]),
                              ak.to_numpy(event[beam_end_diry]),
                              ak.to_numpy(event[beam_end_dirz]))).T
        beam_norm = np.linalg.norm(beam_dir, axis=1)

        if np.sum(beam_dir) == 0.:
            beam_dir_unit = beam_dir
        else:
            beam_dir_unit = beam_dir / np.stack((beam_norm, beam_norm, beam_norm), axis=1)
        beam_dir_unit = beam_dir_unit[0]

        if len(sorted_counts) > 1:
            # This catches the rare case when the 2 largest showers contain an equal number of SPs
            if sorted_counts[-1] == sorted_counts[-2]:
                tmp = angle_array[angle_array[:, 0] == sorted_counts[-1]]
                print("TMP", tmp)
                leading_shower = tmp[0]
                subleading_shower = tmp[1]
                # the last 2 sorted lists and only the energy from those lists
                energy_list = tmp[-2:, 3]
            else:
                leading_shower = angle_array[angle_array[:, 0] == sorted_counts[-1]][0]
                subleading_shower = angle_array[angle_array[:, 0] == sorted_counts[-2]][0]
                energy_list = [leading_shower[3], subleading_shower[3]]

            xyz1 = self.unit_spherical_to_cartesian(np.radians(leading_shower[1]), np.radians(leading_shower[2]))
            xyz2 = self.unit_spherical_to_cartesian(np.radians(subleading_shower[1]), np.radians(subleading_shower[2]))
            print("1/2 ", xyz1.shape, "/", xyz2.shape, "/", xyz1, "/", xyz2)

            # Gamma angles wrt beam direction
            gamma_angle = [np.degrees(np.arccos(xyz1 @ beam_dir_unit.T)), np.degrees(np.arccos(xyz2 @ beam_dir_unit.T))]
            print("RECO GAMMA ANGLES", gamma_angle)

            # Merge any additional clusters into the nearest of the 2 largest
            clusters_to_be_merged = angle_array[angle_array[:, 0] < sorted_counts[-2]]
            print("PRE-MERGE", energy_list)
            if clusters_to_be_merged.size > 0:
                for cls in clusters_to_be_merged:
                    if cls[3] <= 0.:
                        continue
                    dist_leading = np.linalg.norm(cls[1:3] - leading_shower[1:3])
                    dist_subleading = np.linalg.norm(cls[1:3] - subleading_shower[1:3])
                    if dist_leading < dist_subleading:
                        energy_list[0] += cls[3]
                    elif dist_subleading < dist_leading:
                        energy_list[1] += cls[3]
            print("POST-MERGE", energy_list)

            energy_list = np.asarray(energy_list)
            energy_sorted_idx = np.argsort(energy_list)

            return np.degrees(np.arccos(xyz1 @ xyz2.T)), energy_list[energy_sorted_idx], np.asarray(gamma_angle)[energy_sorted_idx]
        else:
            return 0., [0., 0.], np.array([0., 0.])

    def pi0_kinematics(self, energy, angle, theta_gg):
        pi0_mass = 134.98 # MeV

        pi0_momentum = np.sqrt(energy[0]**2 + energy[1]**2 + 2.*energy[1]*energy[1]*np.cos(np.radians(theta_gg)))
        pi0_cos_theta = (energy[0]*np.cos(np.radians(angle[0])) + energy[1]*np.cos(np.radians(angle[1]))) / pi0_momentum
        # T = E - m = sqrt(m**2 + p**2) - m
        return np.sqrt(pi0_momentum**2 + pi0_mass**2) - pi0_mass, pi0_cos_theta


    def run_shower_reco(self, events, hists):

        pp = PdfPages('/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/multipage_scex_only4.pdf')
        reco_selected_npi0_list = []
        true_selected_npi0_list = []
        reco_angle_list = []
        true_angle_list = []

        reco_energy_list = []
        reco_energy_product_list = []
        reco_energy_max_list = []
        reco_energy_min_list = []
        reco_energy_sum_list = []
        reco_pi0_energy_sum_list = []
        reco_alpha_list = []
        reco_pi0_energy_list = []
        reco_pi0_cos_theta_list = []
        reco_leading_angle_list = []
        reco_subleading_angle_list = []
        reco_calc_subleading_angle_list = []

        true_energy_list = []
        true_energy_product_list = []
        true_energy_max_list = []
        true_energy_min_list = []
        true_energy_sum_list = []
        true_alpha_list = []
        true_pi0_energy_list = []
        true_pi0_energy_sum_list = []
        true_pi0_cos_theta_list = []
        true_pi0_inv_energy_list = []
        true_leading_angle_list = []
        true_subleading_angle_list = []
        true_scex_pi0_energy_list = []
        true_scex_pi0_cos_theta_list = []

        reco_invariant_mass_list = []
        true_invariant_mass_list = []

        reco_theta_gg_list = []
        true_theta_gg_list = []

        num_clusters_list = []
        processed_event = 0
        none_coord = 0

        for i in range(0, len(events)):

            if not i % 100:
                print("Events Loop:", i)

            scex = events["single_charge_exchange", i]

            process_dict = ak.to_list(events[self.interaction_process_list, i])
            print("PROCESS DICT", process_dict)
            process_arr = np.asarray(list(process_dict))
            process = process_arr[list(process_dict.values())]
            if len(process) < 1:
                process = ""
            else:
                process = process[0]
            print("PROCESS", process)

            if not scex and self.true_scex_only:
            #if scex and self.true_scex_only:
            #if process != "pi0_production":
                continue

            selected_event, coord = self.select_reco_scex_event(event=events[i])

            if coord is None:
                none_coord += 1
                continue

            # Remove (mask out) the Spacepoints with R > 100cm
            print("PRE-R-CUT", len(coord))
            r_mask = coord[:, 3] > self.lower_rcut
            coord = coord[r_mask]
            print("POST-R-CUT", len(coord))
            print("R-MASK LEN", np.count_nonzero(r_mask))

            if not selected_event and self.use_event_selection:
                continue

            print("*********************************************************")

            true_theta_gg, true_gamma_id, true_gamma_energy, true_gamma_angle = self.get_true_gamma_direction(events[i])
            print("TRUE GAMMA Energy", true_gamma_energy)
            print("TRUE GAMMA THETA_GG", true_theta_gg)
            print("TRUE GAMMA ID", true_gamma_id)

            if not i % 10:
                print("Events processed:", i)

            npi0 = events["true_daughter_nPi0", i]
            event = events["event", i]
            nsp = events["reco_daughter_PFP_shower_spacePts_count", i]

            grandmother_pdg = events["reco_daughter_PFP_shower_spacePts_gmother_PDG", i][r_mask]
            mother_id = events["reco_daughter_PFP_shower_spacePts_mother_ID", i][r_mask]

            # if npi0 != 1:
            #     continue

            reco_selected_npi0_list.append(npi0)

            title_prefix = "Event: " + str(event) + " (i=" + str(i) + " nPi0=" + str(npi0) + " nSP=" + str(nsp) \
                           + " Proc=" + str(process) + ")  nSP Len=" + str(len(coord))

            print(title_prefix)

            if self.plot_2d:
                plt.plot(coord[:, 2], coord[:, 0], marker='.', linestyle='None', markersize=1)
                plt.xlabel("Z [cm]")
                plt.ylabel("X [cm]")
                plt.title(title_prefix + " XZ")
                # plt.savefig(fig_path + "event_" + str(event) + "_i" + str(idx) + "_xz.png")
                pp.savefig()
                plt.close()

                plt.plot(coord[:, 2], coord[:, 1], marker='.', linestyle='None', markersize=1)
                plt.title("Event: " + str(event) + " (i=" + str(i) + ") YZ")
                plt.xlabel("Z [cm]")
                plt.ylabel("Y [cm]")
                # plt.savefig(fig_path + "event_" + str(event) + "_i" + str(idx) + "_yz.png")
                pp.savefig()
                plt.close()
            elif self.cluster_and_plot:
                shifted_points = self.check_wrap_around(phi_coord=coord[:, 5])
                if self.dbscan_cluster and not self.kmean_cluster:
                    clustering = DBSCAN(eps=self.epsilon, min_samples=self.min_points).fit(coord[:, 4:6])
                    count_list = []
                    angle_sp_list = []
                    for cls in np.unique(clustering.labels_):
                        cluster_mask = clustering.labels_ == cls
                        sp_count = np.count_nonzero(cluster_mask)
                        plt.plot(coord[:, 4][cluster_mask], coord[:, 5][cluster_mask], marker='.', linestyle='None', markersize=1)
                        count_list.append(str(sp_count))
                        cluster_mean = np.mean(coord[:, 4:6][cluster_mask], axis=0)
                        if shifted_points:
                            cluster_mean[1] += 180. if cluster_mean[1] < 0 else -180.
                        print("CLUSTER MEAN", cluster_mean, " SP COUNT", sp_count)
                        count_list.append("[" + str(round(cluster_mean[0], 1)) + "," + str(round(cluster_mean[1], 1)) + "]")
                        plt.plot(cluster_mean[0], cluster_mean[1], marker='.', linestyle='None', markersize=7, color='red')
                        cluster_energy = np.sum(events["reco_daughter_PFP_shower_spacePts_E", i][cluster_mask])
                        angle_sp_list.append([sp_count, cluster_mean[0], cluster_mean[1], cluster_energy])
                    reco_theta_gg, reco_energies, reco_angles = self.calculate_reco_open_angle(event[i], angle_sp_list)
                    plt.title("Event: " + str(event) + " (i=" + str(i) + ") $\\theta\phi$ DBSCAN Reco/True $\\theta_{gg}$ = "
                              + str(round(reco_theta_gg, 1)) + "/" + str(round(true_theta_gg, 1)))
                    plt.xlabel("$\\theta$ [deg]")
                    plt.ylabel("$\phi$ [deg]")
                    plt.legend(count_list)
                    pp.savefig()
                    plt.close()
                elif self.kmean_cluster and not self.dbscan_cluster:
                    kmeans = KMeans(n_clusters=self.nclusters, random_state=0, init='k-means++').fit(coord[:, 4:6])
                    for cls in np.unique(kmeans.labels_):
                        cluster_mask = kmeans.labels_ == cls
                        plt.plot(coord[:, 4][cluster_mask], coord[:, 5][cluster_mask], marker='.', linestyle='None',markersize=1)
                    print(kmeans.cluster_centers_)
                    for center in kmeans.cluster_centers_:
                        plt.plot(center[0], center[1], marker='.', linestyle='None', markersize=5, color='red')
                    plt.title("Event: " + str(event) + " (i=" + str(i) + ") $\\theta\phi$ KMeans")
                    plt.xlabel("$\\theta$ [deg]")
                    plt.ylabel("$\phi$ [deg]")
                    pp.savefig()
                    plt.close()

                mother_unique_ids = np.unique(mother_id)
                legend_list = []
                for id in mother_unique_ids:
                    id_mask = mother_id == id
                    legend_list.append(str(np.unique(grandmother_pdg[id_mask])))
                    plt.plot(coord[:, 4][id_mask], coord[:, 5][id_mask], marker='.', linestyle='None', markersize=1)
                plt.title("Event: " + str(event) + " (i=" + str(i) + ") $\\theta\phi$")
                plt.xlabel("$\\theta$ [deg]")
                plt.ylabel("$\phi$ [deg]")
                plt.legend(legend_list, markerscale=20)
                pp.savefig()
                plt.close()

                legend_list = []
                for id in mother_unique_ids:
                    id_mask = mother_id == id
                    legend_list.append(str(np.unique(grandmother_pdg[id_mask])))
                    plt.plot(coord[:, 3][id_mask], coord[:, 4][id_mask], marker='.', linestyle='None', markersize=1)
                plt.title("Event: " + str(event) + " (i=" + str(i) + ") $R\phi$")
                plt.xlabel("$R$ [cm]")
                plt.ylabel("$\\theta$ [deg]")
                plt.legend(legend_list, markerscale=20)
                pp.savefig()
                plt.close()

                #######################
                # legend_list = []
                # for id in mother_unique_ids:
                #     id_mask = mother_id == id
                #     legend_list.append(str(np.unique(grandmother_pdg[id_mask])))
                #     t = 1. / np.sqrt(coord[:, 4][id_mask])
                #     plt.plot(coord[:, 3][id_mask], t, marker='.', linestyle='None', markersize=1)
                # plt.title("t param. Event: " + str(event) + " (i=" + str(i) + ") $R\phi$")
                # plt.xlabel("$R$ [cm]")
                # plt.ylabel("t")
                # plt.legend(legend_list, markerscale=20)
                # pp.savefig()
                # plt.close()
                #######################

                ax = plt.axes(projection='3d')
                legend_list = []
                for id in mother_unique_ids:
                    id_mask = mother_id == id
                    ax.scatter3D(coord[:, 2][id_mask], coord[:, 0][id_mask], coord[:, 1][id_mask], marker='.', s=5)
                ax.scatter3D([0], [0], [0], marker='^', s=10, color='red')
                ax.view_init(20, -80)
                plt.title(title_prefix + " XYZ")
                ax.set_xlabel('Z')
                ax.set_ylabel('X')
                ax.set_zlabel('Y')
                ax.legend(legend_list, loc='upper left', markerscale=20)
                pp.savefig()

                plt.close()
                processed_event += 1

            else:
                shifted_points = self.check_wrap_around(phi_coord=coord[:, 5])
                angle_sp_list = []
                if self.dbscan_cluster and not self.kmean_cluster:
                    clustering = DBSCAN(eps=self.epsilon, min_samples=self.min_points).fit(coord[:, 4:6])
                    for cls in np.unique(clustering.labels_):
                        cluster_mask = clustering.labels_ == cls
                        #cluster_mean = np.median(coords[:, 4:6][cluster_mask], axis=0)
                        cluster_mean = np.mean(coord[:, 4:6][cluster_mask], axis=0)
                        if shifted_points:
                            cluster_mean[1] += 180. if cluster_mean[1] < 0 else -180.
                        counts = np.unique(mother_id[cluster_mask], return_counts=True)
                        id_mask = counts[1] == np.sort(counts)[1][-1]
                        majority_id = counts[0][id_mask][0]
                        sp_count = np.count_nonzero(cluster_mask)
                        print("CLUSTER MEAN:", cluster_mean, " nSP:", sp_count, " ID:", majority_id)
                        cluster_energy = events["reco_daughter_PFP_shower_spacePts_E", i][cluster_mask]
                        angle_sp_list.append([sp_count, cluster_mean[0], cluster_mean[1], np.sum(cluster_energy[cluster_energy > -1.])])
                elif self.kmean_cluster and not self.dbscan_cluster:
                    kmeans = KMeans(n_clusters=self.nclusters, random_state=0, init='k-means++').fit(coord[:, 4:6])
                    for cls in np.unique(kmeans.labels_):
                        cluster_mask = kmeans.labels_ == cls
                        cluster_mean = np.mean(coord[:, 4:6][cluster_mask], axis=0)
                        if shifted_points:
                            cluster_mean[1] += 180. if cluster_mean[1] < 0 else -180.
                        counts = np.unique(mother_id[cluster_mask], return_counts=True)
                        id_mask = counts[1] == np.sort(counts)[1][-1]
                        majority_id = counts[0][id_mask][0]
                        sp_count = np.count_nonzero(cluster_mask)
                        print("CLUSTER MEAN:", cluster_mean, " nSP:", sp_count, " ID:", majority_id)
                        cluster_energy = events["reco_daughter_PFP_shower_spacePts_E", i][cluster_mask]
                        angle_sp_list.append([sp_count, cluster_mean[0], cluster_mean[1], np.sum(cluster_energy[cluster_energy > -1.])])
                    print("KMean Center", kmeans.cluster_centers_)
                else:
                    print("No Clustering Method Selected!")
                    raise RuntimeError
                num_clusters_list.append(len(angle_sp_list))
                reco_theta_gg, reco_energies, reco_angles = self.calculate_reco_open_angle(events[i], angle_sp_list)
                print("ENERGY", reco_energies)
                print("RECO THETA_GG", reco_theta_gg)
                reco_theta_gg_list.append(reco_theta_gg)
                if len(reco_energies) > 0:
                    reco_energy_max_list.append(reco_energies[-1])
                    reco_energy_min_list.append(reco_energies[0])
                else:
                    reco_energy_max_list.append(0.)
                    reco_energy_min_list.append(0.)
                reco_energy_product_list.append(np.prod(np.asarray(reco_energies)/1000.))
                print("RECO PROD", np.prod(reco_energies))
                reco_energy_sum_list.append(np.sum(reco_energies))
                reco_pi0_energy_sum_list.append(np.sum(reco_energies) - self.pi0_mass)
                reco_energy_list.append(reco_energies[0])
                reco_energy_list.append(reco_energies[1])

                if len(true_gamma_energy) > 0:
                    true_energy_max_list.append(true_gamma_energy[-1])
                    true_energy_min_list.append(true_gamma_energy[0])
                else:
                    true_energy_max_list.append(0.)
                    true_energy_min_list.append(0.)
                true_energy_sum_list.append(np.sum(true_gamma_energy))
                true_energy_product_list.append(np.prod(np.asarray(true_gamma_energy)/1000.))
                print("TRUE PROD", np.prod(true_gamma_energy))

                reco_inv_mass = np.sqrt(2. * reco_energies[0] * reco_energies[1] * (1 - np.cos(np.radians(reco_theta_gg))))
                reco_invariant_mass_list.append(reco_inv_mass)

                calc_energy = 135.**2 / (2. * reco_energies[-1] * (1. - np.cos(np.radians(reco_theta_gg))))
                if reco_energies[-1]*reco_energies[0] > 0.:
                    calc_theta_2 = np.arccos((self.pi0_mass**2 / (2.*reco_energies[-1]*reco_energies[0]))-1) - np.radians(reco_angles[-1])
                    print("ARCCOS ARG", ((self.pi0_mass ** 2 / (2. * reco_energies[-1] * reco_energies[0])) - 1))
                    #print("THETA2 TRUE/CALC/MEAS", true_gamma_angle[0], "/", calc_theta_2, "/", reco_angles[0])
                else:
                    calc_theta_2 = 0.
                    print("ARCCOS ARG ELSE", 0.)
                    #print("THETA2 TRUE/CALC/MEAS", true_gamma_angle[0], "/", 0., "/", reco_angles[0])

                print("SUB-LEADING SHOWER E MEAS/CALC =", reco_energies[0], "/", calc_energy)
                reco_pi0_ke, reco_pi0_cos_theta = self.pi0_kinematics([calc_energy, reco_energies[-1]],
                                                                      reco_angles,
                                                                      #[calc_theta_2, reco_angles[-1]],
                                                                      reco_theta_gg)

                calc_theta_2 = np.degrees(calc_theta_2)
                reco_pi0_energy_list.append(reco_pi0_ke)
                reco_pi0_cos_theta_list.append(reco_pi0_cos_theta)
                reco_leading_angle_list.append(reco_angles[-1])
                reco_subleading_angle_list.append(reco_angles[0])
                reco_calc_subleading_angle_list.append(calc_theta_2)

                # Calculate alpha (the asymmetry between the decay photon showers)
                if np.sum(reco_energies) > 0.:
                    reco_alpha = abs(reco_energies[0] - reco_energies[1]) / (reco_energies[0] + reco_energies[1])
                else:
                    reco_alpha = 0.
                reco_alpha_list.append(reco_alpha)

                if len(true_gamma_energy) > 1:
                    #reco_inv_mass = np.sqrt(2. * true_gamma_energy[0] * true_gamma_energy[1] * (1 - np.cos(np.radians(reco_theta_gg))))
                    #reco_invariant_mass_list.append(reco_inv_mass)
                    true_alpha = abs(true_gamma_energy[0] - true_gamma_energy[1]) / (true_gamma_energy[0] + true_gamma_energy[1])
                    inv_mass = np.sqrt(2. * true_gamma_energy[0] * true_gamma_energy[1] * (1 - np.cos(np.radians(true_theta_gg))))
                    true_invariant_mass_list.append(inv_mass)
                    true_theta_gg_list.append(true_theta_gg)
                    print("TRUE INV MASS/THETA_GG", inv_mass, "/", true_theta_gg)
                    true_energy_list.append(true_gamma_energy[0])
                    true_energy_list.append(true_gamma_energy[1])
                    true_alpha_list.append(true_alpha)
                    true_pi0_ke, true_pi0_cos_theta = self.pi0_kinematics(true_gamma_energy, true_gamma_angle, true_theta_gg)
                    true_pi0_energy_list.append(true_pi0_ke)
                    true_pi0_inv_energy_list.append(true_pi0_ke+134.98)
                    true_pi0_cos_theta_list.append(true_pi0_cos_theta)
                    true_leading_angle_list.append(true_gamma_angle[-1])
                    true_subleading_angle_list.append(true_gamma_angle[0])
                    true_pi0_energy_sum_list.append(np.sum(true_gamma_energy) - self.pi0_mass)
                else:
                    #reco_invariant_mass_list.append(0.)
                    true_invariant_mass_list.append(0.)
                    true_theta_gg_list.append(0)
                    true_energy_list.append(0.)
                    true_energy_list.append(0.)
                    true_alpha_list.append(0.)
                    true_pi0_energy_list.append(0.)
                    true_pi0_inv_energy_list.append(0.)
                    true_pi0_cos_theta_list.append(0.)
                    true_leading_angle_list.append(0.)
                    true_subleading_angle_list.append(0.)
                    true_pi0_energy_sum_list.append(0.)

                if scex and len(true_gamma_energy) > 1:
                    true_scex_pi0_energy_list.append(true_pi0_ke)
                    true_scex_pi0_cos_theta_list.append(true_pi0_cos_theta)
                else:
                    true_scex_pi0_energy_list.append(-999)
                    true_scex_pi0_cos_theta_list.append(-999)

                processed_event += 1

            # if processed_event != 0 and not processed_event % 75:
            #     break

        # Close the pdf file
        pp.close()

        cnt, bins, _ = plt.hist(num_clusters_list, range=[0, 20], bins=20)
        plt.xticks(bins, rotation=-90)
        plt.xlabel('Num Clusters')
        plt.ylabel('Count')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/dbscan_number_clusters.png")
        plt.close()
        print("DBSCAN nCLUSTERS", np.sum(cnt), "/", processed_event)

        reco_pi0_mass_counts, bins, _ = plt.hist(reco_invariant_mass_list, range=[0, 500], bins=50)
        plt.plot([135, 135], [0, max(reco_pi0_mass_counts)], color='red', linewidth=1, linestyle='--')
        plt.xticks(bins, rotation=-90)
        plt.xlabel('Reco $M_{\gamma\gamma}$ [MeV]')
        plt.ylabel('Count')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/reco_pi0_mass.png")
        plt.close()
        print("RECO Pi0 MASS", np.sum(reco_pi0_mass_counts), "/", processed_event)

        cnt, _, _, _ = plt.hist2d(reco_alpha_list, true_alpha_list, range=[[0,1],[0,1]], bins=[25, 25], cmap='Blues')
        plt.colorbar()
        plt.plot([0, 1500], [0, 1500])
        plt.xlabel('Reco Shower $\\alpha$')
        plt.ylabel('True Shower $\\alpha$')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_shower_alpha.png")
        plt.close()
        print("SHOWER ALPHA", np.sum(cnt), "/", processed_event)

        cnt, _, _, _ = plt.hist2d(reco_energy_sum_list, true_energy_sum_list, range=[[0,1500],[0,1500]], bins=[50, 50], cmap='Blues')
        plt.colorbar()
        plt.plot([0, 1500], [0, 1500])
        plt.xlabel('Reco $\sum E_{\gamma}$ [MeV]')
        plt.ylabel('True $\sum E_{\gamma}$ [MeV]')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_shower_energy_sum.png")
        plt.close()
        print("ENERGY SUM", np.sum(cnt), "/", processed_event)

        ######################
        cnt, _, _, _ = plt.hist2d(reco_pi0_energy_sum_list, true_pi0_energy_list, range=[[0,1500],[0,1500]], bins=[50, 50], cmap='Blues')
        plt.colorbar()
        plt.plot([0, 1500], [0, 1500])
        plt.xlabel('Reco $T_{\pi^0}$ [MeV]')
        plt.ylabel('True $T_{\pi^0}$ [MeV]')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/pi0_inv_energy_reco_true.png")
        plt.close()
        print("RECO INV Pi0 ENERGY", np.sum(cnt), "/", processed_event)

        cnt, _, _ = plt.hist((np.asarray(reco_pi0_energy_sum_list)/np.asarray(true_pi0_energy_list))-1, range=[-1, 1], bins=50)
        plt.xlabel('$T_{\pi^0}$ Reco/True - 1')
        plt.ylabel('Count')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/pi0_inv_energy_reco_true_bias.png")
        plt.close()
        print("RECO INV Pi0 ENERGY BIAS", np.sum(cnt), "/", processed_event)

        cnt, _, _, _ = plt.hist2d(true_pi0_energy_list,
                                  (np.asarray(reco_pi0_energy_sum_list) / np.asarray(true_pi0_energy_list)) - 1,
                                  range=[[0,1200],[-1,1]], bins=[50, 50], cmap=plt.cm.jet)
        plt.colorbar()
        plt.plot([0, 1200], [0, 0])
        plt.ylabel('Reco $T_{\pi^0}$ Reco/True - 1')
        plt.xlabel('True $T_{\pi^0}$ [MeV]')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/pi0_kinetic_energy_sum_bias.png")
        plt.close()
        #############################

        cnt, _, _, _ = plt.hist2d(reco_energy_product_list, true_energy_product_list, range=[[0,0.5],[0,0.5]], bins=[50, 50], cmap='Blues')
        plt.colorbar()
        plt.plot([0, 0.5], [0, 0.5])
        plt.xlabel('Reco $\prod E_{\gamma}$ [GeV]')
        plt.ylabel('True $\prod E_{\gamma}$ [GeV]')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_shower_energy_product.png")
        plt.close()
        print("ENERGY PRODUCT", np.sum(cnt), "/", processed_event)

        cnt, _, _, _ = plt.hist2d(reco_energy_max_list, true_energy_max_list, range=[[0,1000],[0,1000]], bins=[50, 50], cmap='Blues')
        plt.colorbar()
        plt.plot([0, 1000], [0, 1000])
        plt.xlabel('Reco Max $E_{\gamma}$ [MeV]')
        plt.ylabel('True Max $E_{\gamma}$ [MeV]')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_shower_energy_max.png")
        plt.close()
        print("LEADING SHOWER ENERGY TRUE-RECO", np.sum(cnt), "/", processed_event)

        cnt, _, _, _ = plt.hist2d(reco_energy_min_list, true_energy_min_list, range=[[0,1000],[0,1000]], bins=[50, 50], cmap='Blues')
        plt.colorbar()
        plt.plot([0, 1000], [0, 1000])
        plt.xlabel('Reco Min $E_{\gamma}$ [MeV]')
        plt.ylabel('True Min $E_{\gamma}$ [MeV]')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_shower_energy_min.png")
        plt.close()
        print("SUBLEADING SHOWER ENERGY TRUE-RECO", np.sum(cnt), "/", processed_event)

        # Difference (reco/true)-1
        cnt, _, _ = plt.hist((np.asarray(reco_theta_gg_list)/np.asarray(true_theta_gg_list))-1, range=[-1, 4], bins=50)
        plt.xlabel('$\\theta_{\gamma\gamma}$ Reco/True - 1')
        plt.ylabel('Count')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/pi0_theta_gg_reco_true.png")
        plt.close()
        print("THETA_GG BIAS", np.sum(cnt), "/", processed_event)

        cnt_min, _, _ = plt.hist((np.asarray(reco_energy_min_list)/np.asarray(true_energy_min_list))-1, range=[-1, 4], bins=50)
        plt.xlabel('Min $E_{\gamma}$ Reco/True - 1')
        plt.ylabel('Count')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_shower_min_reco_true.png")
        plt.close()
        print("SUBLEADING SHOWER BIAS", np.sum(cnt_min), "/", processed_event)

        cnt, _, _ = plt.hist((np.asarray(reco_energy_max_list)/np.asarray(true_energy_max_list))-1, range=[-1, 1], bins=50)
        plt.xlabel('Max $E_{\gamma}$ Reco/True - 1')
        plt.ylabel('Count')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_shower_max_reco_true.png")
        plt.close()
        print("LEADING SHOWER BIAS", np.sum(cnt), "/", processed_event)

        # Use calculated subleading shower energy
        calculated_subleading_shower = 135.**2 / (2. * np.asarray(reco_energy_max_list) *
                                                  (1. - np.cos(np.radians(np.asarray(reco_theta_gg_list)))))
        calculated_subleading_shower[calculated_subleading_shower == np.inf] = 0.
        calc_cnt, _, _ = plt.hist((calculated_subleading_shower/np.asarray(true_energy_min_list))-1, range=[-1, 4], bins=50)
        plt.xlabel('Calculated Min $E_{\gamma}$ Reco/True - 1')
        plt.ylabel('Count')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_shower_calc_min_reco_true.png")
        plt.close()
        print("CALC. SUBLEADING SHOWER BIAS", np.sum(calc_cnt), "/", processed_event)

        cnt, _, _, _ = plt.hist2d(calculated_subleading_shower, true_energy_min_list, range=[[0,1000],[0,1000]], bins=[50, 50], cmap='Blues')
        plt.colorbar()
        plt.plot([0, 1000], [0, 1000])
        plt.xlabel('Calc Reco Min $E_{\gamma}$ [MeV]')
        plt.ylabel('True Min $E_{\gamma}$ [MeV]')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_shower_energy_calc_min.png")
        plt.close()
        print("CALC. SUBLEADING SHOWER ENERGY TRUE-RECO", np.sum(cnt), "/", processed_event)

        print("Reco Min Integral Meas/Calc: ", np.sum(cnt_min), "/", np.sum(calc_cnt))

        cnt, _, _, _ = plt.hist2d(reco_energy_list, true_energy_list, range=[[0,500],[0,500]], bins=[25, 25], cmap='Blues')
        plt.colorbar()
        plt.plot([0, 500], [0, 500])
        plt.xlabel('Reco $E_{\gamma}$ [MeV]')
        plt.ylabel('True $E_{\gamma}$ [MeV]')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_shower_energy.png")
        plt.close()

        cnt, _, _, _ = plt.hist2d(reco_invariant_mass_list, true_invariant_mass_list, range=[[0,200],[0,200]], bins=[25, 25], cmap='Blues')
        plt.colorbar()
        plt.xlabel('Reco $M_{\gamma\gamma}$ [MeV]')
        plt.ylabel('True $M_{\gamma\gamma}$ [MeV]')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/pi0_inv_mass.png")
        plt.close()
        print("Pi0 INVARIANT MASS TRUE-RECO", np.sum(cnt), "/", processed_event)

        cnt, _, _, _ = plt.hist2d(reco_theta_gg_list, true_theta_gg_list, range=[[0,160],[0,160]], bins=[30, 30], cmap=plt.cm.jet)
        plt.colorbar()
        plt.plot([0, 80], [0, 80])
        plt.xlabel('Reco $\\theta_{\gamma\gamma}$ [deg]')
        plt.ylabel('True $\\theta_{\gamma\gamma}$ [deg]')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/pi0_theta_gg.png")
        plt.close()
        print("THETA_GG TRUE-RECO", np.sum(cnt), "/", processed_event)

        #############################
        # Gamma Angle wrt to Beam
        cnt, _, _, _ = plt.hist2d(reco_leading_angle_list, true_leading_angle_list, range=[[0,180],[0,180]], bins=[30, 30], cmap=plt.cm.jet)
        plt.colorbar()
        plt.plot([0, 180], [0, 180])
        plt.xlabel('Reco $\\theta_{1}$ [deg]')
        plt.ylabel('True $\\theta_{1}$ [deg]')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_leading_angle.png")
        plt.close()
        print("LEADING SHOWER BEAM-GAMMA ANGLE", np.sum(cnt), "/", processed_event)

        cnt, _, _, _ = plt.hist2d(reco_subleading_angle_list, true_subleading_angle_list, range=[[0, 180],[0, 180]], bins=[30, 30], cmap=plt.cm.jet)
        plt.colorbar()
        plt.plot([0, 180], [0, 180])
        plt.xlabel('Reco $\\theta_{2}$')
        plt.ylabel('True $\\theta_{2}$')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_subleading_angle.png")
        plt.close()
        print("SUBLEADING SHOWER BEAM-GAMMA ANGLE", np.sum(cnt), "/", processed_event)

        cnt, _, _ = plt.hist((np.asarray(reco_leading_angle_list) / np.asarray(true_leading_angle_list)) - 1, range=[-1, 1], bins=50)
        plt.xlabel('Reco $\\theta_{1}$ Reco/True - 1')
        plt.ylabel('Count')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_leading_angle_reco_true.png")
        plt.close()
        print("LEADING SHOWER BEAM-GAMMA ANGLE BIAS", np.sum(cnt), "/", processed_event)

        cnt, _, _ = plt.hist((np.asarray(reco_subleading_angle_list) / np.asarray(true_subleading_angle_list)) - 1, range=[-1, 1], bins=50)
        plt.xlabel('Reco $\\theta_{2}$ Reco/True - 1')
        plt.ylabel('Count')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_subleading_angle_reco_true.png")
        plt.close()
        print("SUBLEADING SHOWER BEAM-GAMMA ANGLE BIAS", np.sum(cnt), "/", processed_event)

        cnt, _, _, _ = plt.hist2d(reco_calc_subleading_angle_list, true_subleading_angle_list, range=[[0, 180],[0, 180]], bins=[30, 30], cmap=plt.cm.jet)
        plt.colorbar()
        plt.plot([0, 180], [0, 180])
        plt.xlabel('Reco Calc $\\theta_{2}$')
        plt.ylabel('True $\\theta_{2}$')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_calc_subleading_angle.png")
        plt.close()
        print("CALC SUBLEADING SHOWER BEAM-GAMMA ANGLE", np.sum(cnt), "/", processed_event)

        cnt, _, _ = plt.hist((np.asarray(reco_calc_subleading_angle_list) / np.asarray(true_subleading_angle_list)) - 1, range=[-1, 1], bins=50)
        plt.xlabel('Reco Calc $\\theta_{\pi^0}$ Reco/True - 1')
        plt.ylabel('Count')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/gamma_calc_subleading_angle_reco_true.png")
        plt.close()
        print("CALC SUBLEADING SHOWER BEAM-GAMMA ANGLE BIAS", np.sum(cnt), "/", processed_event)

        #############################
        # Pi0 Kinematics
        cnt, _, _, _ = plt.hist2d(reco_pi0_energy_list, true_pi0_energy_list, range=[[0,1200],[0,1200]], bins=[60, 60], cmap=plt.cm.jet)
        plt.colorbar()
        plt.plot([0, 1200], [0, 1200])
        plt.xlabel('Reco $T_{\pi^0}$ [MeV]')
        plt.ylabel('True $T_{\pi^0}$ [MeV]')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/pi0_kinetic_energy.png")
        plt.close()
        print("Pi0 KE", np.sum(cnt), "/", processed_event)

        cnt, _, _, _ = plt.hist2d(reco_pi0_cos_theta_list, true_pi0_cos_theta_list, range=[[-1, 1],[-1, 1]], bins=[50, 50], cmap=plt.cm.jet)
        plt.colorbar()
        plt.plot([-1, 1], [-1, 1])
        plt.xlabel('Reco cos$\\theta_{\pi^0}$')
        plt.ylabel('True cos$\\theta_{\pi^0}$')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/pi0_cos_theta.png")
        plt.close()
        print("Pi0 COS THETA", np.sum(cnt), "/", processed_event)

        cnt, _, _ = plt.hist((np.asarray(reco_pi0_energy_list) / np.asarray(true_pi0_energy_list)) - 1, range=[-1, 1], bins=50)
        plt.xlabel('Reco $T_{\pi^0}$ Reco/True - 1')
        plt.ylabel('Count')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/pi0_kinetic_energy_reco_true.png")
        plt.close()
        print("Pi0 KE BIAS", np.sum(cnt), "/", processed_event)

        cnt, _, _ = plt.hist((np.asarray(reco_pi0_cos_theta_list) / np.asarray(true_pi0_cos_theta_list)) - 1, range=[-1, 1], bins=50)
        plt.xlabel('Reco cos$\\theta_{\pi^0}$ Reco/True - 1')
        plt.ylabel('Count')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/pi0_cos_theta_reco_true.png")
        plt.close()
        print("Pi0 COS THETA BIAS", np.sum(cnt), "/", processed_event)

        cnt, _, _, _ = plt.hist2d(true_pi0_energy_list,
                                  (np.asarray(reco_pi0_cos_theta_list) / np.asarray(true_pi0_cos_theta_list)) - 1,
                                  range=[[0,1200],[-1,1]], bins=[60, 50], cmap=plt.cm.jet)
        plt.colorbar()
        plt.plot([0, 1200], [0, 0])
        plt.ylabel('Reco $T_{\pi^0}$ Reco/True - 1')
        plt.xlabel('True $T_{\pi^0}$ [MeV]')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/pi0_kinetic_energy_bias.png")
        plt.close()

        cnt, _, _, _ = plt.hist2d(true_pi0_cos_theta_list,
                                  (np.asarray(reco_pi0_cos_theta_list) / np.asarray(true_pi0_cos_theta_list)) - 1,
                                  range=[[-1, 1],[-1, 1]], bins=[50, 50], cmap=plt.cm.jet)
        plt.colorbar()
        plt.plot([-1, 1], [0, 0])
        plt.ylabel('Reco cos$\\theta_{\pi^0}$ Reco/True - 1')
        plt.xlabel('True cos$\\theta_{\pi^0}$')
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/pi0_cos_theta_bias.png")
        plt.close()

        print("Selected Events True nPi0: ", np.unique(events["true_daughter_nPi0"], return_counts=True))
        print("Selected Events True nPi+: ", np.unique(events["true_daughter_nPiPlus"], return_counts=True))
        print("Successfully Processed", processed_event, " events!")
        print("Events with None coordinates: ", none_coord)
        print("Integral of Reco Pi0 Mass:", np.sum(reco_pi0_mass_counts))

        events["daughter_pi0_ke"] = np.asarray(reco_pi0_energy_sum_list)
        events["true_daughter_pi0_ke"] = np.asarray(true_pi0_energy_list)
        events["daughter_pi0_cos_theta"] = np.asarray(reco_pi0_cos_theta_list)
        events["true_daughter_pi0_cos_theta"] = np.asarray(true_pi0_cos_theta_list)

        # Plot the variable before after cut
        self.plot_particles_base(events=events, pdg=events[self.reco_beam_pdf],
                                 precut=False, hists=hists)

        return np.asarray(reco_invariant_mass_list) < 3000.

    def plot_particles_base(self, events, pdg, precut, hists):
        # pass
        hists.plot_process(x=events, precut=precut)
        for idx, plot in enumerate(self.local_hist_config):
            hists.plot_process_stack(x=events, idx=idx, variable=plot, precut=precut)
            hists.plot_particles_stack(x=events[plot], x_pdg=pdg, idx=idx, precut=precut)
            hists.plot_particles(x=events[plot], idx=idx, precut=precut)

    def efficiency(self, total_events, passed_events, cut, hists):
        pass

    def get_cut_doc(self):
        doc_string = "Plot the shower-like space-points in the event."
        return doc_string
