import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tmp.shower_count_direction as sdir
import tmp.shower_likelihood as sll
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

import awkward as ak
import numpy as np


class RecoShowerDirection:
    def __init__(self):

        self.dir = sdir.ShowerDirection()
        self.ll = sll.ShowerLikelihood()

        self.true_scex_only = True
        self.plot_2d = False
        self.cluster_and_plot = True

        self.dbscan_cluster = True
        self.epsilon = 7
        self.min_points = 15

        self.kmean_cluster = False
        self.nclusters = 2

    @staticmethod
    def rotation(thetax, thetay, thetaz, point):
        """
        Rotate a vector by theta around the respective axis, using right-hand rule.
        :param thetax: Rotation by theta around x axis
        :param thetay: Rotation by theta around y axis
        :param thetaz: Rotation by theta around z axis
        :return: Rotated point
        """
        Rx = np.array([[1., 0., 0.], [0., np.cos(thetax), -np.sin(thetax)], [0., np.sin(thetax), np.cos(thetax)]])
        Ry = np.array([[np.cos(thetay), 0., np.sin(thetay)], [0., 1., 0.], [-np.sin(thetay), 0., np.cos(thetay)]])
        Rz = np.array([[np.cos(thetaz), -np.sin(thetaz), 0.], [np.sin(thetaz), np.cos(thetaz), 0.], [0., 0., 1.]])

        return ((Rx @ Ry @ Rz) @ point.T).T

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

        gamma_id = ak.to_numpy(event["true_beam_Pi0_decay_ID", gamma_daughter_mask])
        gamma_energy = ak.to_numpy(event["true_beam_Pi0_decay_startP", gamma_daughter_mask])
        gamma_energy *= 1.e3

        # Calculate the cos angle between beam and pi0 direction by taking the dot product of their
        # respective direction unit vectors
        return np.arccos(beam_dir_unit @ gamma_dir_unit.T), gamma_id, gamma_energy

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
        vtxx = "reco_beam_calo_endX"
        vtxy = "reco_beam_calo_endY"
        vtxz = "reco_beam_calo_endZ"

        if event[vtxx] is None:
            return None

        vertex = np.vstack((ak.to_numpy(event[vtxx]), ak.to_numpy(event[vtxy]), ak.to_numpy(event[vtxz]))).T
        xyz = np.vstack((ak.to_numpy(event[spx]), ak.to_numpy(event[spy]), ak.to_numpy(event[spz]))).T

        if len(xyz) < 1:
            return None

        # Shift origin to beam vertex
        xyz -= vertex
        #print("SHAPE", xyz.shape)
        #print("PRE-ROTATION", xyz)
        thetax, thetay, thetaz = self.get_beam_angles(event=event)
        rotated_xyz = xyz #self.rotation(thetax, thetay, thetaz, xyz)
        #print("SHAPE", rotated_xyz.shape)
        #print("POST-ROTATION", rotated_xyz)
        return self.dir.transform_to_spherical_numpy(xyz=rotated_xyz)

    def select_reco_scex_event(self, event):

        # Reject event which is empty or not enough spacepoints
        if event is None or len(event["reco_daughter_PFP_shower_spacePts_X"]) < 50:
            return False, None

        # Reject event which has no spacepoints after conversion to spherical coordinates
        #coord = self.dir.transform_to_spherical(events=event)
        coord = self.get_spacepoints(event)

        if coord is None:
            #print("COORD NONE:")
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
        #print("LAST RETURN:")
        return False, None

    def run_shower_reco(self, events):

        pp = PdfPages('/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/multipage1.pdf')
        reco_selected_npi0_list = []
        true_selected_npi0_list = []
        reco_angle_list = []
        true_angle_list = []
        pi0_invariant_mass_list = []

        for i in range(0, len(events)):

            if not i % 100:
                print("Events Loop:", i)

            if not self.true_scex_event(events[i]) and self.true_scex_only:
            #if self.true_scex_event(events[i]) and self.true_scex_only:
                continue

            selected_event, coord = self.select_reco_scex_event(event=events[i])

            if not selected_event:
                continue

            print("*********************************************************")

            gamma_theta, true_gamma_id, true_gamma_energy = self.get_true_gamma_direction(event=events[i])
            print("TRUE GAMMA THETA", np.degrees(gamma_theta))
            print("TRUE GAMMA ID", true_gamma_id)

            if not i % 10:
                print("Events processed:", i)

            scex = self.true_scex_event(events[i])
            npi0 = events["true_daughter_nPi0", i]
            event = events["event", i]
            nsp = events["reco_daughter_PFP_shower_spacePts_count", i]

            if npi0 != 1:
                continue

            reco_selected_npi0_list.append(npi0)

            title_prefix = "Event: " + str(event) + " (i=" + str(i) + " nPi0=" + str(npi0) + " nSP=" + str(nsp) \
                           + " sCEX=" + str(scex) + ")  nSP Len=" + str(len(coord))

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
                if self.dbscan_cluster:
                    clustering = DBSCAN(eps=self.epsilon, min_samples=self.min_points).fit(coord[:, 4:6])
                    count_list = []
                    for cls in np.unique(clustering.labels_):
                        cluster_mask = clustering.labels_ == cls
                        plt.plot(coord[:, 4][cluster_mask], coord[:, 5][cluster_mask], marker='.', linestyle='None', markersize=1)
                        count_list.append(str(np.count_nonzero(cluster_mask)))
                        cluster_mean = np.mean(coord[:, 4:6][cluster_mask], axis=0)
                        print("CLUSTER MEAN", cluster_mean)
                        plt.plot(cluster_mean[0], cluster_mean[1], marker='.', linestyle='None', markersize=7, color='red')
                    plt.title("Event: " + str(event) + " (i=" + str(i) + ") $\\theta\phi$ DBSCAN")
                    plt.xlabel("$\\theta$ [deg]")
                    plt.ylabel("$\phi$ [deg]")
                    plt.legend(count_list)
                    pp.savefig()
                    plt.close()
                elif self.kmean_cluster:
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

                mother_unique_ids = np.unique(events["reco_daughter_PFP_shower_spacePts_mother_ID",i])
                legend_list = []
                for id in mother_unique_ids:
                    id_mask = events["reco_daughter_PFP_shower_spacePts_mother_ID",i] == id
                    legend_list.append(str(np.unique(events["reco_daughter_PFP_shower_spacePts_gmother_PDG", i][id_mask])))
                    plt.plot(coord[:, 3][id_mask], coord[:, 4][id_mask], marker='.', linestyle='None', markersize=1)
                plt.title("Event: " + str(event) + " (i=" + str(i) + ") $R\phi$")
                plt.xlabel("$R$ [cm]")
                plt.ylabel("$\\theta$ [deg]")
                plt.legend(legend_list)
                pp.savefig()
                plt.close()

                for id in mother_unique_ids:
                    id_mask = events["reco_daughter_PFP_shower_spacePts_mother_ID",i] == id
                    legend_list.append(str(np.unique(events["reco_daughter_PFP_shower_spacePts_gmother_PDG", i][id_mask])))
                    plt.plot(coord[:, 4][id_mask], coord[:, 5][id_mask], marker='.', linestyle='None', markersize=1)
                plt.title("Event: " + str(event) + " (i=" + str(i) + ") $\\theta\phi$")
                plt.xlabel("$\\theta$ [deg]")
                plt.ylabel("$\phi$ [deg]")
                plt.legend(legend_list)
                pp.savefig()
                plt.close()

                ax = plt.axes(projection='3d')
                legend_list = []
                for id in mother_unique_ids:
                    id_mask = events["reco_daughter_PFP_shower_spacePts_mother_ID",i] == id
                    ax.scatter3D(coord[:, 2][id_mask], coord[:, 0][id_mask], coord[:, 1][id_mask], marker='.', s=5)
                ax.scatter3D([0], [0], [0], marker='^', s=10, color='red')
                ax.view_init(20, -80)
                plt.title(title_prefix + " XYZ")
                ax.set_xlabel('Z')
                ax.set_ylabel('X')
                ax.set_zlabel('Y')
                ax.legend(legend_list, loc='upper left')
                pp.savefig()

                plt.close()

            else:
                clustering = DBSCAN(eps=self.epsilon, min_samples=self.min_points).fit(coord[:, 4:6])
                max_sp_count = 0.
                sec_max_sp_count = 0.
                leading_shower = 0.
                subleading_shower = 0.
                angle_sp_list = []
                for cls in np.unique(clustering.labels_):
                    cluster_mask = clustering.labels_ == cls
                    cluster_mean = np.mean(coord[:, 4:6][cluster_mask], axis=0)

                    counts = np.unique(events["reco_daughter_PFP_shower_spacePts_mother_ID", i][cluster_mask], return_counts=True)
                    id_mask = counts[1] == np.sort(counts)[1][-1]
                    majority_id = counts[0][id_mask][0]
                    sp_count = np.count_nonzero(cluster_mask)
                    print("CLUSTER MEAN:", cluster_mean, " nSP:", sp_count, " ID:", majority_id)
                    angle_sp_list.append([sp_count, cluster_mean[0]])

                angle_sp_list = np.asarray(angle_sp_list)
                sorted_angles = np.sort(angle_sp_list, axis=0)
                if len(sorted_angles) > 1:
                    theta_gg = abs(sorted_angles[-1, 1] - sorted_angles[-2, 1])
                else:
                    theta_gg = 0.
                print("THETA_GG", theta_gg)
                if len(true_gamma_energy) > 1:
                    inv_mass = np.sqrt(2. * true_gamma_energy[0] * true_gamma_energy[1] * (1 - np.cos(theta_gg)))
                else:
                    inv_mass = 0.
                pi0_invariant_mass_list.append(inv_mass)

            if False and i!=0 and not i % 400:
                break

        # Close the pdf file
        pp.close()

        plt.hist(pi0_invariant_mass_list, range=[0, 500], bins=25)
        plt.savefig("/Users/jsen/tmp/pion_qe/cex_selection/macros/shower_direction_study/pi0_mass.png")

        print("Selected Events True nPi0: ", np.unique(events["true_daughter_nPi0"], return_counts=True))
        print("Selected Events True nPi+: ", np.unique(events["true_daughter_nPiPlus"], return_counts=True))

        return
