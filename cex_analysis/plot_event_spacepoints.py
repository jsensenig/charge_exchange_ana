from cex_analysis.event_selection_base import EventSelectionBase
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tmp.shower_count_direction as sdir
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import numpy as np


class PlotEventSpacePoints(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.dir = sdir.ShowerDirection()

        self.plot_2d = False

        self.dbscan_cluster = True
        self.epsilon = 7
        self.min_points = 15

        self.kmean_cluster = False
        self.nclusters = 2

    def selection(self, events, hists):

        pp = PdfPages('/Users/jsen/tmp/pion_qe/cex_selection/shower_figs/multipage13.pdf')

        for i in range(0, len(events)):
            if not i % 10:
                print("Events processed:", i)
            coord = self.dir.transform_to_spherical(events=events[i])

            scex = events["single_charge_exchange", i]
            npi0 = events["true_daughter_nPi0", i]
            event = events["event", i]
            nsp = events["reco_daughter_PFP_shower_spacePts_count", i]

            title_prefix = "Event: " + str(event) + " (i=" + str(i) + " nPi0=" + str(npi0) + " nSP=" + str(nsp) \
                           + " sCEX=" + str(scex) + ")  nSP Len=" + str(len(coord))

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
            else:
                if self.dbscan_cluster:
                    clustering = DBSCAN(eps=self.epsilon, min_samples=self.min_points).fit(coord[:, 4:6])
                    count_list = []
                    for cls in np.unique(clustering.labels_):
                        cluster_mask = clustering.labels_ == cls
                        plt.plot(coord[:, 4][cluster_mask], coord[:, 5][cluster_mask], marker='.', linestyle='None', markersize=1)
                        count_list.append(str(np.count_nonzero(cluster_mask)))
                        cluster_mean = np.mean(coord[:, 4:6][cluster_mask], axis=0)
                        plt.plot(cluster_mean[0], cluster_mean[1], marker='.',linestyle='None',markersize=7,color='red')
                    plt.title("Event: " + str(event) + " (i=" + str(i) + ") $\\theta\phi$ DBSCAN")
                    plt.xlabel("$\\theta$ [deg]")
                    plt.ylabel("$\phi$ [deg]")
                    plt.legend(count_list)
                    pp.savefig()
                    plt.close()
                elif self.kmean_cluster:
                    print("SHAPE", coord[:, 4:6].shape)
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

            if True and i!=0 and not i % 20:
                break

        # Close the pdf file
        pp.close()

        print("Selected Events True nPi0: ", np.unique(events["true_daughter_nPi0"], return_counts=True))
        print("Selected Events True nPi+: ", np.unique(events["true_daughter_nPiPlus"], return_counts=True))
        # Return event selection mask all true (we only wanted the plots)
        return np.ones_like(events["event"])

    def plot_particles_base(self, events, pdg, precut, hists):
        pass

    def efficiency(self, total_events, passed_events, cut, hists):
        pass

    def get_cut_doc(self):
        doc_string = "Plot the shower-like space-points in the event."
        return doc_string
