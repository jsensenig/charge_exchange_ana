import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

'''
 `find_peaks()` modified from 
 photutil package find_peaks() function
 https://github.com/astropy/photutils/tree/1.2.0
 https://photutils.readthedocs.io/en/stable/_modules/photutils/detection/peakfinder.html#find_peaks
'''


class ShowerDirection:

    def __init__(self):
        print("Initialized ShowerDirection class object!")

    def find_peaks(self, data, threshold, box_size):

        data = np.asanyarray(data)

        verbose = False
        if np.all(data == data.flat[0]):
            if verbose:
                print('Input data is constant. No local peaks can be found.')
            return 0, None, None, None

        # Uses scipy.ndimage.maximum_filter1d() function which implements the maximum finding algorithm
        # described here: http://www.richardhartersworld.com/cri/2001/slidingmin.html
        # Paper: Running Max/Min Calculation Using a Pruned Ordered List, Scott C. Douglas, 1996
        data_max = maximum_filter(data, size=box_size, mode='constant', cval=0.0)

        peak_goodmask = (data == data_max)  # good pixels are True
        peak_goodmask &= (data > threshold)

        x_peaks, y_peaks = peak_goodmask.nonzero()
        peak_values = data[x_peaks, y_peaks]

        return len(x_peaks), peak_values, x_peaks, y_peaks

    def get_shower_directions_polar(self, events, show_plot):

        peak_threshold = 10 # should be 9

        hist, xedge, yedge = self.phi_theta_plot(events, show_plot)
        npeaks, _, xpeaks, ypeaks = self.find_peaks(hist, peak_threshold, 3)

        if npeaks < 1:
            return None, None

        verbose = False
        if verbose:
            print("Number of peaks =", npeaks)
            print("θ =", xedge[xpeaks])
            print("ϕ =", yedge[ypeaks])

        return xedge[xpeaks], yedge[ypeaks]

    def get_shower_direction_unit(self, events, show_plot=False):
        """
        Return a vector of Cartesian unit direction vectors corresponding to the showers
        return shape = (Npeaks, 3)
        """
        theta_peak, phi_peak = self.get_shower_directions_polar(events, show_plot)

        if theta_peak is None:
            return np.array([])

        return np.array([np.cos(phi_peak) * np.sin(theta_peak),
                         np.sin(phi_peak) * np.sin(theta_peak),
                         np.cos(theta_peak)]).T

    def transform_to_spherical(self, events):

        spx = "reco_daughter_PFP_shower_spacePts_X"
        spy = "reco_daughter_PFP_shower_spacePts_Y"
        spz = "reco_daughter_PFP_shower_spacePts_Z"
        vtxx = "reco_beam_endX"
        vtxy = "reco_beam_endY"
        vtxz = "reco_beam_endZ"

        if events[vtxx] is None:
            return None

        vertex = np.vstack((ak.to_numpy(events[vtxx]), ak.to_numpy(events[vtxy]), ak.to_numpy(events[vtxz]))).T
        xyz = np.vstack((ak.to_numpy(events[spx]), ak.to_numpy(events[spy]), ak.to_numpy(events[spz]))).T

        if len(xyz) < 1:
            return None

        # Shift origin to beam vertex
        xyz -= vertex

        return self.transform_to_spherical_numpy(xyz)

    @staticmethod
    def transform_to_spherical_numpy(xyz):
                                                                                                                
        if len(xyz) < 1:
            return None
        spherical_points = np.hstack((xyz, np.zeros(xyz.shape)))

        use_detector_coord = True
        if use_detector_coord:
            xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2  # rho
            spherical_points[:, 3] = np.sqrt(xy + xyz[:, 2] ** 2)  # r
            # for elevation angle defined from Z-axis down
            spherical_points[:, 4] = np.arctan2(np.sqrt(xy), xyz[:, 2]) * (360. / (2 * np.pi)) # theta
            spherical_points[:, 5] = np.arctan2(xyz[:, 1], xyz[:, 0]) * (360. / (2 * np.pi))   # phi
        else: # shift z->y and y->-z
            xy = xyz[:, 0] ** 2 + xyz[:, 2] ** 2  # rho
            spherical_points[:, 3] = np.sqrt(xy + xyz[:, 2] ** 2)  # r
            # for elevation angle defined from Z-axis down
            spherical_points[:, 4] = np.arctan2(np.sqrt(xy), -xyz[:, 1]) * (360. / (2 * np.pi)) # theta
            spherical_points[:, 5] = np.arctan2(xyz[:, 2], xyz[:, 0]) * (360. / (2 * np.pi))   # phi
                                                                                                                
        return spherical_points

    def get_theta_phi_bins(self, events):

        angle_resolution = 7. # prev. 7
        theta_colum = 4
        phi_colum = 5

        # if len(events[:, theta_colum]) or len(events[:, phi_colum]):
        #     print("Empty event!")
        #     return None

        # We want 1 bin / 8 degrees
        theta_range = abs(np.min(events[:, theta_colum]) - np.max(events[:, theta_colum]))
        phi_range = abs(np.min(events[:, phi_colum]) - np.max(events[:, phi_colum]))

        nbins_theta = int(theta_range / angle_resolution)
        nbins_phi = int(phi_range / angle_resolution)

        if theta_range <= angle_resolution:
            nbins_theta = 1
        if phi_range <= angle_resolution:
            nbins_phi = 1

        #print("[nbins_theta, nbins_phi] = [", nbins_theta, nbins_phi, "]")

        return [nbins_theta, nbins_phi]

    def phi_theta_plot(self, events, show_plot):

        xy_bins = self.get_theta_phi_bins(events)

        hist, xedge, yedge = np.histogram2d(events[:, 4], events[:, 5], bins=xy_bins)

        if show_plot:
            hist, xedge, yedge, _ = plt.hist2d(events[:, 4], events[:, 5], bins=xy_bins)
            plt.xlabel("$\\theta$ [deg]")
            plt.ylabel("$\phi$ [deg]")
            plt.title("Event ")
            plt.colorbar()
            plt.xlim(0., 180)
            plt.ylim(-180., 180)
            plt.show()

        return hist, xedge, yedge

