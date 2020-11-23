import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ovito.io import import_file


def pbc_distance(pos1, pos2, cell, dims = 3):
    """ Find distance between a set of atoms and their image in another timeframe
            - this assumes all particles are wrapped inside the simulation box
            - corrects for image convention if particle wraps to other side
        Parameters
        ---------
        pos1: (n_atoms, dims) ndarray for first set of positions
        pos2: (n_atoms, dims) ndarray for second set
        cell: (3,3) ndarray with simulation cell lengths
        dims (optional): number of dimensions for the calculation
    """
    diff = abs(pos1 - pos2)
    for i in range(dims):
        # np.where neatly vectorizes the if-else logic for a 2D matrix
        diff[:,i] = np.where(2*diff[:,i] > cell[i,i], cell[i,i] - diff[:,i], diff[:,i])
    return diff

def calc_onsagers(disp_list_1, disp_list_2, delta_t, nsamples):
    """ Calculate onsager coefficients for a binary system to yield
        a scalar for the Maxwell-Stefan diffusivity
        Parameters
        -----------
        disp_list_1: list of ndarrays w/ shape (natoms,dims) if directional else (natoms,)
        disp_list_2: list of ndarrays w/ shape (natoms,dims) if directional else (natoms,)
        delta_t: float, time in picoseconds per traj frame
        nsamples: int, number of samples for expectation values
        -----------
    """
    ntotatoms = len(disp_list_1[0]) + len(disp_list_2[0])
    onsager = np.zeros((2,2))
    onsager_std = np.zeros((2,2))
    # off-diagonal
    onsager[0,1] = np.average(np.multiply([np.sum(disp_list_1[i]) for i in range(nsamples)],
                                          [np.sum(disp_list_2[i]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    # symmetry of onsager coefficients
    onsager[1,0] = onsager[0,1]
    # diagonal entries
    onsager[0,0] = np.average(np.square([np.sum(disp_list_1[i]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager[1,1] = np.average(np.square([np.sum(disp_list_2[i]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    # similar process for the error bounds on this measurement
    onsager_std[0,1] = np.std(np.multiply([np.sum(disp_list_1[i]) for i in range(nsamples)],
                                          [np.sum(disp_list_2[i]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_std[1,0] = onsager_std[0,1]
    onsager_std[0,0] = np.std(np.square([np.sum(disp_list_1[i]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_std[1,1] = np.std(np.square([np.sum(disp_list_2[i]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    # now convert values from angstroms^2/picosecond to m^2/s
    scale = 1e-8 # m^2/s in 1 ang^2/ ps
    return (onsager*scale, onsager_std*scale)

def calc_directional_onsagers(disp_list_1, disp_list_2, delta_t, nsamples):
    """ Calculate onsager coefficients for a binary alloy to yield
        vectors of diffusivity in each direction for orientation analysis
        Parameters
        -----------
        disp_list_1: list of ndarrays w/ shape (natoms,dims) if directional else (natoms,)
        disp_list_2: list of ndarrays w/ shape (natoms,dims) if directional else (natoms,)
        delta_t: float, time in picoseconds per traj frame
        nsamples: int, number of samples for expectation values
        -----------
    """
    # scaling factor
    ntotatoms = len(disp_list_1[0][:,0]) + len(disp_list_2[0][:,0])
    # x-direction average
    onsager_xx = np.zeros((2,2))
    onsager_xx[0,0] = np.average(np.square([np.sum(disp_list_1[i][:,0]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_xx[1,1] = np.average(np.square([np.sum(disp_list_2[i][:,0]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_xx[0,1] = np.average(np.multiply([np.sum(disp_list_1[i][:,0]) for i in range(nsamples)], [np.sum(disp_list_2[i][:,0]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_xx[1,0] = onsager_xx[0,1]
    # x-direction
    # y-direction
    onsager_yy = np.zeros((2,2))
    onsager_yy[0,0] = np.average(np.square([np.sum(disp_list_1[i][:,1]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_yy[1,1] = np.average(np.square([np.sum(disp_list_2[i][:,1]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_yy[0,1] = np.average(np.multiply([np.sum(disp_list_1[i][:,1]) for i in range(nsamples)], [np.sum(disp_list_2[i][:,1]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_yy[1,0] = onsager_yy[0,1]
    # z-direction
    onsager_zz = np.zeros((2,2))
    onsager_zz[0,0] = np.average(np.square([np.sum(disp_list_1[i][:,2]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_zz[1,1] = np.average(np.square([np.sum(disp_list_2[i][:,2]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_zz[0,1] = np.average(np.multiply([np.sum(disp_list_1[i][:,2]) for i in range(nsamples)], [np.sum(disp_list_2[i][:,2]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_zz[1,0] = onsager_zz[0,1]
    # off-diagonal terms
    #self.onsager_xy = np.average(np.multiply([np.sum(disp_list_1[i][:,0]) for i in range(nsamples)], [np.sum(vector_lengths[i][:,1]) for i in range(nsamples)]))
    #self.onsager_xz = np.average(np.multiply([np.sum(disp_list_1[i][:,0]) for i in range(nsamples)], [np.sum(disp_list_[i][:,2]) for i in range(nsamples)]))
    #self.onsager_yz = np.average(np.multiply([np.sum(disp_list_1[i][:,1]) for i in range(nsamples)], [np.sum(disp_list_2[i][:,2]) for i in range(nsamples)]))
    # standard deviation of this measurement
    onsager_std_xx = np.zeros((2,2))
    onsager_std_yy = np.zeros((2,2))
    onsager_std_zz = np.zeros((2,2))
    onsager_std_xx[0,0] = np.std(np.square([np.sum(disp_list_1[i][:,0]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_std_xx[1,1] = np.std(np.square([np.sum(disp_list_2[i][:,0]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_std_xx[0,1] = np.std(np.multiply([np.sum(disp_list_1[i][:,0]) for i in range(nsamples)], [np.sum(disp_list_2[i][:,0]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_std_yy[0,0] = np.std(np.square([np.sum(disp_list_1[i][:,1]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_std_yy[1,1] = np.std(np.square([np.sum(disp_list_2[i][:,1]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_std_yy[0,1] = np.std(np.multiply([np.sum(disp_list_1[i][:,1]) for i in range(nsamples)], [np.sum(disp_list_2[i][:,1]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_std_yy[1,0] = onsager_std_yy[0,1]
    onsager_std_zz[0,0] = np.std(np.square([np.sum(disp_list_1[i][:,2]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_std_zz[1,1] = np.std(np.square([np.sum(disp_list_2[i][:,2]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_std_zz[0,1] = np.std(np.multiply([np.sum(disp_list_1[i][:,2]) for i in range(nsamples)], [np.sum(disp_list_2[i][:,2]) for i in range(nsamples)]))/(delta_t*nsamples)/ntotatoms/6
    onsager_std_zz[1,0] = onsager_std_zz[0,1]
    # now convert values from angstroms^2/picosecond to m^2/s
    scale = 1e-8 # m^2/s in 1 ang^2/ ps
    onsager = [entry*scale for entry in [onsager_xx, onsager_yy, onsager_zz]]
    onsager_std = [entry*scale for entry in [onsager_std_xx, onsager_std_yy, onsager_std_zz]]
    return onsager, onsager_std

# general function for pandas dataframe cross-correlation with lag
def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))

class TrajStats():
    """ Trajectory Statistics
    Takes LAMMPS NVT outputs, extracts per atom trajectories, and provides
    several functions to compare them/plot features
    """


    def __init__(self, filename, atomid, rich_atomid, vacid, r = 3.0, parsplice = False):
        """
        Parameters
        ----------
        filename: OVITO readable file such as .lmp or .xyz (ideally this is a preprocessed file by the user)
        atomid: integer label for the dopant atom type in the file
        rich_atomid: integer label for the rich component in the file
        vacid: integer label for the vacancy 'atom type'
        parsplice: boolean flag for different time resolutions
        ----------
        """
        self.filename = filename
        self.atomid = atomid
        self.rich_atomid = rich_atomid
        self.vacid = vacid
        self.r = r
        self.parsplice = parsplice
        self.pipeline = import_file(self.filename, sort_particles = True)
        self.timesteps = self.pipeline.source.num_frames
        data = self.pipeline.compute(0)
        # 3x3 lattice dimensions
        self.cell = np.array(data.cell)[:3,:3]
        # parse this file into a numpy array and pandas dataframe for further study
        # put types and xyz positions into a dictionary
        self.trajs = {}
        self.vactrajs  = {}
        self.atomtrajs = {}
        self.rich_atomtrajs = {}
        for frame_index in range(0, self.pipeline.source.num_frames):
            data = self.pipeline.compute(frame_index)
            pos = np.array(data.particles['Position'])
            types = np.array(data.particles['Particle Type'])
            # must be 2D for np.append
            types = np.reshape(types, (len(types), 1))
            self.trajs[frame_index] = np.append(types, pos, axis = 1)
            # naive vacancy tracking, probably isnt reliable and needs to be refined with pymatgen loop next
            self.vactrajs[frame_index] = self.trajs[frame_index][np.ma.where(self.trajs[frame_index][:,0] == self.vacid)]
            self.atomtrajs[frame_index] = self.trajs[frame_index][np.ma.where(self.trajs[frame_index][:,0] == self.atomid)]
            self.rich_atomtrajs[frame_index] = self.trajs[frame_index][np.ma.where(self.trajs[frame_index][:,0] == self.rich_atomid)]

        # atoms of interest
        self.atomsvstime = np.array([self.atomtrajs[frame][:,1:] for frame in self.atomtrajs.keys()], dtype = float)
        self.rich_atomsvstime = np.array([self.rich_atomtrajs[frame][:,1:] for frame in self.rich_atomtrajs.keys()], dtype = float)
        self.natoms = len(self.atomsvstime[0,:,0])
        self.nrichatoms = len(self.rich_atomsvstime[0,:,0])
        self.ntotatoms = self.natoms + self.nrichatoms
        # vacancies only (nvacs required to smooth nested sequence into the same shapes
        # in case there are 0 lattice vacancies in a frame, or some fluctuating number)
        # This fluctuation happens infrequently and can be fixed by finding the average number beforehand
        self.nvacs = int(np.round(np.average([self.vactrajs[i].shape[0] for i in range(len(self.vactrajs))])))
        # a list comprehension with the above logic
        try:
            self.vacsvstime = np.array([self.vactrajs[frame][:self.nvacs,1:]
                   for frame in self.vactrajs.keys()])
        except:
            print("Vacancy count fluctuates significantly in OVITO WS-tracking, please inspect trajectory file")
            sys.exit(1)#
        # put only the z-coordinate atomic data into a pandas dataframe
        ids = [atom + 1 for atom in range(0, self.natoms)]
        self.df = pd.DataFrame(index = range(0, self.pipeline.source.num_frames))
        for atom_id in ids:
            # z -coordinate only over time for each atom of interest
            self.df[atom_id] = self.atomsvstime[:, atom_id - 1, 2]
        cols = list(self.df)
        # calculate variance of each particle's z-trajectory
        self.variances = {}
        for col in cols:
            self.variances[col] = self.df.var()[col]
        # same process for the atom type in richer phase of the alloy
        ids = [atom + 1 for atom in range(0,self.natoms)]
        self.richdf = pd.DataFrame(index = range(0, self.pipeline.source.num_frames))
        for atom_id in ids:
            self.richdf[atom_id] = self.rich_atomsvstime[:, atom_id - 1, 2]

        cols = list(self.richdf)
        self.richvariances = {}
        for col in cols:
            self.richvariances[col] = self.richdf.var()[col]
        # same process for the vacancies in a dataframe
        ids = [vac + 1 for vac in range(0, self.nvacs)]
        self.vacdf = pd.DataFrame(index = range(0, self.pipeline.source.num_frames))
        for vac_id in ids:
            self.vacdf[vac_id] = self.vacsvstime[:, vac_id - 1, 2]
        cols = list(self.vacdf)
        self.vacvariances = {}
        for col in cols:
            self.vacvariances[col] = self.vacdf.var()[col]

    # personally designed flux measurement (very rough) for slab models w/ a centerline
    def naiveflux(self):
        self.centerline = self.cell[2,2]/2
        self.segregated = []
        segregated = []
        for i in range(len(self.atomsvstime[0,:,0])):
            # average first and last 100 frames for accurate position
            # parsplice trajectories are much smoother, so only 10 frames
            if self.parsplice == False:
                final = np.average(self.atomsvstime[-200:,i,2])
                initial = np.average(self.atomsvstime[:200,i,2])
            elif self.parsplice == True:
                final = np.average(self.atomsvstime[-10:,i,2])
                initial = np.average(self.atomsvstime[:10,i,2])
            # below the centerline, segregation is an increase
            if initial < self.centerline:
                if final - initial > self.r/3:
                    self.segregated.append(i)
            # above the centerline, segregation is a decrease
            elif initial > self.centerline:
                if initial - final > self.r/3:
                    self.segregated.append(i)
        nseg = len(self.segregated)
        # atomic flux in atoms/ang^2/ps (2 ps per 1000 frames and 2A on the slab)
        self.flux = nseg/(2*self.cell[0,0]*self.cell[1,1])/(2*self.timesteps)
        # molar flux in mol/m2/s
        self.flux = self.flux/(1e-20)/(6.02e23)/(1e-12)
        return self.flux

    def msflux(self, delta_t, nsamples, directional = False):
        """
        Calculate Maxwell-Stefan diffusivity coefficients using the
        Onsager reciprocal relations and a measurement of MSD
        Parameters
        ===============
        delta_t: Timestep (in picoseconds) per lammps dump file
            *determined by LAMMPS dump frequency and internal timestep*

        nsamples: Number of samples for MSD and vector displacement averaging

        directional:  Boolean flag for diffusion tensor
        ===============
        """
        self.delta_t = delta_t
        self.nsamples = nsamples
        self.directional = directional
        self.binsize = int(self.timesteps/self.nsamples)
        self.binsize = int(self.timesteps/self.nsamples)
        #### these should all have len = nsamples
        # list for MSD samples
        self.disp_magnitudes = []
        self.rich_disp_magnitudes = []
        self.vac_disp_magnitudes = []
        # lists for vectors of length 3 (samples for x y z flux)
        self.vector_lengths = []
        self.rich_vector_lengths = []
        self.vac_vector_lengths = []
        # raw differences (mostly stored for debugging)
        self.diffs = []
        self.rich_diffs = []
        self.vac_diffs = []
        # sliding window displacement collection
        for frame in range(self.binsize, self.timesteps + self.binsize, self.binsize):
            # PBC distance for all samples (assumes particles are all wrapped into simulation box)
            self.diffs.append(pbc_distance(self.atomsvstime[frame - self.binsize, :, :], self.atomsvstime[frame - 1, :, :], self.cell))
            self.rich_diffs.append(pbc_distance(self.rich_atomsvstime[frame - self.binsize, :, :], self.rich_atomsvstime[frame - 1,:,:], self.cell))
            self.vac_diffs.append(pbc_distance(self.vacsvstime[frame - self.binsize, :, :], self.vacsvstime[frame - 1,:,:], self.cell))
            # Euclidean norms (displacement)
            self.disp_magnitudes.append(np.linalg.norm(self.diffs[-1], axis = 1))
            self.rich_disp_magnitudes.append(np.linalg.norm(self.rich_diffs[-1], axis = 1))
            self.vac_disp_magnitudes.append(np.linalg.norm(self.vac_diffs[-1], axis = 1))
            #  Vector magnitudes (for directional flux)
            self.vector_lengths.append(self.diffs[-1])
            self.rich_vector_lengths.append(self.rich_diffs[-1])
            self.vac_vector_lengths.append(self.vac_diffs[-1])
        # using displacement magnitude (cartesian distance) to get a scalar value for D
            ## ONSAGER MATRIX ##
            # onsager[0,0] = Rich element (Cu in current study)
            # onsager[1,1] = Dilute element (Ni in current study)
            # onsager[0,1] = onsager[1,0] = Off-diagonal terms (symmetric)
        self.onsager, self.onsager_std = calc_onsagers(self.rich_disp_magnitudes, self.disp_magnitudes, self.delta_t, self.nsamples)
        # Concentrations
        X_rich = self.nrichatoms/self.ntotatoms
        X_dilute = self.natoms/self.ntotatoms
        # Maxwell-Stefan diffusivity for binary mixture
        self.diff = (X_dilute/X_rich)*self.onsager[0,0] + (X_rich/X_dilute)*self.onsager[1,1] - 2*self.onsager[0,1]
        self.diff_upper = (X_dilute/X_rich)*(self.onsager[0,0]+self.onsager_std[0,0]) + (X_rich/X_dilute)*(self.onsager[1,1]+self.onsager_std[1,1]) - 2*(self.onsager[0,1]+self.onsager_std[0,1])
        self.diff_lower = (X_dilute/X_rich)*(self.onsager[0,0]-self.onsager_std[0,0]) + (X_rich/X_dilute)*(self.onsager[1,1]-self.onsager_std[1,1]) - 2*(self.onsager[0,1]-self.onsager_std[0,1])
        print('Diffusivity is ' + str(self.diff) + '\n')
        print('Upper Confidence bound: ' + str(self.diff_upper) + '\n')
        print('Lower Confidence bound: ' + str(self.diff_lower) + '\n')
        self.diffusivity = {'value': self.diff, 'upper': self.diff_upper, 'lower': self.diff_lower}
        # using the vector magnitudes in each direction to calculate a tensor if requested
        if directional == True:
            self.onsager_direct, self.onsager_direct_std = calc_directional_onsagers(self.rich_vector_lengths, self.vector_lengths, self.delta_t, self.nsamples)
            # diagonal elements of diffusion matrix
            self.diff_xx = (X_dilute/X_rich)*self.onsager_direct[0][0,0] + (X_rich/X_dilute)*self.onsager_direct[0][1,1] - 2*self.onsager_direct[0][0,1]
            self.diff_yy = (X_dilute/X_rich)*self.onsager_direct[1][0,0] + (X_rich/X_dilute)*self.onsager_direct[1][1,1] - 2*self.onsager_direct[1][0,1]
            self.diff_zz = (X_dilute/X_rich)*self.onsager_direct[2][0,0] + (X_rich/X_dilute)*self.onsager_direct[2][1,1] - 2*self.onsager_direct[2][0,1]
            # magnitude of these
            self.diff_directional = np.sqrt(self.diff_xx**2 + self.diff_yy**2 + self.diff_zz**2)
        return self.diffusivity
    # keep variances above 0.1 threshold
    def keeping(self, threshold):
        self.keeps = {}
        for key in self.variances.keys():
            # only relatively high variances are important
            if self.variances[key] > threshold:
                self.keeps[key] = self.df[key]
        return self.keeps

    def vackeeping(self,threshold):
        self.vackeeps = {}
        for key in self.vacvariances.keys():
            if self.vacvariances[key] > threshold:
                self.vackeeps[key] = self.vacdf[key]
        return self.vackeeps

    # plot a sampling of the trajectories over time
    def sample_ztraj(self, n):
        samples = self.df.sample(n, axis = 1)
        legend = []
        for col in list(samples):
            plt.plot(list(range(self.pipeline.source.num_frames)), self.df[col])
            legend.append(col)

        plt.legend(legend, loc = 'upper right')
        plt.show()
        return None

    def sample_vacs_ztraj(self,n):
        samples = self.vacdf.sample(n, axis = 1)
        legend = []
        for col in list(samples):
            plt.plot(list(range(self.pipeline.source.num_frames)), self.df[col])
            legend.append(col)
        plt.legend(legend, loc = 'upper right')
        plt.show()
        return None

    def plot_variances(self):
        # initial z position on x axis and variance on y axis
        plt.plot([self.df[col][0] for col in self.cols], list(self.variances.values()), 'o')
        plt.show()
        return None

    def thresh_variance(self):
        leg_list = []
        # plot the trajectories that remain after filtering
        for key in self.keeps.keys():
            plt.plot(np.arange(1, self.pipeline.source.num_frames + 1), self.keeps[key])
            leg_list.append(key)
        plt.legend(leg_list, loc = 'upper right')
        plt.title('Nickel Trajectories w/ High Variance')
        plt.xlabel('Timestep')
        plt.ylabel('Z-Coordinate')

    def thresh_vacvariance(self):
        leg_list = []
        for key in self.vackeeps.keys():
            plt.plot(np.arange(1, self.pipeline.source.num_frames + 1), self.vackeeps[key])
            leg_list.append(key)
        plt.legend(leg_list, loc ='upper right')
        plt.title('Vacancy Trajectories w/ High Variance')
        plt.xlabel('Timestep')
        plt.ylabel('Z-Coordinate')


    # raw cross correlation
    def cross(self, atomid1, atomid2, differenced = False):
        if differenced:
            d1 = self.diffdf[atomid1]
            d2 = self.diffdf[atomid2]
        else:
            d1 = self.df[atomid1]
            d2 = self.df[atomid2]
        seconds = 5
        fps = 10
        rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
        offset = np.ceil(len(rs)/2)-np.argmax(rs)
        f,ax=plt.subplots(figsize=(14,3))
        ax.plot(rs)
        ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
        ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
        ax.set(title=f'Offset = {offset} frames\n Atom 1 leads <> Atom 2 leads', xlabel='Offset',ylabel='Pearson r')
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels([-50, -25, 0, 25, 50])
        plt.legend()
        plt.show()
        return None

    # Windowed, time-lagged cross correlation
    def windowedcross(self, atomid1, atomid2, differenced = False):
        no_splits = 20
        samples_per_split = thousand.df.shape[0]/no_splits
        rss=[]
        seconds = 5
        fps = 10
        if differenced:
            for t in range(0, no_splits):
                d1 = self.diffdf[atomid1].loc[(t)*samples_per_split:(t+1)*samples_per_split]
                d2 = self.diffdf[atomid2].loc[(t)*samples_per_split:(t+1)*samples_per_split]
                rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
                rss.append(rs)
        else:
            for t in range(0, no_splits):
                d1 = self.df[atomid1].loc[(t)*samples_per_split:(t+1)*samples_per_split]
                d2 = self.df[atomid2].loc[(t)*samples_per_split:(t+1)*samples_per_split]
                rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
                rss.append(rs)
        rss = pd.DataFrame(rss)
        f,ax = plt.subplots()
        sns.heatmap(rss,cmap='coolwarm',ax=ax)
        ax.set(title=f'Windowed Time-Lagged Cross Correlation',xlabel='Offset',ylabel='Window epochs')
        #ax.set_xlim(85, 215)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels([-50, -25, 0, 25, 50])
        plt.show()
        return None

    # Rolling window, time-lagged cross correlation
    def rollingcross(self, atomid1, atomid2, differenced = False):
        seconds = 5
        fps = 10
        window_size = 300 #samples, should be a pretty high number compared to fps*sec to get good rolling averages
        t_start = 0
        t_end = t_start + window_size
        step_size = 20
        rss=[]
        while t_end < self.pipeline.source.num_frames:
            if differenced:
                d1 = self.diffdf[atomid1].iloc[t_start:t_end]
                d2 = self.diffdf[atomid2].iloc[t_start:t_end]
            else:
                d1 = self.df[atomid1].iloc[t_start:t_end]
                d2 = self.df[atomid2].iloc[t_start:t_end]
            rs = [crosscorr(d1,d2, lag, wrap=False) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
            rss.append(rs)
            t_start = t_start + step_size
            t_end = t_end + step_size
        rss = pd.DataFrame(rss)

        f,ax = plt.subplots()
        sns.heatmap(rss,cmap='coolwarm',ax=ax)
        ax.set(title=f'Rolling-Window Time-Lagged Cross Correlation', xlabel='Offset',ylabel='Epochs')
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels([-50, -25, 0, 25, 50])
        plt.show()
        return None

        # Rolling window, time-lagged cross correlation for dataframes in different classes
    def rollingcrossvacs(atomid1, atomid2, differenced = False):
        seconds = 5
        fps = 10
        window_size = 300 #samples, should be a pretty high number compared to fps*sec to get good rolling averages
        t_start = 0
        t_end = t_start + window_size
        step_size = 20
        rss=[]
        while t_end < self.pipeline.source.num_frames:
            if differenced:
                d1 = self.diffdf[atomid1].iloc[t_start:t_end]
                d2 = self.vacdiffdf[atomid2].iloc[t_start:t_end]
            else:
                d1 = self.df[atomid1].iloc[t_start:t_end]
                d2 = self.vacdf[atomid2].iloc[t_start:t_end]
            rs = [crosscorr(d1,d2, lag, wrap=False) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
            rss.append(rs)
            t_start = t_start + step_size
            t_end = t_end + step_size
        rss = pd.DataFrame(rss)

        f,ax = plt.subplots()
        sns.heatmap(rss,cmap='coolwarm',ax=ax)
        ax.set(title=f'Rolling-Window Time-Lagged Cross Correlation', xlabel='Offset',ylabel='Epochs')
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels([-50, -25, 0, 25, 50])
        plt.show()
        return None
