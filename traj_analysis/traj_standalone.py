import os
import sys
import random
from operator import itemgetter
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

# fastest numpy shift w/ NaN fill: preallocate empty array and assign slice by chrisaycock (StackOverflow)
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

# general function for pandas dataframe cross-correlation with lag
def np_crosscorr(datax, datay, lag=0, wrap=False):
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
    datax = pd.Series(datax)
    datay = pd.Series(datay)
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return pd.to_numeric(datax.corr(datay.shift(lag)))

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


    def __init__(self, filename, nvacs, atomid, rich_atomid, vacid, parsplice = False):
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
        self.nvacs = nvacs
        self.atomid = atomid
        self.rich_atomid = rich_atomid
        self.vacid = vacid
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

        # atoms of interest xyz
        self.atomsvstime = np.array([self.atomtrajs[frame][:,1:] for frame in self.atomtrajs.keys()], dtype = float)
        # polar coordinates of the dopant atoms
        self.rvstime = np.sqrt(np.square(self.atomsvstime[:,:,0]) + np.square(self.atomsvstime[:,:,1]))
        self.thetavstime = np.arccos(self.atomsvstime[:,:,0]/self.rvstime)
        # solvent atoms xyz required too
        self.rich_atomsvstime = np.array([self.rich_atomtrajs[frame][:,1:] for frame in self.rich_atomtrajs.keys()], dtype = float)
        # polar coords for the solvent atoms
        self.rich_rvstime = np.sqrt(np.square(self.rich_atomsvstime[:,:,0]) + np.square(self.rich_atomsvstime[:,:,1]))
        self.rich_thetavstime = np.arccos(self.rich_atomsvstime[:,:,0]/self.rich_rvstime)
        self.natoms = len(self.atomsvstime[0,:,0])
        self.nrichatoms = len(self.rich_atomsvstime[0,:,0])
        self.ntotatoms = self.natoms + self.nrichatoms
        # in case there are 0 lattice vacancies in a frame, or some fluctuating number)
        # This fluctuation happens infrequently and can be fixed by propagating the previous frame
        # forward in time starting from the initial count (in the very first frame of the trajectory)
        #self.nvacs = int(np.round(np.average([self.vactrajs[i].shape[0] for i in range(len(self.vactrajs))])))
        # a list comprehension with the above logic
        if self.nvacs < 3:
            try:
                self.vacsvstime = np.array([self.vactrajs[frame][:self.nvacs,1:]
                    for frame in self.vactrajs.keys()])
            except:
                print("Vacancy count fluctuates significantly in OVITO WS-tracking, please inspect trajectory file")
                sys.exit(1)#
        else:
            pass
        # polar coordinates for the vacancies
        self.vacs_rvstime = np.sqrt(np.square(self.vacsvstime[:,:,0]) + np.square(self.vacsvstime[:,:,1]))
        self.vacs_thetavstime = np.arccos(self.vacsvstime[:,:,0]/self.vacs_rvstime)
        # calculate variance of each particle's z-trajectory
        self.variances = {atom_id: np.var(self.atomsvstime[:, atom_id, 2]) for atom_id in range(0,self.natoms)}
        self.rich_variances = {atom_id: np.var(self.rich_atomsvstime[:, atom_id, 2]) for atom_id in range(0,self.nrichatoms)}
        self.vacvariances = {vac_id: np.var(self.vacsvstime[:, vac_id, 2]) for vac_id in range(0,self.nvacs)}

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
            elif self.parpslice == True:
                final = np.average(self.atomsvstime[-10:,i,2])
                initial = np.average(self.atomsvstime[:10,i,2])
            # below the centerline, segregation is an increase
            if initial < self.centerline:
                if final - initial > self.r/2:
                    self.segregated.append(i)
            # above the centerline, segregation is a decrease
            elif initial > self.centerline:
                if initial - final > self.r/2:
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
    
    def top_nvars(self,ntop):
        res = dict(sorted(self.variances.items(), key = itemgetter(1), reverse = True)[:ntop])
        print("The top " + str(ntop) + " variances for this trajectory are "  + str(res) + '\n')
        return res

    def top_nvacvars(self,ntop):
        res = dict(sorted(self.vacvariances.items(), key = itemgetter(1), reverse = True)[:ntop])
        print("The top " + str(ntop) + " variances for this trajectory are "  + str(res) + '\n')
        return res

    def top_nrichvars(self,ntop):
        res = dict(sorted(self.rich_variances.items(), key = itemgetter(1), reverse = True)[:ntop])
        print("The top " + str(ntop) + " variances for this trajectory are "  + str(res) + '\n')
        return res

    # keep variances above 0.1 threshold
    def keeping(self, threshold):
        self.keeps = {}
        for key in self.variances.keys():
            # only relatively high variances are important
            if self.variances[key] > threshold:
                self.keeps[key] = self.atomsvstime[:,key,2]
        return self.keeps

    def vackeeping(self,threshold):
        self.vackeeps = {}
        for key in self.vacvariances.keys():
            if self.vacvariances[key] > threshold:
                self.vackeeps[key] = self.vacsvstime[:,key,2]
        return self.vackeeps

    def richkeeping(self,threshold):
        self.richkeeps = {}
        for key in self.rich_variances.keys():
            if self.rich_variances[key] > threshold:
                self.richkeeps[key] = self.rich_atomsvstime[:,key,2]
        return self.richkeeps

    def thresh_variance(self):
        leg_list = []
        # plot the trajectories that remain after filtering
        for key in self.keeps.keys():
            plt.plot(np.arange(1, self.pipeline.source.num_frames + 1), self.keeps[key])
            leg_list.append(key)
        plt.legend(leg_list, loc = 'upper right')
        plt.title('Solute Atom Trajectories w/ High Variance')
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

    def thresh_richvariance(self):
        leg_list = []
        for key in self.richkeeps.keys():
            plt.plot(np.arange(1, self.pipeline.source.num_frames + 1), self.richkeeps[key])
            leg_list.append(key)
        plt.legend(leg_list, loc ='upper right')
        plt.title('Solvent Atom Trajectories w/ High Variance')
        plt.xlabel('Timestep')
        plt.ylabel('Z-Coordinate')

    def analysis_routine(self, ntop):
        # print the top ten particles in order of highest to lowest variance
        top_nvars = self.top_nvars(ntop=ntop)
        top_nvacvars = self.top_nvacvars(ntop=ntop)
        top_nrichvars = self.top_nrichvars(ntop=ntop)
        # keep these trajectories for plotting 
        self.keeping(min(top_nvars.values()) - (10**-8))
        self.vackeeping(min(top_nvacvars.values()) - (10**-8))
        self.richkeeping(min(top_nrichvars.values()) - (10**-8))
        # plot vacancies only
        self.thresh_vacvariance()
        return None

    # Rolling window, time-lagged cross correlation
    def rollingcross(self, atomid1, atomid2, coord, start=0, subtract=0, rich = False, differenced = False):
        # coord takes values {0: r, 1: theta, 2: z} 
        seconds = 5
        fps = 10
        window_size = 300 #samples, should be a pretty high number compared to fps*sec to get good rolling averages
        t_start = start
        t_end = t_start + window_size
        step_size = 20
        rss=[]
        while t_end < self.pipeline.source.num_frames-subtract:
            if rich == False:
                if coord == 2:
                    d1 = self.atomsvstime[t_start:t_end, atomid1, coord]
                    d2 = self.atomsvstime[t_start:t_end, atomid2, coord]
                elif coord == 0:
                    d1 = self.rvstime[t_start:t_end, atomid1]
                    d2 = self.rvstime[t_start:t_end, atomid2]
                elif coord == 1:
                    d1 = self.thetavstime[t_start:t_end, atomid1]
                    d2 = self.thetavstime[t_start:t_end, atomid2]
            else:
                if coord == 2:
                    d1 = self.atomsvstime[t_start:t_end, atomid1, coord]
                    d2 = self.rich_atomsvstime[t_start:t_end, atomid2, coord]
                elif coord == 0:
                    d1 = self.rvstime[t_start:t_end, atomid1]
                    d2 = self.rich_rvstime[t_start:t_end, atomid2]
                elif coord == 1:
                    d1 = self.thetavstime[t_start:t_end, atomid1]
                    d2 = self.rich_thetavstime[t_start:t_end, atomid2]
            rs = [np_crosscorr(d1,d2, lag, wrap=False) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
            rss.append(rs)
            t_start = t_start + step_size
            t_end = t_end + step_size
        rss = np.stack(rss)
        # any NaN values are converted to one (implying a 1:1 match i.e. no difference)
        rss = np.nan_to_num(rss, nan=1)
        # remove extra dim
        rss = np.squeeze(rss)
        f,ax = plt.subplots()
        sns.heatmap(rss,cmap='coolwarm',ax=ax)
        ax.set(title=f'Rolling-Window Time-Lagged Cross Correlation', xlabel='Offset',ylabel='Epochs')
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels([-50, -25, 0, 25, 50])
        plt.show()
        return None

        # Rolling window, time-lagged cross correlation for dataframes in different classes
    def rollingcrossvacs(self, atomid1, atomid2, coord, start=0, subtract=0, rich = False, differenced = False):
        seconds = 5
        fps = 10
        window_size = 300 #samples, should be a pretty high number compared to fps*sec to get good rolling averages
        t_start = start
        t_end = t_start + window_size
        step_size = 20
        rss=[]
        # remove the final x amount of frames
        while t_end < self.pipeline.source.num_frames-subtract:
            if rich == False:
                if coord == 2:
                    d1 = self.atomsvstime[t_start:t_end, atomid1, coord]
                    d2 = self.vacsvstime[t_start:t_end, atomid2, coord]
                elif coord == 0:
                    d1 = self.rvstime[t_start:t_end, atomid1]
                    d2 = self.vacs_rvstime[t_start:t_end, atomid2]
                elif coord == 1:
                    d1 = self.thetavstime[t_start:t_end, atomid1]
                    d2 = self.vacs_thetavstime[t_start:t_end, atomid2]
            else:
                if coord == 2:
                    d1 = self.rich_atomsvstime[t_start:t_end, atomid1, coord]
                    d2 = self.vacsvstime[t_start:t_end, atomid2, coord]
                elif coord == 0:
                    d1 = self.rich_rvstime[t_start:t_end, atomid1]
                    d2 = self.vacs_rvstime[t_start:t_end, atomid2]
                elif coord == 1:
                    d1 = self.rich_thetavstime[t_start:t_end, atomid1]
                    d2 = self.vacs_thetavstime[t_start:t_end, atomid2]
            rs = [np_crosscorr(d1,d2, lag, wrap=False) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
            rss.append(rs)
            t_start = t_start + step_size
            t_end = t_end + step_size
        rss = np.stack(rss)
        # any NaN values are converted to one (implying a 1:1 match i.e. no difference)
        rss = np.nan_to_num(rss, nan=1)
        f,ax = plt.subplots()
        sns.heatmap(rss,cmap='coolwarm',ax=ax)
        ax.set(title=f'Rolling-Window Time-Lagged Cross Correlation', xlabel='Offset',ylabel='Epochs')
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels([-50, -25, 0, 25, 50])
        plt.show()
        return None
    
    # comparing vacancy trajectories
    def rollingcrossvac2vac(self, vacid1, vacid2, coord, start=0, subtract=0, differenced = False):
        seconds = 5
        fps = 10
        window_size = 300 #samples, should be a pretty high number compared to fps*sec to get good rolling averages
        t_start = start
        t_end = t_start + window_size
        step_size = 20
        rss=[]
        # remove the final x amount of frames
        while t_end < self.pipeline.source.num_frames-subtract:
            # z, vertical coordinate
            if coord == 2:
                d1 = self.vacsvstime[t_start:t_end, vacid1, coord]
                d2 = self.vacsvstime[t_start:t_end, vacid2, coord]
            # r, radial coordinate
            elif coord == 0:
                d1 = self.vacs_rvstime[t_start:t_end, vacid1]
                d2 = self.vacs_rvstime[t_start:t_end, vacid2]
            elif coord == 1:
                d1 = self.vacs_thetavstime[t_start:t_end, vacid1]
                d2 = self.vacs_thetavstime[t_start:t_end, vacid2]
            rs = [np_crosscorr(d1,d2, lag, wrap=False) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
            rss.append(rs)
            t_start = t_start + step_size
            t_end = t_end + step_size
        rss = np.stack(rss)
        # any NaN values are converted to one (implying a 1:1 match i.e. no difference)
        rss = np.nan_to_num(rss, nan=1)
        f,ax = plt.subplots()
        sns.heatmap(rss,cmap='coolwarm',ax=ax)
        ax.set(title=f'Rolling-Window Time-Lagged Cross Correlation', xlabel='Offset',ylabel='Epochs')
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels([-50, -25, 0, 25, 50])
        plt.show()
        return None

    #def mean_detection(self, atomid):
