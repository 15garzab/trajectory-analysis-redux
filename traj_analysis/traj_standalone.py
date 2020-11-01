import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ovito.io import import_file


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


    def __init__(self, filename, atomid, vacid, r = 3.0, parsplice = False):
        """
        Parameters
        ----------
        filename: OVITO readable file such as .lmp or .xyz (ideally this is a preprocessed file by the user)
        atomid: integer label for the dopant atom type in the file
        vacid: integer label for the vacancy 'atom type'
        r: cut-off radius for find_points_in_spheres (pymatgen)
        ----------
        """
        self.filename = filename
        self.atomid = atomid
        self.vacid = vacid
        self.parsplice = parsplice
        self.pipeline = import_file(self.filename, sort_particles = True)
        self.timesteps = self.pipeline.source.num_frames
        data = self.pipeline.compute(0)
        # parameters for pymatgen's find_points_in_spheres
        self.r = r
        # 3x3 lattice dimensions
        self.cell = np.array(data.cell).copy(order = 'C').astype('double')[:3,:3]
        self.a = np.array([1, 1, 1]).copy(order = 'C')
        # parse this file into a numpy array and pandas dataframe for further study
        # put types and xyz positions into a dictionary
        self.trajs = {}
        self.vactrajs  = {}
        self.atomtrajs = {}
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

        # atoms of interest
        self.atomsvstime = np.array([self.atomtrajs[frame][:,1:] for frame in self.atomtrajs.keys()], dtype = float)
        self.natoms = len(self.atomsvstime[0,:,0])
        # vacancies only (nvacs required to smooth nested sequence into the same shapes
        # in case there are 0 lattice vacancies in a frame, or some fluctuating number)
        # This fluctuation happens infrequently and can be fixed by propagating the previous frame
        # forward in time starting from the initial count (in the very first frame of the trajectory)
        self.nvacs = len(self.vactrajs[0][:,0])
        # a list comprehension with the above logic
        try:
            self.vacsvstime = np.array([self.vactrajs[frame][:,1:] if self.vactrajs[frame][:,1:].shape == (self.nvacs,3)
                               else self.vactrajs[frame-1][:self.nvacs,1:]
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
        self.cols = list(self.df)
        # calculate variance of each particle's z-trajectory
        self.variances = {}
        for col in self.cols:
            self.variances[col] = self.df.var()[col]
        # same process for the vacancies in a dataframe
        ids = [vac + 1 for vac in range(0, self.nvacs)]
        self.vacdf = pd.DataFrame(index = range(0, self.pipeline.source.num_frames))
        for vac_id in ids:
            self.vacdf[vac_id] = self.vacsvstime[:, vac_id - 1, 2]
        self.vaccols = list(self.vacdf)
        self.vacvariances = {}
        for col in self.vaccols:
            self.vacvariances[col] = self.vacdf.var()[col]

    def Naiveflux(self):
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

    def MBflux(self, delta_t, nsamples = 20):
        self.delta_t = delta_t
        self.binsize = int(self.timesteps/self.nsamples)
        for frame in range(self.binsize, self.timesteps + self.binsize, self.binsize):
                frame = np.zeros((1,1), dtype = int)
        return None
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
