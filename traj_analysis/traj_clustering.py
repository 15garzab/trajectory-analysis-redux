import sys
import numpy as np
import matplotlib.pyplot as plt
from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier
from scipy.signal import argrelextrema

class ClusterStats():
    """ Cluster Statistics
    Takes LAMMPS MC/NVT data, calculating the time-averaged partial radial
    distribution functions (partial RDFs) for all the bond types in the system."""

    def __init__(self, filename, atomid, rich_atomid, vacid, parsplice = False):
        """
        Parameters:
        ___________
        filename: OVITO readable file such as .lmp or .xyz (ideally preprocessed by the user)
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
        self.parsplice = parsplice
        self.pipeline = import_file(filename, sort_particles = True)

    # calculate distribution functions with OVITO
    def calc_rdfs(self, cutoff):
        self.cutoff = cutoff
        # Modifier to calculate distribution functions
        if self.parsplice == False:
            modifier = CoordinationAnalysisModifier(cutoff = self.cutoff, number_of_bins = 500, partial = True)
        else:
            modifier = CoordinationAnalysisModifier(cutoff = self.cutoff, number_of_bins = 1000, partial = True)
        self.pipeline.modifiers.append(modifier)
        # storage for all pdfs in all frames
        self.total_pdf = np.zeros((modifier.number_of_bins,7))
    # time-averaging method from OVITO documentation
        for frame in range(self.pipeline.source.num_frames):
            data = self.pipeline.compute(frame)
            self.total_pdf += data.tables['coordination-rdf'].xy()

        self.total_pdf /= self.pipeline.source.num_frames

        return self.total_pdf

    # plot the distributions
    def plot_rdfs(self, rich = 'Cu', dilute = 'Ni'):
        fig, axs = plt.subplots(1,3, sharex = True, sharey = True)
        # different order of data indexing depending on the order of atom types
        if self.rich_atomid == 1 and self.atomid == 2:
            # rich atomid first
            axs[0].plot(self.total_pdf[:,0], self.total_pdf[:, 1])
            axs[0].set_title(rich + '-' + rich + ' bond')
            # heterobond
            axs[1].plot(self.total_pdf[:,0], self.total_pdf[:,4])
            axs[1].set_title(rich + '-' + dilute + ' bond')
            # atomid
            axs[2].plot(self.total_pdf[:,0], self.total_pdf[:,5])
            axs[2].set_title(dilute + '-' + dilute + ' bond')
        elif self.atomid == 1 and self.rich_atomid == 2:
            # atomid first in this case
            axs[0].plot(self.total_pdf[:,0], self.total_pdf[:,1])
            axs[0].set_title(dilute + '-' + dilute + ' bond')
            # heterobond
            axs[1].plot(self.total_pdf[:,0], self.total_pdf[:,4])
            axs[1].set_title(dilute + '-' + rich + ' bond')
            # rich_atomid
            axs[2].plot(selt.total_pdf[:,0], self.total_pdf[:,5])
            axs[2].set_title(rich + '-' + rich + ' bond')
        else:
            print("Error: Inconsistent particle type IDs")
            sys.exit(1)
        # for pretty labeling, make a large, invisible subplot with the axis labels only
        fig.add_subplot(111, frameon = False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.title('Average Partial Distribution Functions\n')
        plt.xlabel('Pair Separation Distance (angstroms)')
        plt.ylabel('Average g(r)')

    # plot the RDFs with respect to the lattice vacancies
    def plot_vacrdfs(self, rich = 'Cu', dilute = 'Ni'):
        if self.vacid == 3:
            if self.rich_atomid == 1 and self.atomid == 2:
                fig, axs = plt.subplots(1,3, sharex = True, sharey = True)
                # rich_atomid
                axs[0].plot(self.total_pdf[:,0], self.total_pdf[:,2])
                axs[0].set_title(rich + '-' + 'vacancy distance')
                # atomid
                axs[1].plot(self.total_pdf[:,0], self.total_pdf[:,3])
                axs[1].set_title(dilute + '-' + 'vacancy distance')
                # Vacancy-vacancy distance
                axs[2].plot(self.total_pdf[:,0], self.total_pdf[:,6])
                axs[2].set_title('Vacancy-vacancy distance')
            elif self.atomid == 1 and self.rich_atomid == 2:
                fig, axs = plt.subplots(1,3, sharex = True, sharey = True)
                # atomid
                axs[0].plot(self.total_pdf[:,0], self.total_pdf[:,2])
                axs[0].set_title(dilute + '-' + 'vacancy distance')
                # rich_atomid
                axs[1].plot(self.total_pdf[:,0], self.total_pdf[:,3])
                axs[1].set_title(rich + '-' + 'vacancy distance')
                # vacancy-vacancy
                axs[2].plot(self.total_pdf[:,0], self.total_pdf[:,6])
                axs[2].set_title('Vacancy-vacancy distance')
            else:
                print('Error: Inconsistent particle type IDs')
                sys.exit(1)
        else:
            print("Error: Vacid must be equal to 3")
            sys.exit(1)

    # report local maxima in the RDFs
    def bond_averages(self):
        # scipy function for the local maxima in the distribution
        # check if point is greater than 30 points in either direction
        # heterobond
        self.het_maxima = self.total_pdf[:,0][argrelextrema(self.total_pdf[:,4], np.greater, order = 30)[0]]
        if self.rich_atomid == 1 and self.atomid == 2:

            self.rich_maxima = self.total_pdf[:,0][argrelextrema(self.total_pdf[:,1], np.greater, order = 30)[0]]
            self.maxima = self.total_pdf[:,0][argrelextrema(self.total_pdf[:,5], np.greater, order = 30)[0]]
        elif self.atomid == 1 and self.rich_atomid == 2:
            self.maxima = self.total_pdf[:,0][argrelextrema(self.total_pdf[:,1], np.greater,  order = 30)[0]]
            self.rich_maxima = self.total_pdf[:,0][argrelextrema(self.total_pdf[:,5], np.greater, order = 30)[0]]
        else:
            print("Error: Inconsistent particle type IDs")
            sys.exit(1)

        return [self.rich_maxima, self.het_maxima, self.maxima]
