import pytest
from pytest import approx
import importlib.resources
import numpy as np
import traj_analysis
from traj_analysis import TrajStats

# pytest will read the setup and teardown methods to structure a class-specific
# test environment, instantiating TrajStats only one time and checking outputs
class TestClassTrajStats():
# These tests REQUIRE that 'test.xyz' is in the 'traj_analysis' directory
# within the package directories. This first setup method is only called once
# per class and will instantiate TrajStats with the data in 'test.xyz'.
    def setup_class(self):
        rich_atomid = 1
        atomid = 2
        vacid = 3
        try:
            with importlib.resources.path(traj_analysis, 'test.xyz') as p:
                self.test = traj_analysis.TrajStats(str(p), atomid, rich_atomid, vacid)
        except:
            print("Did you put 'test.xyz' inside the package install folder 'traj_analysis'?\n")
            print("All tests require this file to initialize a class (in a test fixture function).\n")

    # check filename, time length, and simulation cell from OVITO
    def test_trajstats_initialization(self):
        correct_cell = np.array([[25.305, 0, 0], [0, 25.305, 0], [0, 0, 25.305]])
        correct_name = 'test.xyz'
        correct_length = 500
        assert all([self.test.filename.split('/')[-1] == correct_name,
                    self.test.timesteps == correct_length,
                    np.all(self.test.cell == correct_cell)])

    # correct number of frames in dictionaries?
    def test_traj_dictionaries_size(self):
        assert all([self.test.timesteps == entry for entry in
                    [len(self.test.trajs), len(self.test.vactrajs), len(self.test.atomtrajs)]])

    # did the code catch all atoms in all frames/keys in the dictionary?
    def test_atom_count_per_frame(self):
        correct_shape = (119, 4)
        assert all([self.test.atomtrajs[i].shape for i in self.test.atomtrajs.keys()])
    # did the code properly reshape the dictionaries into arrays
    def test_atom_array_concat(self):
        correct_shape = (500, 119, 3)
        assert self.test.atomsvstime.shape == correct_shape

    def test_vac_array_concat(self):
        correct_shape = (500, 50, 3)
        assert self.test.vacsvstime.shape == correct_shape

    # size checks for pandas data
    def test_atom_dataframe(self):
        correct_shape = (500, 119)
        assert self.test.df.shape == correct_shape

    def test_vac_dataframe(self):
        correct_shape = (500, 50)
        assert self.test.vacdf.shape == correct_shape

    # top ten variances for this file (with particle IDs) for atoms and vacancies
    def test_top_atom_variances(self):
        res = {84: 0.03165263970468473, 118: 0.03073558520592621, 105: 0.02945724534919901,
        4: 0.027794965156911815, 114: 0.02740776288040494, 19: 0.027351845566151424,
        14: 0.027104000009445666, 110: 0.027046999283903916, 10: 0.02686846871048099,
        88: 0.026447482052576552}
        assert all([self.test.variances[i] == res[i] for i in res.keys()])

    def test_top_vac_variances(self):
        res = {25: 0.013042036172344678, 26: 1.0636184710074083e-26, 27: 1.0636184710074083e-26,
        28: 1.0636184710074083e-26, 29: 1.0636184710074083e-26, 30: 1.0636184710074083e-26,
        31: 1.0636184710074083e-26, 32: 1.0636184710074083e-26, 33: 1.0636184710074083e-26,
        34: 1.0636184710074083e-26}
        assert all(self.test.vacvariances[i] == res[i] for i in res.keys())

    # will need to change hard-coded number as this method develops
    def test_flux_calculation(self):
        correct_flux = 8430.904820655327
        assert approx(self.test.naiveflux(), abs = 1e-3) == correct_flux

    def test_keeping_high_variances(self):
        self.test.keeping(0.03)
        correct_kept = [84, 118]
        assert correct_kept == list(self.test.keeps.keys())

    def test_keeping_high_vacvariances(self):
        self.test.vackeeping(0.01)
        correct_kept = [25]
        assert correct_kept == list(self.test.vackeeps.keys())
