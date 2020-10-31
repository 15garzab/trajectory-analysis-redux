# Trajectory Analysis
This package uses OVITO, NumPy, and Pandas to read molecular dynamics trajectory
files for solid state systems, providing methods for analysis and comparison between multiple files.
Diffusion calculations of multiple kinds are available as well as cross-correlation heatmaps
to identify atomic mechanisms of bulk mass transfer phenomena.

**IMPORTANT:** In order to run the tests using pytest, you will need the file "test.xyz" inside your cloned package folder. The tests will look for this path:

`/.../{directory where you ran 'git clone ...'}/trajectory-analysis-redux/traj_analysis/test.xyz`

If you would like this file to run the tests, or if you have any questions about the use of this package, please contact me at <rbg18@pitt.edu>
