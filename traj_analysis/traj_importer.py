import os
import sys
import numpy as np
from ovito.pipeline import ReferenceConfigurationModifier, FileSource
from ovito.modifiers import WignerSeitzAnalysisModifier, ComputePropertyModifier, InvertSelectionModifier, DeleteSelectedModifier, CombineDatasetsModifier
from ovito.io import import_file, export_file
""" Vacancy Labeler and File I/O
This code is meant for use on a single .xyz trajectory file with a pristine,
crystalline reference configuration (.lmp) to locate vacancies in all frames
of the trajectory based on Voronoi cell binning (Wigner-Seitz Analysis).

Input (requested): Directory path and file name ____.xyz
* FCCreference.lmp must also be present in the working directory*

Output: Two files named '___vacancies.xyz' (1) and 'new___.xyz' (2)
    (1): contains only the labeled vacancies - good for visualization
    (2): contains all particles including vacancies - good for analysis
"""
# custom modifier to select vacancies after Wigner-Seitz analysis
def modify(frame,data):
    occupancies = data.particles['Occupancy']
    total_occupancy = np.sum(occupancies, axis = 1)
    selection = data.particles_.create_property('Selection')
    selection[...] = (total_occupancy == 0)

# vacancylabel_io
def vacancylabel_io(file_path, filename):
    orig_dir = os.getcwd()
    try:
        os.chdir(file_path)
    except:
        print('Couldn''t find that directory, check syntax (/mnt/d/UserName/...)')
        sys.exit(1)
    try:
        pipeline = import_file(filename)
    except:
        print('File not found (check the name and that you''re in the right directory)')
        sys.exit(1)
    wigmod = WignerSeitzAnalysisModifier(per_type_occupancies = True, affine_mapping = ReferenceConfigurationModifier.AffineMapping.ToReference)
    wigmod.reference = FileSource()
    try:
        wigmod.reference.load('FCCreference.lmp')
    except:
        print('FCCreference.lmp was not found!')
        sys.exit(1)
    pipeline.modifiers.append(wigmod)
    pipeline.modifiers.append(ComputePropertyModifier(output_property='Particle Type', expressions = '3'))
    # see custom modifier above
    pipeline.modifiers.append(modify)
    pipeline.modifiers.append(InvertSelectionModifier())
    pipeline.modifiers.append(DeleteSelectedModifier())
    # export a file with the vacancies only (type = 3)
    export_file(pipeline, filename.split('.')[0] + "vacancies.xyz", "xyz", columns = ['Particle Identifier', 'Particle Type', 'Position.X', 'Position.Y', 'Position.Z'], multiple_frames = True)
    # new pipeline to combine the vacancies with the original data (this will append the vacancies at the end of the Particle IDs and reassign new IDs accordingly)
    newpipeline = import_file(filename)
    mod = CombineDatasetsModifier()
    mod.source.load(filename.split('.')[0] + 'vacancies.xyz')
    newpipeline.modifiers.append(mod)
    # make a new file with the vacancies and all atoms included for visualization
    export_file(newpipeline, "new" + filename.split('.')[0] + ".xyz", "xyz", columns = ['Particle Identifier', 'Particle Type', 'Position.X', 'Position.Y', 'Position.Z'], multiple_frames = True )
    os.chdir(orig_dir)
