from setuptools import setup, find_packages

setup(
    name='traj_analysis',
    version='0.2.0',
    packages=find_packages(include=['traj_analysis']),
    install_requires=['numpy', 'pandas', 'matplotlib', 'seaborn', 'ovito', 'click']
)
