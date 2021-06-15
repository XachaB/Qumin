from setuptools import setup, find_packages


setup(
    name='qumin',
    version='1',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['qumin.H=qumin.calc_paradigm_entropy:H_command',
                            'qumin.patterns=qumin.find_patterns:pat_command',
                            'qumin.macroclasses=qumin.find_macroclasses:macroclasses_command',
                            'qumin.lattice=qumin.make_lattice:lattice_command',
                            "qumin.heatmap=qumin.microclass_heatmap:heatmap_command",
                            "qumin.eval=qumin.eval:eval_command"
                            ],
    },
    url='https://github.com/XachaB/Qumin',
    license='GPLv3',
    author='Sacha Beniamine',
    author_email='',
    description='Qumin: Quantitative Modelling of Inflection',
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'matplotlib',
        'networkx',
        'pydot',
        'tqdm',
        'seaborn',
        'tabulate',
        'concepts',
        'prettytable',
    ],
)
