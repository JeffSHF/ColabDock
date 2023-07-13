from setuptools import setup, find_packages
setup(
    name='colabdock',
    version='1.0.0',
    description='docking with experimental restraints',
    long_description="Inverting AlphaFold2 structure prediction model for protein-protein docking with experimental restraints",
    long_description_content_type='text/markdown',
    packages=find_packages(include=['colabdock']),
    install_requires=['py3Dmol','absl-py','biopython',
                      'chex','dm-haiku','dm-tree',
                      'immutabledict','jax','ml-collections',
                      'numpy','pandas','scipy','optax','joblib',
                      'matplotlib', 'scikit-learn', 'tqdm'],
    include_package_data=True
)