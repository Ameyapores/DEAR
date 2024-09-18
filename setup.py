from setuptools import find_packages, setup

print(
    "Installing DEAR. Dependencies should already be installed with the provided conda env."
)

setup(
    name="dear",
    version="0.1.0",
    packages=find_packages(),
    description="DEAR: Disentangled Environment and Agent Representations for Reinforcement Learning without Reconstruction",
    author="Ameya Pore, Riccardo Muradore, Diego Dall'Alba",
    author_email="ameya.pore@univr.it",
)
