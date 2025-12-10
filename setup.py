from setuptools import find_packages, setup

setup(
    name="p2m",
    version="0.0.1",
    packages=find_packages("."),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "wandb",
        "imageio",
        "plotly",
        "einops",
        "pandas",
        "moviepy",
        "av",
        "torchrl==0.3.1",
    ],
)
