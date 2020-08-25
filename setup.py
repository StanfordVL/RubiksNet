import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def read_file(fname):
    this_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(this_dir, fname)) as f:
        return f.read()


pkg_name = "rubiksnet"


setup(
    name=pkg_name,
    version="1.0",
    author="Linxi (Jim) Fan*, Shyamal Buch*, Guanzhi Wang, Ryan Cao, "
    "Yuke Zhu, Juan Carlos Niebles, and Li Fei-Fei",
    url="http://github.com/stanfordvl/rubiksnet",
    description="Learnable 3D-Shift for Efficient Video Action Recognition",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    keywords=[
        "Deep Learning",
        "Computer Vision",
        "Video Learning",
        "Action Recognition",
    ],
    license="MIT",
    packages=[package for package in find_packages() if package.startswith(pkg_name)],
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    zip_safe=False,
    ext_modules=[
        CUDAExtension(
            "rubiksnet_cuda",
            [
                "cuda_src/rubiks.cpp",
                "cuda_src/rubiks2d_kernels.cu",
                "cuda_src/rubiks3d_kernels.cu",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
