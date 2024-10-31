#!/usr/bin/env python

import os
import glob

import torch

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension
from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        
        # 基本 CUDA 编译选项
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        # 使用更低的计算能力版本
        extra_compile_args["nvcc"].extend([
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_86,code=compute_86",
        ])

        # 额外的编译选项
        extra_compile_args["nvcc"].extend([
            "--expt-relaxed-constexpr",
            "-O2",
            "--ptxas-options=-v",
            "--gpu-architecture=compute_86",
            "--gpu-code=sm_86",
            "-maxrregcount=128"
        ])

        # 添加 C++ 编译选项
        extra_compile_args["cxx"] = ["-O2"]
    else:
        raise NotImplementedError('Cuda is not available')

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    author="Weijie Su",
    url="xxx",
    description="Multi-Scale Deformable Attention Module in Deformable DETR",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)