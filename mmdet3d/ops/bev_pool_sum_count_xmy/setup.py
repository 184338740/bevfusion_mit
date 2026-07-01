from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bev_pool_sum_count_xmy_ext',
    ext_modules=[
        CUDAExtension(
            name='bev_pool_sum_count_xmy_ext',
            sources=[
                'src/bev_pool_sum_count_xmy_cpu.cpp',
                'src/bev_pool_sum_count_xmy_cuda.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_86,code=sm_86',  # 根据你的 GPU 修改，例如 80, 89, 90 等
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)