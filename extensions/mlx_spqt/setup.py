# Copyright © 2023-2024 Apple Inc.

from setuptools import setup

from mlx import extension

if __name__ == "__main__":
    setup(
        name="mlx_spqt",
        version="0.0.0",
        description="Sample C++ and Metal extensions for MLX primitives.",
        ext_modules=[extension.CMakeExtension("mlx_spqt._spqt_ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mlx_spqt"],
        package_data={"mlx_spqt": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.8",
    )
