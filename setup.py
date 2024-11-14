from setuptools import setup, find_packages#, Extension

#utils_module = Extension(
#    'torchstate.C.utils',
#    sources=['torchstate/C/csrc/utils.cpp'],
#    include_dirs=[],  # Add any necessary include directories
#    extra_compile_args=['-std=c++11'],  # Add any necessary compile arguments
#)

setup(
    name="torchstate",
    version="0.1.0",
    author="Jack Min Ong",
    author_email="ongjackm@gmail.com",
    description="Torch state package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Jackmin801/torchstate",
    packages=find_packages(),
    #ext_modules=[utils_module],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=2.0.0",
        # Add other dependencies
    ],
)