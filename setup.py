from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import subprocess
import os
import shutil
import glob as _glob

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        # Assuming the setup.py is located at the project root which also contains the CMakeLists.txt
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        # Determine the number of cores to use
        num_cores = os.cpu_count()  # Get the number of cores available on your system

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        build_cmd = ['cmake', '--build', '.', '--config', cfg]
        if sys.platform == 'win32':
            build_cmd += ['--', f'/m:{num_cores}']  # MSVC parallel build
        else:
            build_cmd += ['--', f'-j{num_cores}']  # Make/Ninja parallel build
        subprocess.check_call(build_cmd, cwd=self.build_temp)

        # For editable installs: copy built .pyd/.so and .dll/.so to the source
        # root so the setuptools editable finder can locate them.
        srcdir = ext.sourcedir
        for pattern in ['*.pyd', '*.so', '*.dll']:
            # MSVC multi-config puts outputs under extdir/Release/
            for f in _glob.glob(os.path.join(extdir, cfg, pattern)):
                dst = os.path.join(srcdir, os.path.basename(f))
                if os.path.abspath(f) != os.path.abspath(dst):
                    shutil.copy2(f, dst)
            for f in _glob.glob(os.path.join(extdir, pattern)):
                dst = os.path.join(srcdir, os.path.basename(f))
                if os.path.abspath(f) != os.path.abspath(dst):
                    shutil.copy2(f, dst)

setup(
    name='pysuperansac',
    version='1.0',
    author='Daniel Barath',
    author_email="majti89@gmail.com",
    description='A RANSAC implementation for robust estimation.',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension('pysuperansac', sourcedir='.')],
    cmdclass=dict(build_ext=CMakeBuild),
    url="https://github.com/you/superansac",
    zip_safe=False,
    license='MIT',
)
