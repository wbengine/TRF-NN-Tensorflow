from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import platform

tools_dir = './src/'

system_name = platform.system()
if system_name == "Windows":
    print("Call Windows tasks")
    ext_modules = [
        Extension('trie',
                  sources=['trie.pyx'],
                  include_dirs=[tools_dir],
                  language='c++',
                  extra_compile_args=['/openmp'],
                  extra_link_args=['/openmp'])
    ]
else:
    ext_modules = [
        Extension('trie',
                  sources=['trie.pyx'],
                  include_dirs=[tools_dir],
                  language='c++',
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-fopenmp'])
    ]

setup(
    name='trie',
    ext_modules=cythonize(ext_modules)
)
