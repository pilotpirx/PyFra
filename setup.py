from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("MatheGraphik", 
                             ["src/MatheGraphik.pyx"],
                             #extra_compile_args=["-g"],
                             #extra_link_args=["-g"]
                             )
                  ],
    name="PyFra",
    version="0.1",
    author='Adrian Seyboldt',
    author_email='adrian_seyboldt@web.de',
    package_dir = {'': 'src'},
    py_modules = ['Graphische_Oberfl', 'Oberfl_main']
)
