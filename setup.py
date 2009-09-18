# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    name="PyFra",
    version="0.1",
    author='Adrian Seyboldt',
    author_email='adrian_seyboldt@web.de',
    package_dir = {'pyfra': 'src/pyfra'},
    packages = ['pyfra'],
    ext_modules = [Extension("pyfra.mvkm", 
                             ["src/pyfra/mvkm.pyx"]
                             #extra_compile_args=["-g"],
                             #extra_link_args=["-g"]
                             ),
                   Extension("pyfra.mathe_graphic",
                             ["src/pyfra/matheGraphic.pyx"]
                            )
                  ],
    scripts=['src/pyfra/pyfra']
)
