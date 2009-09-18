# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils.core import Command
import shutil, os
from subprocess import check_call, call, Popen, PIPE
import tarfile

version = '0.1'
project_name = 'PyFra'

class build_deb (Command):
    
    files = {
        'changelog': \
        """%s (%s) jaunty; urgency=low

  * I don't know

 -- Adrian Seyboldt <adrian.seyboldt@web.de>  %s""" \
         % (project_name.lower(), version, Popen(["date", "-R"], stdout=PIPE).communicate()[0],),
        'compat':"""7""",
        'control':"""
Source: pyfra                                   
Section: math
Priority: extra
Maintainer: Adrian Seyboldt <adrian.seyboldt@web.de>
Build-Depends: debhelper (>= 7), python, python-support (>= 0.6), cython, python-qt4, python-numpy, cdbs (>=0.4.49), python-all-dev
Standards-Version: 3.8.0

Package: pyfra
Architecture: any
Depends: python, ${shlibs:Depends}, ${misc:Depends}, ${python:Depends}
Description: Make fractal pictures
 nerai
""",
        'copyright':"""
Copyright:

    Copyright (C) 2008-2009 Adrian Seyboldt

License:

    This package is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This package is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this package; if not, see .

    On Debian systems, the complete text of the GNU General
    Public License can be found in `/usr/share/common-licenses/GPL-3'.

        """,
        'dirs':"""
usr/bin
usr/sbin
        """,
        'docs':"""
        """,
        'pycompat':"""2""",
        'rules':\
"""#!/usr/bin/make -f
# -*- makefile -*-

DEB_PYTHON_SYSTEM := pysupport

include /usr/share/cdbs/1/rules/debhelper.mk
include /usr/share/cdbs/1/class/python-distutils.mk

clean::
\trm -rf build build-stamp configure-stamp build/ MANIFEST
\tdh_clean
"""
    }
    
    
    # Brief (40-50 characters) description of the command
    description = "Builds the corresponding .deb"

    # List of option tuples: long name, short name (None if no short
    # name), and help string.
    user_options = [('launchpad', 'l',
                     "export to launchpad"),
                   ]


    def initialize_options (self):
        self.launchpad = False


    def finalize_options (self):
        pass

    def write_files(self):
        for name in build_deb.files:
            f = open('debian/'+name, 'w')
            f.write(build_deb.files[name])
            f.close()

    def run (self):
        check_call(['python', 'setup.py', 'clean'])
        check_call(['python', 'setup.py', 'sdist'])
        try:
            shutil.rmtree('temp_deb')
        except:
            pass
        os.mkdir('temp_deb')
        shutil.copy('dist/%s-%s.tar.gz' % (project_name, version), 'temp_deb/src.tar.gz')
        tarfile.open('temp_deb/src.tar.gz').extractall('temp_deb')
        os.remove('temp_deb/src.tar.gz')
        os.chdir('temp_deb/%s-%s' % (project_name, version))
        os.mkdir('debian')
        self.write_files()
        os.chmod('debian/rules', 0777)
        if self.launchpad:
            check_call(['dpkg-buildpackage', '-S', '-rfakeroot'])
        else:
            check_call(['dpkg-buildpackage', '-rfakeroot'])



setup(
    cmdclass = {'build_ext': build_ext, 'build_deb': build_deb},
    name=project_name,
    version=version,
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
