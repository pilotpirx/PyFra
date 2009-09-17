# -*- coding: utf-8 -*-

from PyQt4 import QtGui, QtCore, QtSvg
from fraktale import *
from mathe_graphic import *

def save_fraktal(fraktal='Sierpinski',
                 method='mvkm',
                 file='out.png',
                 depth=6,
                 resolution=(1000, 1000)
                 ):
    fraktal_class = eval(fraktal)
    fraktal = fraktal_class(method)
    bild_koord_trans = Bild_Koord_Trans(width  = resolution[0],
                                        height = resolution[1],
                                        min    = fraktal.um_rechteck().links_unten(),
                                        max    = fraktal.um_rechteck().rechts_oben()
                                        )
    im = QtGui.QImage(resolution[0], resolution[1], QtGui.QImage.Format_Mono)
    painter = QtGui.QPainter(im)
    painter.eraseRect(QtCore.QRect(0, 0, *resolution))
    fraktal.zeichne_mit_tiefe(depth, painter, bild_koord_trans)
    painter.end()
    im.save(file)


if __name__ == '__main__':
    save_fraktal(fraktal='Kochkurve',  method='mvkm', depth=6,      file='Kochkurve.png' )
    save_fraktal(fraktal='Sierpinski', method='mvkm', depth=7,      file='Sierpinski.png')
    save_fraktal(fraktal='Farn',       method='mvkm', depth=10,     file='Farn.png'      )
