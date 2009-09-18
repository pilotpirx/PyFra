# -*- coding: utf-8 -*-

from PyQt4 import QtGui, QtCore, QtSvg
from fraktale import *
from mathe_graphic import *

from optparse import OptionParser

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


def main():
    usage = "usage: %prog [options] fractal-name"
    parser = OptionParser(usage)
    
    parser.add_option("-f", "--file", dest="filename", metavar='FILE',
                      help="write picture to FILE", default='out.png')
    parser.add_option("-m", "--method",
                      action="store", dest="method",
                      help="Which method to use (mvkm or gvkm)", default='mvkm')
    parser.add_option("-n", "--iterations",
                      action="store", dest="depth", type='int', default=5,
                      help="how many iterations to use")
    parser.add_option("-r", "--resolution", dest="resolution", metavar='X Y',
                      action="store", type='int', nargs=2, default=(1000,1000),
                      help="Set the resolution of the picture to X times Y")
    
    (options, args) = parser.parse_args()
    if len(args) != 1:
        parser.error("incorrect number of arguments\nPossible names are 'Sierpinsky', 'Kochkurve' or 'Farn' ")
    
    save_fraktal(fraktal=args[0], method=options.method, depth=options.depth, file=options.filename, resolution=options.resolution)
    
if __name__ == "__main__":
    main()

#if __name__ == '__main__':
#    save_fraktal(fraktal='Kochkurve',  method='mvkm', depth=6,      file='Kochkurve.png' )
#    save_fraktal(fraktal='Sierpinski', method='mvkm', depth=7,      file='Sierpinski.png')
#    save_fraktal(fraktal='Farn',       method='mvkm', depth=10,     file='Farn.png'      )
