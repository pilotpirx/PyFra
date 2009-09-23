# -*- coding: utf-8 -*-

from PyQt4 import QtGui, QtCore, QtSvg, QtOpenGL
import sys
import Oberfl_main as Ui_Main
from mathe_graphic import *
from mvkm import *
import numpy as n
from fraktale import *

class Fraktal_Bild(QtOpenGL.QGLWidget):
    def __init__(self, *opts):
        QtGui.QWidget.__init__(self, *opts)
        self._neues_fraktal = True
        self.tiefe = 8
        self.fPicture = None
        self.fraktal = None
    
    
    def paintEvent(self, paintEvent):
        if self.fraktal is None:
            return
        if self._neues_fraktal:
            self.fPicture = QtGui.QPicture()
            painter = QtGui.QPainter(self.fPicture)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            self.fraktal.zeichne_mit_tiefe(self.tiefe, painter, self.bild_koord_trans)
            painter.end()
            self._neues_fraktal = False
        painter = QtGui.QPainter(self)
        painter.drawPicture(0, 0, self.fPicture)
        painter.end()


    def setFraktal(self, fraktal, bild_koord_tans=None):
        self._neues_fraktal = True
        self.fraktal = fraktal
        if bild_koord_tans is None:
            bild_koord_tans = Bild_Koord_Trans(width = self.width(),
                                               height= self.height(),
                                               min   = fraktal.um_rechteck().links_unten(),
                                               max   = fraktal.um_rechteck().rechts_oben()
                                               )
        self.bild_koord_trans = bild_koord_tans
        self.repaint()
        self._neues_fraktal = False
        
    def saveFraktal(self):
        if self.fraktal is None:
            return
        alt_fPic = self.fPicture
        alt_bild_koord_trans = self.bild_koord_trans
        
        self.bild_koord_trans = Bild_Koord_Trans(width = 5000,
                                                 height= 5000,
                                                 min   = self.fraktal.um_rechteck().links_unten(),
                                                 max   = self.fraktal.um_rechteck().rechts_oben()
                                                )
        self._neues_fraktal = True
        self.repaint()
        file = "out.png"
        im = QtGui.QImage(5000,5000,QtGui.QImage.Format_RGB32)
        painter = QtGui.QPainter(im)
        painter.eraseRect(QtCore.QRect(0,0,5000,5000))
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.drawPicture(0,0,self.fPicture)
        painter.end()
        im.save(file)
        
        self.bild_koord_trans = alt_bild_koord_trans
        self.fPic = alt_fPic
        self._neues_fraktal = False
        self.repaint()
                
    
    def setTiefe(self, tiefe):
        self.tiefe = tiefe
        self.repaint()


class GMain(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        
        ui_main = Ui_Main.Ui_MainWindow()
        ui_main.setupUi(self)
        
        self.frBild = ui_main.FraktalBildWidget
        
        self.connect(ui_main.actionBeenden, QtCore.SIGNAL("activated()"),
                     QtGui.qApp, QtCore.SLOT("quit()"))
        self.connect(ui_main.TiefeSpinBox, QtCore.SIGNAL("valueChanged(int)"),
                     self.frBild.setTiefe)
        self.connect(ui_main.actionBild_speichern, QtCore.SIGNAL("activated()"),
                     self.frBild.saveFraktal)
        
        self.setVisible(True)
        
        
        
        self.frBild.setFraktal(Farn(method='mvkm'))
        #r = 2*pi/360.0
        #Adaptiv_MVKM(
                       #[Transformation(array([[0.85*cos(-r*2.5), -0.85*sin(-r*2.5)], 
                                              #[0.85*sin(-r*2.5), 0.85*cos(-r*2.5)]]), 
                                       #array([0.0, 1.6])), 
                        #Transformation(array([[0.3*cos(r*49), -0.34*sin(r*49)], 
                                              #[0.3*sin(r*49), 0.34*cos(r*49)]]), 
                                       #array([0.0, 1.6])), 
                        #Transformation(array([[0.3*cos(r*120), -0.37*sin(-r*50)], 
                                              #[0.3*sin(r*120), 0.37*cos(-r*50)]]), 
                                       #array([0.0, 0.44])), 
                        #Transformation(array([[0.0, 0.0], 
                                              #[0.0, 0.16]]), 
                                       #array([0.0 , 0.0]))]))
        
        self.ui_main = ui_main
        self.frBild.repaint()
        
        
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    m = GMain()
    sys.exit(app.exec_())

