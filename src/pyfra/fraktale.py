# -*- coding: utf-8 -*-

from mathe_graphic import *
from mvkm import *
from numpy import *
from gvkm import *

class _Fraktal(object):
    def __init__(self,start, transformationen, method='mvkm', wahrscheinlichkeiten=None):
        self.start = start
        self.method=method
        self.transformationen = transformationen
        if method == 'mvkm':
            self.mvkm = MVKM(self.start, self.transformationen)
        elif method == 'gvkm':
            self.gvkm = GVKM(self.transformationen, wahrscheinlichkeiten)
        elif method == 'adaptiv_mvkm':
            self.mvkm = Adaptiv_MVKM(self, self.transformationen)
        else:
            raise ValueError("Method '%s' not found" % (method,))

    def diag_size(self):
        return self.um_rechteck().diag_size()

class _Fraktal_Painter(_Fraktal):
    u""" Stellt Fraktale dar, die durch einen Painter gezeichnet werden können,
    also keinen Zugang auf die einzelnen Pixel benötigen. Diese Fraktale können auch
    als svg ausgegeben werden.
    """
    def __init__(self, start, transformationen, method='mvkm', wahrscheinlichkeiten=None):
        _Fraktal.__init__(self, start, transformationen, method, wahrscheinlichkeiten)
        if method == 'adaptiv_mvkm':
            raise ValueError('Kann kein Painter-Fraktal mit adaptiv_mvkm erzeugen')
        if method == 'gvkm':
            raise ValueError('Kann kein Painter-Fraktal mit gvkm erzeugen')

    def zeichne(self, painter, tiefe, bild_koord_trans):
        u""" Zeichne das Fraktal mit dem übergebenen QPainter."""
        
        if self.method == 'mvkm':
            self.mvkm.get_graphisches_objekt_tiefe(tiefe).zeichne(painter, bild_koord_trans)
            
    def requires_image(self):
        return False
        

class _Fraktal_Pixel(_Fraktal):
    u""" Stellt Fraktale dar, die direkt auf das Bild (QImage) gezeichnet werden.
    Sie können nicht als svg ausgegeben werden
    """
    def __init__(self, start, transformationen, method='gvkm', wahrscheinlichkeiten=None):
        _Fraktal.__init__(self, start, transformationen, method, wahrscheinlichkeiten)
        if method == 'mvkm':
            raise ValueError('Kann kein Pixel-Fraktal mit mvkm erzeugen')
    
    def zeichne(self, imag, bild_koord_trans, tiefe=None):
        if self.method == 'gvkm':
            if tiefe is None:
                raise ValueError(u'tiefe muss übergeben werden')
            self.gvkm.zeichne_mit_tiefe(imag, tiefe, bild_koord_trans)
        elif self.method == 'adaptiv_mvkm':
            self.mvkm.zeichne_alle_pixel(imag, bild_koord_trans)
            
    def requires_image(self):
        return True

class Sierpinski(_Fraktal_Painter):
    def __init__(self, method='mvkm', start=None):
        if start is None:
            start = Dreieck(array([0, 0]), array([1, 0]), array([0.5, 0.5*sqrt(3)]))
        
        transformationen = [Transformation(
                                         array([[0.5, 0], [0, 0.5]]),
                                         array([0, 0])
                                     ),
                                 Transformation(
                                         array([[0.5, 0], [0, 0.5]]),
                                         array([0.5, 0])
                                     ),
                                 Transformation(
                                         array([[0.5, 0], [0, 0.5]]),
                                         array([0.25, 0.25 * sqrt(3)])
                                     )
                                ]
        _Fraktal_Painter.__init__(self, start, transformationen, method)
    
    def um_rechteck(self):
        return Rect(array([-0.1,-0.1]), array([1.1, 1.1]))

class Kochkurve(_Fraktal_Painter):
    def __init__(self, method='mvkm', start=None):
        if start is None:
            start = Line(array([0,0]), array([1,0]))
        
        transformationen = [Transformation(array([[1/3.0, 0],
                                                  [0, 1/3.0]]), 
                                           array( [0, 0])), 
                            Transformation(array([[-1/6.0, -sqrt(3)/6], 
                                                  [-sqrt(3)/6, 1/6.0]]), 
                                           array( [0.5, sqrt(3)/6])),
                            Transformation(array([[-1/6.0, sqrt(3)/6], 
                                                  [sqrt(3)/6, 1/6.0]]), 
                                           array( [2/3.0, 0])),
                            Transformation(array([[1/3.0, 0], 
                                                  [0, 1/3.0]]), 
                                           array( [2/3.0, 0]))
                            ]
        _Fraktal_Painter.__init__(self, start, transformationen, method)
    
    def um_rechteck(self):
        return Rect(array([-0.1,-0.1]), array([1.1, 1.1]))

class Farn(_Fraktal_Pixel):
    def __init__(self, method='gvkm', start=None):
        if start is None and method != 'gvkm':
            start = Rect(array([-1,-1]), array([1,6]))
        r = 2*pi/360.0
        transformationen = [Transformation(array([[0.85*cos(-r*2.5),-0.85*sin(-r*2.5)],
                                              [0.85*sin(-r*2.5),0.85*cos(-r*2.5)]]),
                                       array([0.0, 1.6])),
                        Transformation(array([[0.3*cos(r*49),-0.34*sin(r*49)],
                                              [0.3*sin(r*49),0.34*cos(r*49)]]),
                                       array([0.0, 1.6])),
                        Transformation(array([[0.3*cos(r*120),-0.37*sin(-r*50)],
                                              [0.3*sin(r*120),0.37*cos(-r*50)]]),
                                       array([0.0, 0.44])),
                        Transformation(array([[0.0, 0.0],
                                              [0.0, 0.16]]),
                                       array([0.0 ,0.0]))
                        ]
        _Fraktal_Pixel.__init__(self, start, transformationen, method, wahrscheinlichkeiten=[0.75, 0.08, 0.08, 0.08])
    
    def um_rechteck(self):
        return Rect(array([-5,-2]), array([6,12]))
    
class Levy_C_Kurve_Was_anderes(_Fraktal_Painter):
    def __init__(self, method='mvkm', start=None):
        if start is None:
            start = Line(array([0,0]), array([1,0]))
        transformationen = [Transformation_by_angle(pi/4, factor=1/1.45),
                            Transformation_by_angle(3*pi/4, factor=1/1.45, move=array([1,0]))]
        _Fraktal_Painter.__init__(self, start, transformationen, method)
        
    def um_rechteck(self):
        return Rect(array([-5,-5]), array([5,5]))

class Levy_C_Kurve(_Fraktal_Painter):
    def __init__(self, method='mvkm', start=None):
        if start is None:
            start = Line(array([0,0]), array([1,0]))
        transformationen = [Transformation_by_angle(5*pi/4, factor=1/sqrt(2), move=array([0.5,sqrt(2)/2/sqrt(2)])),
                            Transformation_by_angle(3*pi/4, factor=1/sqrt(2), move=array([1,0]))]
        _Fraktal_Painter.__init__(self, start, transformationen, method)
        
    def um_rechteck(self):
        return Rect(array([-1,-1]), array([2,2]))

                        
    