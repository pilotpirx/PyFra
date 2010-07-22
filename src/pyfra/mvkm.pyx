# -*- coding: utf-8 -*-

from mathe_graphic import *


from numpy import array, ndarray, pi, sin, cos, \
                  log, ceil, zeros, dot, int32, isnan
import numpy
from PyQt4 import QtGui, QtCore
from numpy.linalg import det
from pprint import pprint, pformat
#import numpy as np


cimport cython
cimport numpy as np
DTYPE = numpy.float64
ctypedef np.float64_t DTYPE_t



class MVKM(object):
    """ Eine Mehrfach-Verkleinerungs-Kopier-Maschine. Sie
    wendet mehrere Transformationen auf ein Anfangsobjekt
    an.
    """
    
    def __init__(self, anfangs_objekt , *transformationen):
        if len(transformationen) == 0:
            raise Exception("kann keine leere MVKM erzeugen")
        if isinstance(transformationen[0], list):
            if len(transformationen) > 1:
                raise Exception("mehr als eine Liste angegeben")
            if len(transformationen[0]) == 0:
                raise Exception("kann keine leere MVKM erzeugen")
            transformationen = tuple(transformationen[0])
        self.transformationen = transformationen
        self.anfangsobjekt = anfangs_objekt
        self.tiefe_objekte_dict = {0: anfangs_objekt}
        self.max_ganz_berechnete_tiefe = 0
    
    def get_graphisches_objekt_tiefe(self, tiefe, ausschnitt=None):
        """ Gibt eine Liste graphischer Objekte zurueck, die
        durch die wiederholte (tiefe - mal) Anwendung der beim
        Erzeugen der MVKM angegebenen Transformationen entstehen.
        
        >>> T_0 = Transformation(array([[1,2],[2,1]]), array([1,2]))
        >>> T_1 = Transformation(array([[3,2],[1,2]]), array([-1,2]))
        >>> G = Graphisches_Objekt(array([1,2]), array([3,2]))
        >>> M = MVKM(G, [T_0, T_1])
        >>> print M.get_graphisches_objekt_tiefe(2)
        Graphisches_Objekt(
            Graphisches_Objekt(
                Graphisches_Objekt(
                    [19 20]
                    [29 28]
                )
                Graphisches_Objekt(
                    [21 21]
                    [31 35]
                )
            )
            Graphisches_Objekt(
                Graphisches_Objekt(
                    [29 20]
                    [43 30]
                )
                Graphisches_Objekt(
                    [31 22]
                    [53 32]
                )
            )
        )
        """
        if self.max_ganz_berechnete_tiefe >= tiefe:
            assert self.tiefe_objekte_dict.has_key(tiefe)
            return self.tiefe_objekte_dict[tiefe]
        for i in range(self.max_ganz_berechnete_tiefe + 1, tiefe + 1):
            self.tiefe_objekte_dict[i] = Graphisches_Objekt(
                                        [f(self.tiefe_objekte_dict[i - 1])
                                         for f in self.transformationen])
        return self.tiefe_objekte_dict[tiefe]
    
    
    def get_pixmap(self, array, tiefe, bild_trans, 
                         area=None, progress=False):
        for gr in self._gr_mit_tiefe_iter(tiefe, area, progress):
            gr.zeichne(imag, bild_trans)

    def has_vector_format(self):
        return True

    def get_vector(self, args*):
        raise NotImplementedError("Vector format output not yet implemented")
    
    def _gr_mit_tiefe_iter(self, tiefe, area, progress=False):
        raise NotImplementedError()
        
    def um_rechteck(self):
        """ TODO: allgemein implementieren
        """
        return Rect(array([-8,-2]), array([8,12]))


cdef inline DTYPE_t d_abs(DTYPE_t x): return x if x > 0 else - x
cdef inline DTYPE_t d_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b

#cdef inline DTYPE_t get_verkl_faktor(np.ndarray m ):
#    return d_max(d_abs(m[0,0]) + d_abs(m[0,1]),
#                 d_abs(m[1,0]) + d_abs(m[1,1])
#                 )

def get_verkl_faktor(np.ndarray[DTYPE_t, ndim=2] m):
    cdef DTYPE_t p = m[0,0]**2 + m[1,0]**2 + m[0,1]**2 + m[1,1]**2
    cdef DTYPE_t q = (m[0,0]*m[1,1] - m[1,0]*m[0,1])**2
    cdef DTYPE_t erg = sqrt((p + sqrt(p**2 - 4*q))/2)
    return erg
        

#cdef inline np.ndarray[DTYPE_t, ndim=2] dot_2x2_2x2(np.ndarray a,
#                                                    np.ndarray b):
#    cdef np.ndarray[DTYPE_t, ndim=2] erg = zeros((2,2))
#    erg[0,0] = a[0,0]*b[0,0] + a[1,0]*b[0,1]
#    erg[1,0] = a[0,0]*b[1,0] + a[1,0]*b[1,1]
#    erg[0,1] = a[0,1]*b[0,0] + a[0,1]*b[0,1]
#    erg[1,1] = a[0,1]*b[1,0] + a[0,1]*b[1,1]
#    return erg

cdef inline void dot_2x2_2x2(DTYPE_t a[2][2], DTYPE_t b[2][2], DTYPE_t erg[2][2]):
    erg[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0]
    erg[1][0] = a[1][0]*b[0][0] + a[1][1]*b[1][0]
    erg[0][1] = a[0][0]*b[0][1] + a[0][1]*b[1][1]
    erg[1][1] = a[1][0]*b[0][1] + a[1][1]*b[1][1]


class Adaptiv_MVKM(object):
    
    def __init__(self, fraktal, *transformationen):
        self.fraktal = fraktal
        if len(transformationen) == 0:
            raise Exception("kann keine leere MVKM erzeugen")
        if isinstance(transformationen[0], list):
            if len(transformationen) > 1:
                raise Exception("mehr als eine Liste angegeben")
            if len(transformationen[0]) == 0:
                raise Exception("kann keine leere MVKM erzeugen")
            transformationen = tuple(transformationen[0])
        self.transformationen = transformationen
        self.anfangsobjekt = Punkt(array([0,0]))
    
    def is_in_umkreis(self, punkt, area, epsilon):
        return True
    
    def get_verkl_faktor(self, matrix):
        m = abs(matrix)
        #print max(m[0,0] + m[0,1], m[1,0] + m[1,1])
        return max(m[0,0] + m[0,1], m[1,0] + m[1,1])
        #q = det(matrix)**2
        #p = (matrix*matrix).sum()
        #print sqrt((p + sqrt(p*p - 4*q)) / 2)
        #return sqrt((p + sqrt(p*p - 4*q)) / 2)
    
    def zeichne_mit_tiefe(self, *opts):
        raise NotImplementedError("Does not make sense for Adaptiv_MVKM")
    
    @cython.boundscheck(False)
    def get_pixmap(self, array, bild_trans):
        
        cdef double epsilon = bild_trans.pixel_size_diag() / 2.0
        
        # für kleine Schleifen
        cdef unsigned int k
        
        print "epsilon: %f" % (epsilon,)
        
        #cdef np.ndarray[DTYPE_t] bild_trans_verkl_faktor = \
        #                array(bild_trans.get_verkl_faktor(), dtype=DTYPE)
        
        btmin = bild_trans.min
        cdef DTYPE_t bild_min[2]
        bild_min[0] = btmin[0]
        bild_min[1] = btmin[1]
        #cdef np.ndarray[DTYPE_t] bild_min = array([btmin[0], btmin[1]], dtype=DTYPE)
        
        btmax = bild_trans.max
        cdef DTYPE_t bild_max[2]
        bild_max[0] = btmax[0]
        bild_max[1] = btmax[1]
        #cdef np.ndarray[DTYPE_t] bild_max = array([btmax[0], btmax[1]], dtype=DTYPE)
        
        bl_wh = bild_trans.wh
        cdef DTYPE_t bild_wh[2]
        bild_wh[0] = bl_wh[0]
        bild_wh[1] = bl_wh[1]
        #cdef np.ndarray[DTYPE_t] bild_wh = array(bild_trans.wh, dtype=DTYPE)
        
        
        cdef double max_verkl_faktor = max([self.get_verkl_faktor(f.matrix)
                                for f in self.transformationen])
        cdef double durchm_fraktal = self.fraktal.diag_size()
        #cdef int max_transf = int(ceil(log(epsilon / durchm_fraktal) /
        #                      log(max_verkl_faktor)))
        #cdef unsigned int max_transf = 200 # TODO
        DEF MAX_TRANSF = 200
        DEF MAX_ANZAHL_TRANSF = 10
        
        cdef unsigned int anzahl_transf = len(self.transformationen)
        if anzahl_transf > MAX_ANZAHL_TRANSF:
            raise ValueError("Zu viele Transformationen")
        
        # Die Matrizen der bereits zusammengesetzten Transformationen in
        # einem Dreidimensionalen Array. matrizen[0] ist zum Beispiel die
        # erste Transformationsmatrix
        #cdef np.ndarray[DTYPE_t, ndim=3] matrizen = zeros((max_transf, 2, 2))
        cdef DTYPE_t matrizen[MAX_TRANSF][2][2]
        
        
        # Das selbe fuer die Vektoren der Transformationen
        #cdef np.ndarray[DTYPE_t, ndim=2] vektoren = zeros((max_transf, 2))
        cdef DTYPE_t vektoren[MAX_TRANSF][2]
        
        #cdef np.ndarray[DTYPE_t] punkt = array([0.0, 0.0])
        cdef DTYPE_t punkt[2]
        punkt[0] = 0
        punkt[1] = 0
        
        # Die Matrizen und Vektoren der einzelnen elementaren Transformationen
        #cdef np.ndarray[DTYPE_t, ndim=3] abb_matrizen = array([j.matrix for j in self.transformationen])
        cdef DTYPE_t abb_matrizen[MAX_ANZAHL_TRANSF][2][2]
        for k in range(anzahl_transf):
            abb_matrizen[k][0][0] = self.transformationen[k].matrix[0,0]
            abb_matrizen[k][1][0] = self.transformationen[k].matrix[1,0]
            abb_matrizen[k][0][1] = self.transformationen[k].matrix[0,1]
            abb_matrizen[k][1][1] = self.transformationen[k].matrix[1,1]
                                    
        #cdef np.ndarray[DTYPE_t, ndim=2] abb_vektoren = array([j.vector for j in self.transformationen])
        cdef DTYPE_t abb_vektoren[MAX_ANZAHL_TRANSF][2]
        for k in range(anzahl_transf):
            abb_vektoren[k][0] = self.transformationen[k].vector[0]
            abb_vektoren[k][1] = self.transformationen[k].vector[1]
        
        # Anzahl der elementaren Abbildungen
        cdef unsigned int n = anzahl_transf
        
        # Eintragen der ersten Abbildung in die Matrizenarrays
        for k in range(n):
            #matrizen[k] = abb_matrizen[k]
            matrizen[k][0][0] = abb_matrizen[k][0][0]
            matrizen[k][1][0] = abb_matrizen[k][1][0]
            matrizen[k][0][1] = abb_matrizen[k][0][1]
            matrizen[k][1][1] = abb_matrizen[k][1][1]
            
            #vektoren[k] = abb_vektoren[k]
            vektoren[k][0] = abb_vektoren[k][0]
            vektoren[k][1] = abb_vektoren[k][1]
        
        # Zaehler fuer die Nummer der aktuellen Transformation im Array, 
        # zeigt auf die letzte eingefuegte Abbildung 
        cdef int i = n - 1
        
        #cdef np.ndarray[DTYPE_t, ndim=2] matrix
        cdef DTYPE_t matrix[2][2]
        
        #cdef np.ndarray[DTYPE_t] vektor
        cdef DTYPE_t vektor[2]
        
        #cdef np.ndarray[DTYPE_t] t_objekt
        cdef DTYPE_t t_objekt[2]
        
        cdef double groesse
        
        cdef DTYPE_t bild_koord[2]
        
        cdef int x_pixel, y_pixel
        
        cdef DTYPE_t p, q, erg
        
        while i >= 0:
            #matrix = matrizen[i].copy()
            matrix[0][0] = matrizen[i][0][0]
            matrix[1][0] = matrizen[i][1][0]
            matrix[0][1] = matrizen[i][0][1]
            matrix[1][1] = matrizen[i][1][1]
            
            #vektor = vektoren[i].copy()
            vektor[0] = vektoren[i][0]
            vektor[1] = vektoren[i][1]
            
            
            #t_objekt = dot(matrix, punkt) + vektor
            t_objekt[0] = vektor[0]
            t_objekt[1] = vektor[1]
            
            # wenn der Punkt nicht im Epsilonkragen des zu zeichnenden 
            # bereichs ist
            #if not self.is_in_umkreis(t_objekt, area, epsilon):
            #    i -= 1
            #    assert False
            #    continue
            
            # zeichnen, wenn die gewuenschte Genauigkeit erreicht ist
             
            p = matrix[0][0]**2 + matrix[1][0]**2 + matrix[0][1]**2 + matrix[1][1]**2
            q = (matrix[0][0]*matrix[1][1] - matrix[1][0]*matrix[0][1])**2
            erg = sqrt((p + sqrt(p**2 - 4*q))/2)
            groesse = erg * durchm_fraktal
            
            #groesse = get_verkl_faktor(matrix) * durchm_fraktal
            if groesse <= epsilon: # TODO: Is groesse None?
                i -= 1
                #r1, r2 = bild_trans_verkl_faktor * groesse
                for k in range(2):
                    bild_koord[k] = (t_objekt[k] - bild_min[k])
                    bild_koord[k] = bild_koord[k] * bild_wh[k]
                    bild_koord[k] /= bild_max[k] - bild_min[k]
                bild_koord[1] = bild_wh[1] - bild_koord[1]
                
                
                #x_pixel = int(round(bild_koord[0]))
                #y_pixel = int(round(bild_koord[1]))
                x_pixel = <np.int> bild_koord[0]
                y_pixel = <np.int> bild_koord[1]
                
                imag.setPixel(x_pixel, y_pixel, 1)
                continue
            
            for k in range(n):
                #matrizen[<unsigned int> (i + k)] = dot(matrix, abb_matrizen[k])
                matrizen[i + k][0][0] = matrix[0][0]*abb_matrizen[k][0][0] +\
                                        matrix[0][1]*abb_matrizen[k][1][0]
                matrizen[i + k][1][0] = matrix[1][0]*abb_matrizen[k][0][0] +\
                                        matrix[1][1]*abb_matrizen[k][1][0]
                matrizen[i + k][0][1] = matrix[0][0]*abb_matrizen[k][0][1] +\
                                        matrix[0][1]*abb_matrizen[k][1][1]
                matrizen[i + k][1][1] = matrix[1][0]*abb_matrizen[k][0][1] +\
                                        matrix[1][1]*abb_matrizen[k][1][1]

                #vektoren[<unsigned int> (i + k)] = dot(matrix, abb_vektoren[k]) + vektor
                vektoren[i + k][0] = matrix[0][0]*abb_vektoren[k][0] +\
                                     matrix[0][1]*abb_vektoren[k][1] +\
                                     vektor[0]
                vektoren[i + k][1] = matrix[1][0]*abb_vektoren[k][0] +\
                                     matrix[1][1]*abb_vektoren[k][1] +\
                                     vektor[1]
                
            i += n - 1
#        if area is None:
#            area = self.um_rechteck()
#        transfStack = list(self.transformationen[:])
#        while len(transfStack) > 0:
#            f = transfStack.pop()
#            transO = f(self.anfangsobjekt)
#            if not transO.um_rechteck().zum_teil_in(area):
#                continue
#            if f.get_verkl_faktor() <= epsilon/durchm_fraktal:
#                yield transO
#                continue
#            for f2 in self.transformationen:
#                transfStack.append(Transformation(
#                                                  dot(f.matrix, f2.matrix),
#                                                  dot(f.matrix, f2.vector) + 
#                                                      f.vector
#                                                  )
#                                   )

    
#    def zeichne_mit_tiefe(self, imag, double epsilon, bild_trans, 
#                          area=None, progress=False):
#        
