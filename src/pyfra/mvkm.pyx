# -*- coding: utf-8 -*-

from mathe_graphic import *


from numpy import array, ndarray, pi, sin, cos, \
                  log, ceil, zeros, dot, int32
from PyQt4 import QtGui, QtCore
from numpy.linalg import det
from pprint import pprint, pformat
import numpy as np

cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t



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
    
    
    def zeichne_mit_tiefe(self, imag, tiefe, bild_trans, 
                          area=None, progress=False):
        for gr in self._gr_mit_tiefe_iter(tiefe, area, progress):
            gr.zeichne(imag, bild_trans)
    
    def _gr_mit_tiefe_iter(self, tiefe, area, progress=False):
        raise NotImplementedError()
        
    def um_rechteck(self):
        """ TODO: allgemein implementieren
        """
        return Rect(array([-8,-2]), array([8,12]))


cdef inline DTYPE_t d_abs(DTYPE_t x): return x if x > 0 else - x
cdef inline DTYPE_t d_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b


cdef inline DTYPE_t get_verkl_faktor(np.ndarray m):
    return d_max(d_abs(m[0,0]) + d_abs(m[0,1]),
                 d_abs(m[1,0]) + d_abs(m[1,1])
                 )




class Adaptiv_MVKM(MVKM):
    
    def __init__(self, *transformationen):
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
    
    def zeichne_mit_tiefe(self, imag, double epsilon, bild_trans, 
                          area=None, progress=False):
        fPunkt = QtCore.QPointF
        cdef np.ndarray[DTYPE_t] bild_trans_verkl_faktor = \
                        array(bild_trans.get_verkl_faktor(), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t] bild_min = array(bild_trans.min, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t] bild_max = array(bild_trans.max, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t] bild_wh = array(bild_trans.wh, dtype=DTYPE)
        
        
        cdef double max_verkl_faktor = max([self.get_verkl_faktor(f.matrix)
                                for f in self.transformationen])
        cdef double durchm_fraktal = 8.0
        cdef int max_transf = int(ceil(log(epsilon / durchm_fraktal) /
                              log(max_verkl_faktor)))
        max_transf = 40
        
        # Die Matrizen der bereits zusammengesetzten Transformationen in
        # einem Dreidimensionalen Array. matrizen[0] ist zum Beispiel die
        # erste Transformationsmatrix
        cdef np.ndarray[DTYPE_t, ndim=3] matrizen = zeros((max_transf, 2, 2))
        
        # Das selbe fuer die Vektoren der Transformationen
        cdef np.ndarray[DTYPE_t, ndim=2] vektoren = zeros((max_transf, 2))
        
        cdef np.ndarray[DTYPE_t] punkt = array([0.0, 0.0])
        
        # Die Matrizen und Vektoren der einzelnen elementaren Transformationen
        cdef np.ndarray[DTYPE_t, ndim=3] abb_matrizen = array(
                              [j.matrix for j in self.transformationen])
        cdef np.ndarray[DTYPE_t, ndim=2] abb_vektoren = array(
                              [j.vector for j in self.transformationen])
        
        # Anzahl der elementaren Abbildungen
        cdef unsigned int n = len(abb_matrizen)
        
        cdef unsigned int k
        # Eintragen der ersten Abbildung in die Matrizenarrays
        for k in range(n):
            matrizen[k] = abb_matrizen[k]
            vektoren[k] = abb_vektoren[k]
        
        # Zaehler fuer die Nummer der aktuellen Transformation im Array, 
        # zeigt auf die letzte eingefuegte Abbildung 
        cdef int i = n - 1
        
        cdef np.ndarray[DTYPE_t, ndim=2] matrix
        cdef np.ndarray[DTYPE_t] vektor
        cdef np.ndarray[DTYPE_t] t_objekt
        cdef double groesse
        
        while i >= 0:
            
            matrix = matrizen[i].copy()
            vektor = vektoren[i].copy()
            
            t_objekt = dot(matrix, punkt) + vektor
            
            # wenn der Punkt nicht im Epsilonkragen des zu zeichnenden 
            # bereichs ist
            #if not self.is_in_umkreis(t_objekt, area, epsilon):
            #    i -= 1
            #    assert False
            #    continue
            
            # zeichnen, wenn die gewuenschte Genauigkeit erreicht ist
            groesse = get_verkl_faktor(matrix) * durchm_fraktal
            if groesse is not None and groesse <= epsilon:
                i -= 1
                r1, r2 = bild_trans_verkl_faktor * groesse
                bild_koord = (t_objekt - bild_min)
                bild_koord = bild_koord * bild_wh
                bild_koord /= bild_max - bild_min
                bild_koord[1] = bild_wh[1] - bild_koord[1]
                imag.drawEllipse(fPunkt(*bild_koord.round().astype(int32)),
                                 r1,
                                 r2,
                                 ) 
                continue
            
            for k in range(n):
                matrizen[<unsigned int> (i + k)] = dot(matrix, abb_matrizen[k])
                vektoren[<unsigned int> (i + k)] = dot(matrix, abb_vektoren[k]) + vektor
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
