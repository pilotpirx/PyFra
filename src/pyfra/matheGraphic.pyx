# -*- coding: utf-8 -*-

from numpy import *
from PyQt4 import QtGui, QtCore


class Graphisches_Objekt(object):
    """ Ein graphisches Objekt. Es ist aus anderen 
    graphischen Objekten zusammengesetzt, die dem Konstruktor
    übergeben werden. Darunter können auch arrays sein, die
    als Punkte interpretiert werden. Ein graphisches Objekt
    mit solchen arrays kann aber nicht direkt gezeichnet
    werden, dafür muss eine eigene Unterklasse erzeugt 
    werden.
    
    >>> G = Graphisches_Objekt(array([1,2]), array([2,3]))
    >>> G.objekte
    (array([1, 2]), array([2, 3]))
    
    >>> G.is_elementar
    True
    
    >>> G = Graphisches_Objekt([array([1,2]), array([2,3])])
    >>> G.objekte
    (array([1, 2]), array([2, 3]))
    
    >>> Graphisches_Objekt()
    Traceback (most recent call last):
        ...
    Exception: kann kein leeres graphisches Objekt erzeugen
    
    >>> Graphisches_Objekt([])
    Traceback (most recent call last):
        ...
    Exception: kann kein leeres graphisches Objekt erzeugen
    
    >>> Graphisches_Objekt([array([1,2])], G)
    Traceback (most recent call last):
        ...
    Exception: mehr als eine Liste angegeben
    
    >>> H = Graphisches_Objekt(G); H.is_elementar
    False
    >>> H.is_rein_zusammengesetzt
    True
    """
    
    
    def __init__(self, *objekte):
        if len(objekte) == 0:
            raise Exception("kann kein leeres graphisches Objekt erzeugen")
        if isinstance(objekte[0], list):
            if len(objekte) > 1:
                raise Exception("mehr als eine Liste angegeben")
            if len(objekte[0]) == 0:
                raise Exception("kann kein leeres graphisches Objekt erzeugen")
            objekte = tuple(objekte[0])
        self.objekte = objekte
        self.is_elementar = all([isinstance(i, ndarray) for i in self.objekte])
        self.is_rein_zusammengesetzt = \
            all([isinstance(i, Graphisches_Objekt) for i in self.objekte])
    
    
    def transformiere(self, transf):
        """ Wendet die Transformation auf alle Objekte an
        und gibt ein solchermaßen verändertes graphisches
        Objekt zurück. Das ursprüngliche graphische Objekt 
        bleibt unverändert
        
        Tests in 'Transformation'
        """
        return self.__class__(*[transf(i) for i in self.objekte])
    
    
    def zeichne(self, g, bild_koord_tans):
        """ Zeichnet das graphische Objekt in g. Muss von
        einer Unterklasse implementiert werden, wenn self
        nicht rein zusammengesetzt ist.
        
        >>> G = Graphisches_Objekt([array([1,2]), array([2,3])])
        >>> G.zeichne(None, None)
        Traceback (most recent call last):
            ...
        Exception: Ist Abstrakt
        """
        if self.is_rein_zusammengesetzt:
            for obj in self.objekte:
                obj.zeichne(g, bild_koord_tans)
        else:
            raise Exception("Ist Abstrakt")
    
    def um_rechteck(self):
        """ gibt das kleinste alle Punkte umgebende senkrechte Rechteck
        zurueck
        >>> G = Graphisches_Objekt(array([0,0]), array([2,4]), array([1,5]))
        >>> print G.um_rechteck()
        Rect(
            [0 0]
            [2 0]
            [2 5]
            [0 5]
        )
        >>> G2 = Rect(array([2,2]), array([4,4]))
        >>> K = Graphisches_Objekt(G, G2)
        >>> print K.um_rechteck()
        Rect(
            [0 0]
            [4 0]
            [4 5]
            [0 5]
        )
        """
        if self.is_elementar:
            xs = [i[0] for i in self.objekte]
            ys = [i[1] for i in self.objekte]
            return Rect(array([min(xs),
                               min(ys)]),
                        array([max(xs),
                               max(ys)])
                        )
        elif self.is_rein_zusammengesetzt:
            rechtecke = [i.um_rechteck() for i in self.objekte]
            
            links_unten = [i.links_unten() for i in rechtecke]
            links = min([i[0] for i in links_unten])
            unten = min([i[1] for i in links_unten])
            
            rechts_oben = [i.rechts_oben() for i in rechtecke]
            rechts = max([i[0] for i in rechts_oben])
            oben   = max([i[1] for i in rechts_oben])
            
            return Rect(array([links, unten]), array([rechts, oben]))
        else:
            raise NotImplementedError()
    
    def __str__(self):
        """ 
        >>> G = Graphisches_Objekt([array([1,2]), array([2,3])])
        >>> K = Graphisches_Objekt(G,array([1,2]))
        >>> print K
        Graphisches_Objekt(
            Graphisches_Objekt(
                [1 2]
                [2 3]
            )
            [1 2]
        )
        """
        s = '%s(\n    ' % self.__class__.__name__
        a = "\n".join([str(i) for i in self.objekte]).replace("\n", "\n    ")
        s += a
        s += "\n)"
        return s



class Rect(Graphisches_Objekt):
    
    def __init__(self, *objekte):
        """ Erzeugt ein Rechteck. Es koennen 2 Ecken uebergeben werden.
        Diese werden als das linke untere und das rechte obere Eck 
        des Rechtecks interpretiert. Dieses steht dann waagrecht
        Werden 4 Ecken uebergeben, so sind diese die Ecken des Rechtecks.
        >>> print Rect(array([0,1]), array([1,2]))
        Rect(
            [0 1]
            [1 1]
            [1 2]
            [0 2]
        )
        >>> print Rect(array([0,0]), array([1,0]), array([1,1]), array([0,1]))
        Rect(
            [0 0]
            [1 0]
            [1 1]
            [0 1]
        )
        """
        self._senkrecht = False
        if len(objekte) == 4:
            Graphisches_Objekt.__init__(self, *objekte)
        elif len(objekte) == 2:
            Graphisches_Objekt.__init__(self,
                                        objekte[0],
                                        array([objekte[1][0], objekte[0][1]]),
                                        objekte[1],
                                        array([objekte[0][0], objekte[1][1]]),
                                        )
            self._senkrecht = True
        else:
            print objekte
            raise ValueError("Es muessen 2 oder 4 Ecken uebergeben werden")
        
        if not self._senkrecht and (
               self.objekte[0][1] == self.objekte[1][1] and
               self.objekte[1][0] == self.objekte[2][0] and
               self.objekte[3][1] == self.objekte[2][1]):
            self._senkrecht = True
    
    def is_senkrecht(self):
        """ gibt True zurueck, wenn das Dreieck senkrecht steht
        >>> Rect(array([0,0]), array([1,0]), array([1,1]), array([0,1])).is_senkrecht()
        True
        >>> Rect(array([0,0]), array([4,0]), array([1,1]), array([0,1])).is_senkrecht()
        False
        """
        return self._senkrecht
    
    def links_unten(self):
        if not self.is_senkrecht(): return self.um_rechteck().links_unten()
        return self.objekte[0]
    
    def rechts_oben(self):
        if not self.is_senkrecht(): return self.um_rechteck().rechts_oben()
        return self.objekte[2]
    
    def zeichne(self, g, bild_koord_trans):
        g.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*bild_koord_trans(i)) for i in self.objekte]))
    
    def zum_teil_in(self, objekt):
        """ TODO: noch schreiben"""
        return True
    
    def diag_size(self):
        return sqrt(sum([i**2 for i in self.rechts_oben() - self.links_unten()]))


class Dreieck(Graphisches_Objekt):
    def __init__(self, a, b, c):
        Graphisches_Objekt.__init__(self, [a, b, c])
    
    def zeichne(self, g, bild_koord_trans):
        g.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*bild_koord_trans(i)) for i in self.objekte]))
        
class Line(Graphisches_Objekt):
    def __init__(self, a, b):
        Graphisches_Objekt.__init__(self, [a,b])
        
    def zeichne(self, g, bild_koord_trans):
        g.drawLine(*[QtCore.QPointF(*bild_koord_trans(i)) for i in self.objekte])

class Transformation(object):
    """ Erzeugt eine linear-affine Transformation, 
    bestehend aus einer 2x2 Matrix und einem Vektor
    """
    
    def __init__(self, matrix, vector):
        self.matrix = matrix
        self.vector = vector
    
    
    def __call__(self, gra_Obj):
        """ Wendet die Transformation auf das graphische
        Objekt an.
        
        >>> T_0 = Transformation(array([[1,0],[0,1]]), array([0,0]))
        >>> G = Graphisches_Objekt([array([1,2]), array([2,3])])
        >>> print T_0(G)
        Graphisches_Objekt(
            [1 2]
            [2 3]
        )
        >>> T_1 = Transformation(array([[1,2],[3,4]]), array([1,2]))
        >>> G = Graphisches_Objekt([array([2,1])])
        >>> print T_1(G)
        Graphisches_Objekt(
            [ 5 12]
        )
        """
        if isinstance(gra_Obj, Graphisches_Objekt):
            return gra_Obj.transformiere(self)
        elif isinstance(gra_Obj, ndarray):
            return dot(self.matrix, gra_Obj) + self.vector
    
    
    def get_verkl_faktor(self, metrik='euklidisch'):
        """ gibt den Kontraktionsfaktor der Abbildung in einer
        Metrik an. Im Moment wird nur die euklidische Metrik
        unterstuetzt.
        Der Kontraktionsfaktor einer Abbildung f ist definiert als
        das kleinste c >= 0, fuer das mit der Distanzfunktion d gilt:
        d( f(x), f(y) ) <= c d(x,y), fuer alle x, y im Raum
        
        >>> f = Transformation(array([[.5, 0], [0, .5]]), array([1,1]))
        >>> f.get_verkl_faktor()
        0.5
        """
        if metrik == 'euklidisch':
            m = self.matrix
            q = linalg.det(m)**2
            p = sum(m*m)
            return sqrt((p + sqrt(p**2 - 4*q)) / 2)

def Transformation_by_angle(angle, factor=1, move=array([0,0])):
    return Transformation(dot(array([[cos(angle), sin(angle)],
                                     [sin(angle),-cos(angle)]]),
                              array([[factor, 0],
                                     [0, factor]])), move)


class Bild_Koord_Trans(Transformation):
    def __init__(self, width, height, min, max):
        """ min ist ein array (Fraktalkoordinaten der linken unteren
        Ecke), max die der rechten oberen Ecke
        """
        self.wh = array([width, height])
        self.min = min
        self.max = max
    
    def __call__(self, gra_Obj):
        """ 
        >>> G = Graphisches_Objekt(array([0.5, 1]), array([1, 2]))
        >>> T = Bild_Koord_Trans(100,200, array([0,0]), array([1,2]))
        >>> print T(G)
        Graphisches_Objekt(
            [ 50 100]
            [100   0]
        )
        """
        if isinstance(gra_Obj, Graphisches_Objekt):
            return gra_Obj.transformiere(self)
        elif isinstance(gra_Obj, ndarray):
            erg = (gra_Obj - self.min) * self.wh / (self.max - self.min)
            erg[1] = self.wh[1] - erg[1]None, progress=False):
        """ Zeichnet das Fraktal mit der uebergebenen Anzahl Pixeln in
        das uebergebene Bild
        """


            return erg.round().astype(int32)
    
    def get_verkl_faktor(self):
        return self.wh / (self.max - self.min)
    
    def pixel_size_diag(self):
        u""" Gibt die Länge der Diagonale eines Pixels zurück
        
        >>> T = Bild_Koord_Trans(200, 100, array([-1, -1]), array([1, 1]))
        >>> T.pixel_size_diag()
        .000125
        """
        return sqrt(sum([i**2 for i in (self.max.astype(float) - self.min) / self.wh]))



class Punkt(Graphisches_Objekt):
    def __init__(self, pos, r=None):
        Graphisches_Objekt.__init__(self, pos)
        self.r = r
    
    def zeichne(self, g, bild_koord_trans):
        r1, r2 = bild_koord_trans.get_verkl_faktor() * self.r
        g.drawEllipse(QtCore.QPointF(*bild_koord_trans(self.objekte[0])),r1, r1)
    
    def transformiere(self, transformation):
        erg = Graphisches_Objekt.transformiere(self, transformation)
        erg.r = self.r
        return erg
    
    def um_rechteck(self):
        """ ueberschreibt die Implementation von Graphisches_Objekt
        aus Geschwindigkeitsgruenden"""
        r = array([self.r, self.r])
        pos = self.objekte[0]
        return Rect(pos - r, pos + r)
