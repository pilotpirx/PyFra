# coding=utf8
from numpy import *


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
        return self.__class__([transf(i) for i in self.objekte])
    
    
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
            return dot(self.matrix,gra_Obj) + self.vector


class MVKM(object):
    """ Eine Mehrfach-Verkleinerungs-Kopier-Maschine. Sie
    wendet mehrere Transformationen auf ein Anfangsobjekt
    an.
    """
    
    def __init__(self, anfangs_objekt ,*transformationen):
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
        """ Gibt eine Liste graphischer Objekte zurück, die
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
    
    
    def zeichne_mit_tiefe(self, imag, punkte, bild_trans, 
                          area=None, progress=False):
        raise NotImplementedError()


class GVKM(object):
    def __init__(self, transformationen, wahrscheinlichkeiten = None):
        if isinstance(wahrscheinlichkeiten, list):
            if len(wahrscheinlichkeiten) != len(transformationen):
                raise Exception("Transf. und Wahrscheinlichk. haben nich die gl. Laenge ")
            self.wahrscheinl = wahrscheinlichkeiten
            self.transf = transformationen
        else:
            self.wahrscheinl = [i[1] for i in transformationen]
            self.transf = [i[0] for i in transformationen]
        self.wahrsch_summe = sum(self.wahrscheinl)
        import random
        self.rand = random
        self.punkt = array([0,0])
    
    
    def _get_next_trans(self):
        a = self.rand.uniform(0,self.wahrsch_summe)
        summe = 0
        i = 0
        while summe <= a:
            summe += self.wahrscheinl[i]
            i += 1
        return self.transf[i - 1]
    
    
    """ Zeichnet das Fraktal mit der übergebenen Anzahl Pixeln in
    das übergebene Bild
    """
    def zeichne_mit_tiefe(self, imag, punkte, bild_trans, 
                          area=None, progress=False):
        if progress:
            t = punkte // 100
        p = self.punkt
        for i in range(punkte):
            if progress and (i + 1) % t == 0:
                print "Zu %3i%% fertig" % ceil(float(i)/punkte*100)
            T = self._get_next_trans()
            p = T(p)
            try:
                imag.putpixel(bild_trans(p), 0)
            except Exception:
                pass


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
            erg[1] = self.wh[1] - erg[1]
            return erg.round().astype(int32)


class Rect(Graphisches_Objekt):
    def zeichne(self, g, bild_koord_trans):
        g.polygon([tuple(bild_koord_trans(i)) for i in self.objekte])


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '-t':
        import doctest
        doctest.testmod(verbose=True)
    else:
        from Image import *
        from ImageDraw2 import *
        im = Image.new("1", (10000, 10000), "white")
        g = ImageDraw.Draw(im)
        R = Rect(array([0,0]), array([1,0]), array([1,1]), array([0,1]))
        r = 2*pi/360.0
        mvkm = GVKM([Transformation(array([[0.85*cos(-r*2.5),-0.85*sin(-r*2.5)],
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
                                       array([0.0 ,0.0]))],
                        [0.75, 0.08, 0.08, 0.08])
        T = Bild_Koord_Trans(10000, 10000, array([-6,-1]), array([6,11]))
        mvkm.get_Bild_in_area_mit_tiefe(im, 10000000, T, progress=True)
        #mvkm.get_graphisches_objekt_tiefe(10).zeichne(g,T)
        im.save("blubb2.png", "PNG")
