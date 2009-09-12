from numpy import array, ndarray, pi, sin, cos, \
                  log, ceil, zeros, dot, int32
from PyQt4 import QtGui, QtCore
from numpy.linalg import det
from pprint import pprint, pformat
import numpy as np

cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t


class Graphisches_Objekt(object):
    """ Ein graphisches Objekt. Es ist aus anderen 
    graphischen Objekten zusammengesetzt, die dem Konstruktor
    uebergeben werden. Darunter koennen auch arrays sein, die
    als Punkte interpretiert werden. Ein graphisches Objekt
    mit solchen arrays kann aber nicht direkt gezeichnet
    werden, dafuer muss eine eigene Unterklasse erzeugt 
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
        und gibt ein solchermassen veraendertes graphisches
        Objekt zurueck. Das urspruengliche graphische Objekt 
        bleibt unveraendert
        
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



class GVKM(MVKM):
    def __init__(self, transformationen, wahrscheinlichkeiten = None):
        if isinstance(wahrscheinlichkeiten, list):
            if len(wahrscheinlichkeiten) != len(transformationen):
                raise Exception("Transf. und Wahrscheinlichk. haben nicht " +
                         "die gl. Laenge")
            self.wahrscheinl = wahrscheinlichkeiten
            self.transf = transformationen
        else:
            self.wahrscheinl = [i[1] for i in transformationen]
            self.transf = [i[0] for i in transformationen]
        self.wahrsch_summe = sum(self.wahrscheinl)
        import random
        self.rand = random
        self.punkt = array([0, 0])
    
    
    def _get_next_trans(self):
        a = self.rand.uniform(0, self.wahrsch_summe)
        summe = 0
        i = 0
        while summe <= a:
            summe += self.wahrscheinl[i]
            i += 1
        return self.transf[i - 1]
    
    
    """ Zeichnet das Fraktal mit der uebergebenen Anzahl Pixeln in
    das uebergebene Bild
    """
    def zeichne_mit_tiefe(self, painter, punkte, bild_trans, 
                          area=None, progress=False):
        if progress:
            t = punkte // 100
        p = self.punkt
        for i in range(punkte):
            if progress and (i + 1) % t == 0:
                print "Zu %3i%% fertig" % ceil(float(i)/punkte*100)
            T = self._get_next_trans()
            p = T(p)
            painter.drawPoint(*bild_trans(p))

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
    
    def get_verkl_faktor(self):
        return self.wh / (self.max - self.min)



if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '-t':
        import doctest
        doctest.testmod(verbose=False)
    else:
        from Image import *
        from ImageDraw2 import *
        im = Image.new("1", (10000, 10000), "white")
        g = ImageDraw.Draw(im)
        R = Rect(array([0, 0]), array([1, 0]), array([1, 1]), array([0, 1]))
        r = 2*pi/360.0
        mvkm = GVKM([Transformation(array([[0.85*cos(-r*2.5), -0.85*sin(-r*2.5)], 
                                              [0.85*sin(-r*2.5), 0.85*cos(-r*2.5)]]), 
                                       array([0.0, 1.6])), 
                        Transformation(array([[0.3*cos(r*49), -0.34*sin(r*49)], 
                                              [0.3*sin(r*49), 0.34*cos(r*49)]]), 
                                       array([0.0, 1.6])), 
                        Transformation(array([[0.3*cos(r*120), -0.37*sin(-r*50)], 
                                              [0.3*sin(r*120), 0.37*cos(-r*50)]]), 
                                       array([0.0, 0.44])), 
                        Transformation(array([[0.0, 0.0], 
                                              [0.0, 0.16]]), 
                                       array([0.0 , 0.0]))], 
                        [0.75, 0.08, 0.08, 0.08])
        T = Bild_Koord_Trans(10000, 10000, array([-6, -1]), array([6, 11]))
        mvkm.zeichne_mit_tiefe(im, 10000000, T, progress=True)
        im.save("blubb2.png", "PNG")
        
