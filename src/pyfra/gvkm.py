# -*- coding: utf-8 -*-


from mathe_graphic import *
from mvkm import MVKM

class GVKM(MVKM):
    def __init__(self, transformationen, wahrscheinlichkeiten = None):
        if isinstance(wahrscheinlichkeiten, list):
            if len(wahrscheinlichkeiten) != len(transformationen):
                raise Exception("Transf. und Wahrscheinlichk. haben nicht " +
                         "die gl. Laenge")
            self.wahrscheinl = wahrscheinlichkeiten
            self.transf = transformationen
        else:
            if isinstance(transformationen[0], list):
                self.wahrscheinl = [i[1] for i in transformationen]
                self.transf = [i[0] for i in transformationen]
            else:
                self.wahrscheinl = [1] * len(transformationen)
                self.transf = transformationen
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
    
    
    def get_pixmap(self, array, punkte, bild_trans, 
                         area=None, progress=False):
        """ Zeichnet das Fraktal mit der uebergebenen Anzahl Pixeln in
        das uebergebene Bild
        """

        if progress:
            t = punkte // 100
        p = self.punkt
        for i in range(punkte):
            if progress and (i + 1) % t == 0:
                print "Zu %3i%% fertig" % ceil(float(i)/punkte*100)
            T = self._get_next_trans()
            p = T(p)
            pixel = bild_trans(p)
            imag.setPixel(pixel[0], pixel[1], 1)

    def has_vector_format(self):
        return False

    def get_vector(self, args*):
        raise NotImplementedError("Can't save in vector format")
