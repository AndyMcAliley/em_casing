import numpy as np
import unittest
from em_casing import halfspace as chs
from empymod import bipole,dipole

freq = 0.125
con = 0.18
casing_con = 1e7
outer_radius=0.1095
inner_radius=0.0961
casing_length = 1500
num_segments = 300
segment_length = casing_length/num_segments
casing_area = np.pi*(outer_radius**2-inner_radius**2)

class Test_halfspace(unittest.TestCase):
    def test_dht(self):
        print('Guptasarma & Singh 140 agrees with Key 201')
        funii = lambda x:chs.fii(x,5,5,freq,con)
        result140 = outer_radius*chs.hankel_J1_140(funii,outer_radius)
        result201 = outer_radius*chs.hankel_J1_201(funii,outer_radius)
        self.assertTrue(np.isclose(result140,result201,atol=1e-20).all())

    def test_201_values(self):
        print('Check Key 201 filter values against empymod')
        from empymod.filters import key_201_2012
        dlf201 = key_201_2012()
        self.assertTrue(np.all(chs.Wab201[:,0]==dlf201.base))
        self.assertTrue(np.all(chs.Wab201[:,1]==dlf201.j0))
        self.assertTrue(np.all(chs.Wab201[:,2]==dlf201.j1))

    def test_intracasing_segment(self):
        print('VEB_Ez, Gij, and dipole agree for distant segments within one casing')
        zi = 52.5
        zj = 1352.5
        segment_length = 5
        zj1 = zj-segment_length/2
        zj2 = zj+segment_length/2
        gij = chs.Gij(zi,zj,segment_length,
                      frequency=freq,
                      background_conductivity=con,
                      outer_radius=outer_radius,
                      inner_radius=inner_radius)
        gij = np.conj(gij)
        epm_gij = dipole([0,0,zj],
                         [.01,0,zi],
                         depth=[0],
                         res=[3e14,1/con],
                         freqtime=freq,
                         ab=33,
                         ht='quad',
                         verb=0
                        )*casing_area*segment_length
        veb_gij = chs.VEB_Ez(0,0,zi,
                             xp=0,yp=0,zp1=zj1,zp2=zj2,
                             conductivity=con,
                             frequency=freq)*casing_area
        self.assertTrue(np.isclose(gij,epm_gij,rtol=1e-4,atol=1e-20))
        self.assertTrue(np.isclose(gij,veb_gij,rtol=1e-4,atol=1e-20))

    def test_HED_Ez(self):
        print('HED_Ez and dipole agree')
        epm_hed = dipole([0,0,1e-2],[1,0,100],
                         depth=[0],
                         res=[3e14,1/con],
                         freqtime=freq,
                         ab=31,
                         epermH=[0,1],
                         epermV=[0,1],
                         verb=0
                        )
        chs_hed = chs.HED_Ez(1,0,100,0,0,conductivity=con,frequency=freq)
        self.assertTrue(np.isclose(epm_hed,chs_hed,atol=1e-20))

    def test_VEB_Ez(self):
        print('VEB_Ez and VED_Ez agree with bipole')
        x1 = 0
        y1 = 0
        casing_length1 = 1500
        num_segments1 = 300
        x2 = 2000
        y2 = 1000
        casing_length2 = 1600
        num_segments2 = 200
        segment_length1 = casing_length1/num_segments1
        segment_length2 = casing_length2/num_segments2
        zi = 52.5
        zj = 1352.5
        segment_length = 5
        zj1 = zj-segment_length/2
        zj2 = zj+segment_length/2
        epm_veb = bipole(src=[x2,x2,y2,y2,zj-segment_length2,zj+segment_length2],
                         rec=[x1,x1,y1,y1,zi,zi+.01],
                         depth=[0],
                         res=[3e14,1/con],
                         freqtime=freq,
                         verb=0
                        )
        chs_veb = chs.VEB_Ez(x1,y1,zi,
                             xp=x2,
                             yp=y2,
                             zp1=zj-segment_length2/2,
                             zp2=zj+segment_length2/2,
                             conductivity=con,
                             frequency=freq
                            )
        chs_ved = chs.VED_Ez(x1,y1,zi,
                             xp=x2,yp=y2,zp=zj,
                             conductivity=con,
                             frequency=freq
                            )
        self.assertTrue(np.isclose(epm_veb,chs_ved,rtol=1e-3,atol=1e-20))
        self.assertTrue(np.isclose(epm_veb*segment_length2,chs_veb,rtol=1e-3,atol=1e-20))


if __name__ == '__main__':
  unittest.main()

