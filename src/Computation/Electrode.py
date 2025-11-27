import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy  import signal
import math
from scipy.io import loadmat
#plt.rcParams.update({'font.size': 26})


class ElectrodeModel:
    def __init__(self, re=62.5, sigma=33e-5, Etype=0, coated=0, th=2, Rt=0, Ct=0, Ren=0, day=0,cyl=0,h=2000):
        # Electrode properties
        # re ---> electrode radius in micrometers
        # sigma ---> Tissue/Electrolyte conductivity in ohm - 1.mm - 1
        # Etype is for electrode material
        # coated ----> 0: for model 1 non-coated electrodes 1: for model 2 coated electrodes (PEDOT equivalent circuit)
        # th --->  is the coating thickness
        # Rt, Ct and Ren are the Encapsulation tissue equivalent circuit parameters
        # day is the day post implantation for encapsulation tissue
        self.h=h
        self.coated = coated
        self.day = day
        self.re = re
        self.sigma = sigma
        self.Cs = 1 * 1e-8  # shunt capacity (shunt capacity to the ground value usually standard value)
        if cyl==1:
            self.s_e= 2* np.pi * self.re *self.h
        else:
            self.s_e = np.pi * self.re ** 2  # surface area in micrometer^2
        self.s_e_c = self.s_e + 2*np.pi*self.re*th  # total surface area with coating thickness
        # Compute the spreading resistance value
        if coated == 0:
            self.Rs = 1 / (self.sigma * 1e-3 * 4 * self.re )
        else:
            self.Rs = np.sqrt(np.pi)/(self.sigma * 1e-3 * 4 * np.sqrt(self.s_e_c) )

        self.w = np.logspace(0, 4, 50)  # frequency vector
        self.Rt = Rt
        self.Ct = Ct
        self.Ren = Ren

        # 0     ->      carbon_non-coated_mean , radius = 200 µm
        # 1     ->      gold_non-coated_mean , radius = 62.5 µm
        # 2     ->      stainless_steel_non-coated_mean, radius = 62.5 µm
        # 3     ->      carbon_coated_5s, radius = 200 µm
        # 4     ->      carbon_coated_10s, radius = 200 µm
        # 5     ->      carbon_coated_50s, radius = 200 µm
        # 6     ->      gold_coated_5s, radius = 62.5 µm
        # 7     ->      gold_coated_10s, radius = 62.5 µm
        # 8     ->      gold_coated_50s, radius = 62.5 µm

        # Model 1 : non-coated
        Rct_1 = {0: 1.5524e+14, 1: 4.9222e+12, 2: 1.4671e+14,3:4.48e+13 }
        Cdl_1 = {0: 2.4794e-15, 1: 8.7817e-14, 2: 1.6904e-14,3:  2.72e-13}
        n_1   = {0: 0.8833    , 1: 0.8983    , 2: 0.8032    ,3: 0.9}

        # Model 2 : coated
        Cdl_2 = {3: 5.9054e-12, 4: 1.7746e-10, 5: 6.6972e-10, 6: 8.1400e-12, 7: 1.7699e-11, 8: 1.9606e-11}
        Rct_2 = {3: 7.1506e+08, 4: 1.7662e+09, 5: 3.5324e+09, 6: 6.4574e+07, 7: 1.3586e+08, 8: 9.8960e+06}
        CD_2  = {3: 2.7937e-12, 4: 7.7506e-11, 5: 7.9847e-11, 6: 1.6936e-12, 7: 7.7726e-13, 8: 5.5107e-11}
        n_2   = {3: 0.9617    , 4: 0.9680    , 5: 0.9779    , 6: 0.9389    , 7: 0.8876    , 8: 0.9693    }
        self.rough = 1.58

        if coated == 0:
            self.c = Cdl_1.get(Etype)  # F/mm2
            self.r = Rct_1.get(Etype)
            self.n = n_1.get(Etype)
            self.Rct = self.r / self.s_e
            self.Cdl = self.c * self.s_e
        else:
            self.cc = Cdl_2.get(Etype)  # F/mm2
            self.rp = Rct_2.get(Etype)
            self.n = n_2.get(Etype)
            self.q = CD_2.get(Etype)
            self.Rp = self.rp / self.s_e_c
            self.Cc = self.cc * self.s_e_c
            self.Qc = self.q * self.s_e_c

    def Zelec(self, w):
        Ze = np.zeros(shape=(1, len(w)), dtype=complex)
        for i in range(0, len(w)):
            Ze[0, i] = self.Rs + 1 / (1/self.Rct + (self.Cdl * 2 * np.pi * w[i] * 1j) ** self.n)
        return Ze

    def Zelec_2(self, w):
        Ze = np.zeros(shape=(1, len(w)), dtype=complex)
        for i in range(0, len(w)):
            Ze[0, i] = self.Rs + ((self.Rp + (self.Qc * 2 * np.pi * w[i] * 1j) ** self.n) / ((self.Rp * self.Cc * 2 * np.pi * w[i] * 1j) + (((self.Qc * 2 * np.pi * w[i] * 1j) ** self.n) * self.Cc * 2 * np.pi * w[i] * 1j)))
        return Ze

    def TF(self, w):
        # He = np.zeros(shape=(1, len(w)), dtype=complex)
        # for i in range(0, len(w)):
        #     He[0, i] = 1 / ((self.Rs * self.Cs * 2 * np.pi * w[i] * 1j) + ((self.Cs * 2 * np.pi * w[i] * 1j) / (1 / self.Rct + (self.Cdl * 2 * np.pi * w[i] * 1j) ** self.n)) + 1)
        He = 1 / ((self.Rs * self.Cs * 2 * np.pi * w * 1j) + ((self.Cs * 2 * np.pi * w * 1j) / (1 / self.Rct + (self.Cdl * 2 * np.pi * w * 1j) ** self.n)) + 1)


        return He

    def TF_2(self, w):
        He = 1 / (1 + self.Cs * 2 * 1j * w * np.pi * (
                              self.Rs + (1 + self.Rp * self.Qc * (2 * 1j * w * np.pi) ** self.n) / (
                                  self.Qc * (2 * 1j * w * np.pi) ** self.n + self.Cc * 1j * 2 * np.pi * w * (
                                      1 + self.Rp * self.Qc * (2 * 1j * w * np.pi) ** self.n))))
        He[0] = 1
        return He

    def GetVelec(self, V, Fs):
        #  He is the electrode interface transfer function
        #  V is the LFP of the electrode
        Vf = fft(V)
        freq = np.fft.fftfreq(len(V), d=1/Fs)
        #       print(freq)
        if self.coated==0:
            He = self.TF(freq)
        else:
            He = self.TF_2(freq)
        #       plt.plot(np.abs(He[0,:]),'.')
        #       plt.show()
        Velec = ifft(He * Vf)
        return Velec.real

    def GetVelecInv(self, V, Fs):
        #  He is the electrode interface transfer function
        #  V is the LFP of the electrode
        Vf = fft(V)
        freq = np.fft.fftfreq(len(V), d=1/Fs)
        #       print(freq)
        He = self.TF(freq)
        #       plt.plot(np.abs(He[0,:]),'.')
        #       plt.show()
        Velec = ifft(Vf/He )
        return Velec.real


e_SS = ElectrodeModel(Etype=0,re=62.5,coated=0)
e_Pt= ElectrodeModel(Etype=3,re=150,coated=0,cyl=1,h=2000)
# #
# #
Z_SS = e_SS.Zelec(e_SS.w)
H_SS = e_SS.TF(e_SS.w)
Z_Pt = e_Pt.Zelec(e_Pt.w)
H_Pt = e_Pt.TF(e_Pt.w)
# #
# #
# plt.plot(e_SS.w, np.log10(abs(Z_Pt[0, :])), color='goldenrod',marker='s', label='Pt', markersize=10)
# plt.plot(e_SS.w, np.log10(abs(Z_SS[0, :])), color='slateblue',marker='>', label='SS', markersize=10)
# # #
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.tight_layout()
# plt.xscale('log')
# # #
# plt.ylabel('Log10|Z|', fontsize=30)
# plt.xlabel('Frequency(Hz)', fontsize=30)
# plt.legend()
# plt.show()
# # #
# #
# plt.plot(e_SS.w, np.log10(abs(H_Pt)), color='goldenrod',marker='s', label='Pt', markersize=10)
# plt.plot(e_SS.w, np.log10(abs(H_SS)), color='slateblue',marker='>', label='SS', markersize=10)
# #
# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30)
# plt.tight_layout()
# plt.xscale('log')
# #
# #
# plt.ylabel('Log10|H|', fontsize=30)
# plt.xlabel('Frequency(Hz)', fontsize=30)
# plt.legend(fontsize=30)
# plt.show()
# #
# #
# plt.plot(e_SS.w, -np.angle(H_Pt) * 180 / math.pi, color='goldenrod',marker='s', linewidth=0,markersize=10)
# plt.plot(e_SS.w, -np.angle(H_SS) * 180 / math.pi, color='slateblue',marker='>', linewidth=0,markersize=10)
#
# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30)
# plt.tight_layout()
# plt.xscale('log')
# #
# #
# plt.ylabel('-Phase(H)(°)', fontsize=30)
# plt.xlabel('Frequency(Hz)', fontsize=30)
# # plt.legend(fontsize=30)
# plt.show()