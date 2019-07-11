# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:17:43 2019

@author: Helder
"""

import readUview as ru
import matplotlib.pyplot as plt

fn = r'K:\Data\SMART-2\2019\0507_HM_MP_TS_FU-Berlin\20190507a001.dat'

ruObj = ru.readUviewClass()
img = ruObj.getImage(fn)

plt.imshow(img, cmap=plt.cm.gray)
fn = r'K:\Data\SMART-2\2019\0507_HM_MP_TS_FU-Berlin\20190507b001.dat'

for p in ruObj.paramList:
    print(p)


