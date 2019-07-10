# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:46:01 2016

@author: Dr. Helder Marchetto

"""

import time
import struct
import numpy as np

class fileHeader():
    def __init__(self, fc, verbose=False):
        posGen = (i for i, e in enumerate(fc) if e == 0)
        stPos = 0
        endPos = next(posGen)
        self.UK_id = "".join(map(chr, fc[stPos:endPos]))
        self.UK_size         = int.from_bytes(fc[20:22], byteorder='little')
        self.UK_version      = int.from_bytes(fc[22:24], byteorder='little')
        self.UK_bitsPerPixel = int.from_bytes(fc[24:26], byteorder='little')
        if self.UK_version >= 8:
            self.UK_cameraBitsPerPixel = int.from_bytes(fc[26:28], byteorder='little')
            self.MCPDiameterInPixels   = int.from_bytes(fc[28:30], byteorder='little')
            self.hBinning              = int.from_bytes(fc[30:31], byteorder='little')
            self.vBinning              = int.from_bytes(fc[31:32], byteorder='little')
        else:
            voidNr = int(0)
            self.UK_cameraBitsPerPixel = voidNr
            self.MCPDiameterInPixels   = voidNr
            self.hBinning              = voidNr
            self.vBinning              = voidNr
        #8 bytes are spared!
        if self.UK_version >= 2:
            self.imageWidth  = int.from_bytes(fc[40:42], byteorder='little')
            self.imageHeight = int.from_bytes(fc[42:44], byteorder='little')
            self.nrImages    = int.from_bytes(fc[44:46], byteorder='little')
        else:
            voidNr = int(0)
            self.imageWidth  = voidNr
            self.imageHeight = voidNr
            self.nrImages    = voidNr
        if self.UK_version >= 7:
            self.attachedRecipeSize = int.from_bytes(fc[46:48], byteorder='little')
        else:
            self.attachedRecipeSize = int(0)
        self.hasRecipe = self.attachedRecipeSize > 0
        self.fixedRecipeSize = 128
        if self.hasRecipe:
            self.headerSize = 104+128
        else:
            self.headerSize = 104

        if verbose:
            print('headerSize = %i' %self.headerSize)
            print('UK_id = '+self.UK_id)
            print('UK_size = '+str(self.UK_size))
            print('UK_version = '+str(self.UK_version))
            print('UK_bitsPerPixel = '+str(self.UK_bitsPerPixel))
            print('UK_cameraBitsPerPixel = '+str(self.UK_cameraBitsPerPixel))
            print('MCPDiameterInPixels = '+str(self.MCPDiameterInPixels))
            print('hBinning = '+str(self.hBinning))
            print('vBinning = '+str(self.vBinning))
            print('imageWidth = '+str(self.imageWidth))
            print('imageHeight = '+str(self.imageHeight))
            print('nrImages = '+str(self.nrImages))
            print('attachedRecipeSize = '+str(self.attachedRecipeSize))
            print('hasRecipe = '+str(self.hasRecipe))

class imageHeader():
    def __init__(self, fc, fh, verbose=False):
        filePointer = fh.headerSize
        self.imageHeadersize    = int.from_bytes(fc[filePointer   :filePointer+ 2], byteorder='little')
        self.version            = int.from_bytes(fc[filePointer+ 2:filePointer+ 4], byteorder='little')
        self.colorScaleLow      = int.from_bytes(fc[filePointer+ 4:filePointer+ 6], byteorder='little')
        self.colorScaleHigh     = int.from_bytes(fc[filePointer+ 6:filePointer+ 8], byteorder='little')
        self.imageTime          = int.from_bytes(fc[filePointer+ 8:filePointer+16], byteorder='little')
        self.maskXShift         = int.from_bytes(fc[filePointer+16:filePointer+18], byteorder='little')
        self.maskYShift         = int.from_bytes(fc[filePointer+18:filePointer+20], byteorder='little')
        self.rotateMask         = int.from_bytes(fc[filePointer+20:filePointer+22], byteorder='little', signed=False)
        self.attachedMarkupSize = int.from_bytes(fc[filePointer+22:filePointer+24], byteorder='little')
        self.hasAttachedMarkup = self.attachedMarkupSize != 0
        if self.hasAttachedMarkup:
            self.attachedMarkupSize = 128*((self.attachedMarkupSize//128)+1)
        self.spin               = int.from_bytes(fc[filePointer+24:filePointer+26], byteorder='little')
        self.LEEMDataVersion    = int.from_bytes(fc[filePointer+26:filePointer+28], byteorder='little')
        filePointer = filePointer+28
        if self.version > 5:
            self.LEEMData          = struct.unpack('240c',fc[filePointer:filePointer+240])
            filePointer = filePointer+240
            self.appliedProcessing = fc[filePointer]
            self.grayAdjustZone    = fc[filePointer+1]
            self.backgroundvalue   = int.from_bytes(fc[filePointer+2:filePointer+4], byteorder='little', signed=False)
            self.desiredRendering  = fc[filePointer+4]
            self.desired_rotation_fraction = fc[filePointer+5]
            self.rendering_argShort  = int.from_bytes(fc[filePointer+6:filePointer+8], byteorder='little')
            self.rendering_argFloat  = struct.unpack('f',fc[filePointer+8:filePointer+12])[0]
            self.desired_rotation    = int.from_bytes(fc[filePointer+12:filePointer+14], byteorder='little')
            self.rotaion_offset      = int.from_bytes(fc[filePointer+14:filePointer+16], byteorder='little')
            #spare 4
        else:
            voidNr = int(0)
            self.LEEMData          = struct.unpack('240c','\x00'*240)
            self.appliedProcessing = b''
            self.grayAdjustZone    = b''
            self.backgroundvalue   = voidNr
            self.desiredRendering  = b''
            #spare 1
            self.rendering_argShort  = voidNr
            self.rendering_argFloat  = 0.0
            self.desired_rotation    = voidNr
            self.rotaion_offset      = voidNr

        if verbose:
            print('imageHeadersize = '+str(self.imageHeadersize))
            print('version = '+str(self.version))
            print('colorScaleLow = '+str(self.colorScaleLow))
            print('colorScaleHigh = '+str(self.colorScaleHigh))
            print('imageTime = '+str(self.imageTime))
            print('maskXShift = '+str(self.maskXShift))
            print('maskYShift = '+str(self.maskYShift))
            print('rotateMask = '+str(self.rotateMask))
            print('attachedMarkupSize = '+str(self.attachedMarkupSize))
            print('hasAttachedMarkup = '+str(self.hasAttachedMarkup))
            print('spin = '+str(self.spin))
            print('LEEMDataVersion = '+str(self.LEEMDataVersion))

def readUview(fc, fh, ih):
    totalHeaderSize = fh.headerSize + ih.imageHeadersize + ih.attachedMarkupSize + ih.LEEMDataVersion
    return np.reshape(struct.unpack(str(fh.imageWidth*fh.imageHeight)+'H',fc[totalHeaderSize:]), (fh.imageHeight, fh.imageWidth))


#get file name
datFileName = r'K:\Data\SMART-2\2019\0507_HM_MP_TS_FU-Berlin\20190507a001.dat'

with open(datFileName, mode='rb') as file: # b is important -> binary
    fileContent = file.read()

fh = fileHeader(fileContent)
ih = imageHeader(fileContent,fh)
img = readUview(fileContent, fh, ih)

import matplotlib.pyplot as plt
plt.imshow(img, cmap=plt.cm.gray)

