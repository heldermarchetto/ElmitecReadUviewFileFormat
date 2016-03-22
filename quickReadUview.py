# -*- coding: utf-8 -*-
"""
Copyright 2016 Dr. Helder Marchetto
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import struct
import numpy as np
def getUviewImg(fn):
    with open(fn, mode='rb') as file:
        fileContent = file.read()
    #define some constants useful later on
    headerSize = 104 #headerSize is always 104 bytes
    imageHeaderSize = 288
    recipeBlockSize = 128
    
    #get the image width and height
    imageWidth  = struct.unpack('h',fileContent[40:42])[0]
    imageHeight = struct.unpack('h',fileContent[42:44])[0]
    #get the image version to find the recipe size
    UK_version  = struct.unpack('h',fileContent[22:24])[0]
    if UK_version >= 7:
        attachedRecipeSize = struct.unpack('h',fileContent[46:48])[0]
        hasRecipe = attachedRecipeSize > 0
    else:
        attachedRecipeSize = struct.unpack('h','\x00\x00')[0]
        hasRecipe = attachedRecipeSize > 0
    #define the filePointer length
    if hasRecipe:
        filePointer = headerSize+recipeBlockSize #recipe is always 128 bytes
    else:
        filePointer = headerSize
    #the attachedMarkupSize is zero unless otherwise
    attachedMarkupSize = struct.unpack('h',fileContent[filePointer+22:filePointer+24])[0]
    hasAttachedMarkup = attachedMarkupSize != 0
    if hasAttachedMarkup:
        attachedMarkupSize = 128*((attachedMarkupSize/128)+1)
    else:
        attachedMarkupSize = 0
    LEEMDataVersion = struct.unpack('h',fileContent[filePointer+26:filePointer+28])[0]
    totalHeaderSize = headerSize + attachedRecipeSize + imageHeaderSize + attachedMarkupSize + LEEMDataVersion
    return np.reshape(struct.unpack(str(imageWidth*imageHeight)+'H',fileContent[totalHeaderSize:]), (imageWidth, imageHeight))

