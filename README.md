# ElmitecReadUviewFileFormat
Import Uview files in Python

These procedures can be used to read uview file formats (.dat).

First read the file data as binary with:

with open('yourFileName, mode='rb') as file:
    fileContent = file.read()

Then process the fileContent with:

1) the fileHeader class:

myFileHeader = fileHeader(fileContent)

2) Then feed the result to the imageHeader class:

myImageHeader = imageHeader(fileContent,myFileHeader)

3) Now you can read the image with:

myImage = readUview(fileContent,myFileHeader,myImageHeader)

Now myImage is an image in numpy format.

Please send any comments, bugs or requests to me. Feel free to join/improve the python code.
