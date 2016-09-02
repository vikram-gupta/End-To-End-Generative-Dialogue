# -*- coding: utf-8 -*-
from random import shuffle

data_name="Jerry"

validationSplit = 0.1

totalDataFilePath = "../data/"+data_name+"/input/"+data_name+"Full.txt"
trainDataFilePath = "../data/"+data_name+"/input/"+data_name+"Train.txt"
valDataFilePath = "../data/"+data_name+"/input/"+data_name+"Validation.txt"

trainDataFileHandle = open(trainDataFilePath,"w")
valDataFileHandle = open(valDataFilePath,"w")

totalDataFileHandle = open(totalDataFilePath,"r")
totalData = totalDataFileHandle.readlines()
totalDataLength = len(totalData)

print "Total Data Length is "+str(totalDataLength)

valDataLength = int(validationSplit*totalDataLength)
trainDataLength = totalDataLength - valDataLength

print "Train Data Length is "+str(trainDataLength)
print "Val Data Length is "+str(valDataLength)

shuffle(totalData)

trainData = totalData[:trainDataLength]
validationData = totalData[trainDataLength:]

for it in trainData:
    trainDataFileHandle.write(it)

for it in validationData:
    valDataFileHandle.write(it)

trainDataFileHandle.close()
valDataFileHandle.close()

print "Done"