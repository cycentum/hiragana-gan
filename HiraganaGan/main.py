###
#	Generating hiraganas (Japanese characters) with conditional GAN.
# https://github.com/cycentum/KanaGan
# (c) 2019 Takuya KOUMURA.
# MIT LICENSE.
###

import numpy as np
from numpy import uint8, uint32, int32, float32, float64, newaxis
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from collections import defaultdict
import itertools
from chainer import Variable, functions, optimizers, serializers
import cupy
import chainer

from net import DCGANGenerator, DCGANDiscriminator


def removeLineComment(line, target="#"):
	if target in line:
		index=line.index(target)
		line=line[:index]
	return line


def readValidFontChar(file):
	validFontChar={}
	with open(file, "r", encoding="utf8") as f:
		for line in f:
			line=line.strip()
			line=removeLineComment(line)
			if len(line)<0: continue
			line=line.split("\t")
			chars=line[1:]
			chars=set(chars)
			validFontChar[line[0]]=chars
	return validFontChar
			

def makeImageLabel(numImages, fonts, chars, fontChars, imageSize, fontSizeRange, centerShiftRange):
	images=np.empty((numImages, imageSize[1], imageSize[0]), uint8)
	labels=np.empty(numImages, int32)
	numValid=0
	while numValid<numImages:
		remainingSize=numImages-numValid
		fontIndex=np.random.choice(len(fonts), remainingSize)
		charIndex=np.random.choice(len(chars), remainingSize)
		fontSize=np.random.randint(fontSizeRange[0], fontSizeRange[1], remainingSize)
		centerShift=np.random.randint(centerShiftRange[0], centerShiftRange[1], (remainingSize,2))
		for ii,(fi,chi,fs,cs) in enumerate(zip(fontIndex, charIndex, fontSize, centerShift)):
			f=fonts[fi]
			ch=chars[chi]
			if ch not in fontChars[f]: continue
			im=Image.new("L", tuple(imageSize), color=0)
			font=ImageFont.truetype(str(DIR_FONT/f), size=fs)
			charSize=font.getsize(ch)
			pos=(imageSize-charSize)//2+cs
			draw=ImageDraw.Draw(im)
			draw.text(pos, ch, fill=255, font=font)
			images[numValid]=np.asarray(im)
			labels[numValid]=chi
			numValid+=1
	return images, labels


def makeLabelLatent(labels, numChars):
	inputLabel=-np.ones((len(labels), numChars), float32)
	for ii,la in enumerate(labels): inputLabel[ii, la]=1
	return inputLabel
	
	
def tileImage(images, numH, numV):
	'''
	@param images: shape=(batch, ..., height, width)
	'''
	imageSize=images.shape[-2:]
	assert numH*numV==images.shape[0]
	images=images.reshape(numV, numH, imageSize[1], imageSize[0]).transpose(0, 2, 1, 3).reshape(numV*imageSize[1], numH*imageSize[0])
	return images
	

def saveGeneratedGenerator(numH, numV, yGen, generator, fileImage, fileGenerator, labels, chars, fileLabel, discriminator=None, fileDiscriminator=None):
	if isinstance(yGen, Variable): yGen=yGen.data
	yGen=cupy.asnumpy(yGen)
	yGen=(yGen*256).astype(uint8)
	yGen=tileImage(yGen, numH, numV)
	with open(fileImage, "wb") as f: Image.fromarray(yGen, "L").save(f)
	if generator is not None:
		serializers.save_hdf5(fileGenerator, generator)
	if discriminator is not None:
		serializers.save_hdf5(fileDiscriminator, discriminator)
	if fileLabel is not None:
		with open(fileLabel, "w", encoding="utf8") as f:
			for li,la in enumerate(labels):
				if li%numH==numH-1: print(chars[la], file=f)
				else: print(chars[la], file=f, end="\t")


def setupNet(net, gpuId):
	net.to_gpu(gpuId)
# 	optGen=optimizers.Adam()
	opt=optimizers.Adam(alpha=0.0002, beta1=0.0, beta2=0.9) #parameters are from https://github.com/pfnet-research/chainer-gan-lib/blob/master/train.py
	opt.setup(net)
	return opt
	

def setupGenerator(latentSize, numChars, gpuId, fileParam=None):
	generator=DCGANGenerator(n_hidden=latentSize+numChars, out_channel=1, output_activation=functions.sigmoid)
	if fileParam is not None: serializers.load_hdf5(fileParam, generator)
	optGen=setupNet(generator, gpuId)
	return generator, optGen


def setupDiscriminator(numChars, gpuId):
	discriminator=DCGANDiscriminator(in_channel=1+numChars)
	optDis=setupNet(discriminator, gpuId)
	return discriminator, optDis


def backward(net, er, opt):
	net.cleargrads()
	er.backward()
	opt.update()
	er.unchain_backward()


def catLabelToImage(x, labels, numChars):
	xp=cupy.get_array_module(x.data)
	labelX=np.zeros((x.shape[0], numChars, x.shape[-2], x.shape[-1]), float32)
	for ii,la in enumerate(labels): labelX[ii, la]=1
	labelX=Variable(xp.asarray(labelX))
	x=functions.concat((x, labelX), axis=1)
	return x


def catLabelToLatent(z, labels, numChars):
	xp=cupy.get_array_module(z)
	labelZ=np.zeros((z.shape[0], numChars), float32)
	for ii,la in enumerate(labels): labelZ[ii, la]=1
	z=xp.concatenate((z, labelZ), axis=1)
	return z


def trainGan(lossType, numEpoch):
	fontChars=readValidFontChar(FILE_VALID_FONT_CHAR)
	chars=sorted(set(itertools.chain.from_iterable([ch for font,ch in fontChars.items()])))
	fonts=sorted(fontChars)
	imageSize=np.array((32, 32))
	fontSizeRange=(imageSize[0]-10, imageSize[0]-2)
	centerShiftRange=(-2, +2)
	
	cupy.cuda.Device(GPU_ID).use()
	xp = cupy
	
	batchSize=64
	numH=8
	numV=8
	saveEpoch=numEpoch
	np.random.seed(201905051748%np.iinfo(uint32).max)
	
	latentSize=32
	generator, optGen=setupGenerator(latentSize, len(chars), GPU_ID)
	discriminator, optDis=setupDiscriminator(len(chars), GPU_ID)
	for epoch in itertools.count():
		if numEpoch>=0 and epoch>=numEpoch: break
		images,labels= makeImageLabel(batchSize, fonts, chars, fontChars, imageSize, fontSizeRange, centerShiftRange)
		xReal=images.reshape(batchSize, 1, imageSize[1], imageSize[0]).astype(float32)/255
		
		xReal = Variable(xp.asarray(xReal))
		xReal=catLabelToImage(xReal, labels, len(chars))
		y_real = discriminator(xReal)
		
		z=np.random.uniform(-1, 1, (batchSize, latentSize)).astype(float32)
		latentLabels=np.random.randint(0, len(chars), batchSize)
		z=catLabelToLatent(z, latentLabels, len(chars))
		z=Variable(xp.asarray(z))
		imFake= generator(z)
		xFake=catLabelToImage(imFake, latentLabels, len(chars))
		y_fake = discriminator(xFake)
		
		if lossType=="dcgan":
			lossDis = functions.mean(functions.softplus(-y_real))
			lossDis += functions.mean(functions.softplus(y_fake))
			
			lossGen = functions.mean(functions.softplus(-y_fake))
		elif lossType=="relgan":
			diff=y_real-y_fake.T
			lossDis=-functions.softplus(-diff) #log(sigmoid(diff))
			lossDis=-functions.mean(lossDis)
			lossGen=-functions.softplus(diff) #log(sigmoid(-diff))
			lossGen=-functions.mean(lossGen)
		
		generator.cleargrads()
		lossGen.backward()
		optGen.update()
		xFake.unchain_backward()
		
		discriminator.cleargrads()
		lossDis.backward()
		optDis.update()
		
		print("epoch", epoch, 'loss_gen', lossGen.data, 'loss_dis', lossDis.data, sep="\t")
		
		if epoch%saveEpoch==saveEpoch-1:
			dirResult=DIR_RESULT/lossType
			dirResult.mkdir(exist_ok=True, parents=True)
			fileImage=dirResult/("Epoch"+str(epoch)+".png")
			fileImage.parent.mkdir(exist_ok=True, parents=True)
			fileLabel=dirResult/("Epoch"+str(epoch)+".txt")
			fileGen=dirResult/("Epoch"+str(epoch)+".gen")
			saveGeneratedGenerator(numH, numV, imFake, generator, fileImage, fileGen, latentLabels, chars, fileLabel, None, None)


def generate(lossType, epoch):
	fontChars=readValidFontChar(FILE_VALID_FONT_CHAR)
	chars=sorted(set(itertools.chain.from_iterable([ch for font,ch in fontChars.items()])))
	
	cupy.cuda.Device(GPU_ID).use()
	xp = cupy
	np.random.seed(201905051927%np.iinfo(uint32).max)
	
	batchSize=len(chars) #48
	numH,numV=5,10
	latentSize=32
	dirResult=DIR_RESULT/lossType
	generator, _=setupGenerator(latentSize, len(chars), GPU_ID, dirResult/("Epoch"+str(epoch)+".gen"))
	
	for latentIndex in range(4):
		z=np.random.uniform(-1, 1, (1, latentSize)).astype(float32)
		z=np.repeat(z, batchSize, axis=0)
		latentLabels=np.arange(len(chars))
		z=catLabelToLatent(z, latentLabels, len(chars))
		with chainer.using_config('train', False), chainer.using_config("enable_backprop", False):
			z=Variable(xp.asarray(z))
			imGen= generator(z)
			
			fileImage=dirResult/("Latent"+str(latentIndex)+".png")
			fileImage.parent.mkdir(exist_ok=True, parents=True)
			imGen=xp.concatenate((imGen.data, xp.zeros_like(imGen)[:2]), axis=0)
			saveGeneratedGenerator(numH, numV, imGen, None, fileImage, None, latentLabels, chars, None, None, None)
		


if __name__=="__main__":
	DIR_HIRAGANA_GAN=Path(r"./")
	DIR_FONT=Path(r"C:\Windows\Fonts")
	FILE_VALID_FONT_CHAR=DIR_HIRAGANA_GAN/"ValidFontChar.txt"
	DIR_RESULT=DIR_HIRAGANA_GAN/"Result"
	GPU_ID=0
	
	trainGan("dcgan", 3000)
	generate("dcgan", 3000-1)
	
	trainGan("relgan", 5000)
	generate("relgan", 5000-1)