import torch
import os

dictdir='./checkpoints'
testsave = os.path.join(dictdir,'101_DAVS_checkpoint.pth.tar')
trainsave = os.path.join(dictdir, '406_0_Speech_reco_checkpoint.pth.tar')

testdict = torch.load(testsave)
traindict = torch.load(trainsave)

print(testdict.keys())
print(traindict.keys())