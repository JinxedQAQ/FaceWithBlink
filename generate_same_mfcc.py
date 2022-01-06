import shutil
import os

root = '/home/h2/Talkingface_BnoLip/inputaudios/057200190003'
source = os.path.join(root,'2.bin')

for i in range(3,50):
    tgt = os.path.join(root,str(i)+'.bin')
    shutil.copyfile(source,tgt)


source = os.path.join(root,'s2.bin')


for i in range(50,100):
    tgt = os.path.join(root,str(i)+'.bin')
    shutil.copyfile(source,tgt)
