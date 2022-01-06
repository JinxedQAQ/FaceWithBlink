import os
from tqdm import tqdm
import imghdr

datasetdir='/home/h2/lrwdatasetalign'

phases = ['test','val']

for phase in phases:
    phase_path = os.path.join(datasetdir,phase)
    for word in tqdm(range(500)):
        word_path = os.path.join(phase_path, str(word))
        for mp4 in range(1000):
            mp4_path = os.path.join(word_path, str(mp4))
            if os.path.exists(mp4_path):
                frames_path = os.path.join(mp4_path, "align_face256")
                frames = os.listdir(frames_path)
                for frame in frames:
                    fn = os.path.join(frames_path, frame)
                    a = os.path.getsize(fn)
                    if a < 256:
                        print("ERR %s,%d,%d,%s"%(phase,word,mp4,frame))
                    with open(fn, "rb") as f:
                        f.seek(-2, 2)
                        buf = f.read()
                        valid=buf.endswith(b'\xff\xd9') or buf.endswith(b'\xae\x82') #or \
                            #buf.endswith(b'\x00\x3B') or buf.endswith(b'\x60\x82') #检测jpg图片完整性， 检测png图片完整性
                        buf.endswith(b'\x00\x00')
                    if not valid:
                        print("ERR %s,%d,%d,%s"%(phase,word,mp4,frame))
