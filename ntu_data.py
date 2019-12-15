
import os
import glob

seq_id = 'S001'
# cam_id = 'C001'
cam_id = '*'
pid = 'P001'
rid = 'R001'
aid = 'A007'

data_root = './data/NTU'

seq_path = '{seq_id}{cam_id}{pid}{rid}{aid}_rgb/img_00001.jpg'.format(**locals())
seq_path = os.path.join(data_root, seq_path)

print(seq_path)

def get_images(path):
    files = glob.glob(path)
    return files

images = get_images(seq_path)
print(images)