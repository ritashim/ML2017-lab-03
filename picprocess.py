import pickle
import PIL.Image
import numpy as np

def to_array (src,dest,amt,prefix):
    res = np.array([])
    for i in range (amt):
        img = PIL.Image.open(src+prefix+conv_index(i)+".jpg")
        img = img.resize((24,24))
        img = img.convert('L') # Luma transform
        res = np.concatenate((res,np.array(img).reshape(24*24)))
    res = res.reshape((amt,24*24))
    with open(dest+"/opt",mode='wb') as d:
        pickle.dump(res,d)

def conv_index(amt):
    if(amt<10):
        return "00"+str(amt)
    if(amt<100):
        return "0"+str(amt)
    return str(amt)

face_path = "./datasets/original/face/"
nonface_path = "./datasets/original/nonface/"
face_path_conv = "./datasets/processed/face"
nonface_path_conv = "./datasets/processed/nonface"

to_array(face_path,face_path_conv,500,"face_")
to_array(nonface_path,nonface_path_conv,500,"nonface_")
