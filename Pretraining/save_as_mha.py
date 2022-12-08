import numpy as np
import os
from PIL import Image
import pandas as pd
from tiger.io import write_image
import pickle


def read_DL_slice(path):
    #Note that for DeepLesion we must subtract 32768 
    # from pixel intensities to obtain HUs
    img = Image.open(path)
    np_img = np.array(img)
    return np_img - 32768
    

path_chansey = '/output/'
ar = path_chansey + 'Data/DeepLesion/images'
df = pd.read_csv(path_chansey+ 'Pretraining'  + "/DL_info_v3.csv", sep=';')
groups = df.groupby('Folder_name')
out_path = path_chansey + 'Data/volumes/'

filename_set = set()

for root, _, files in os.walk(ar):
    files.sort()
    print("root: ", root)
    #print("files: ", files)
    scan_name = root.split("/")[-1]
    try:
        lesion_files = groups.get_group(scan_name)['Key_slice_index']
    except:
        continue
    
    if files:
        for lesion_file in lesion_files:
            img_nr = int(lesion_file)
            start_s = img_nr - 1
            stop_s = img_nr + 1
        
         
            #print("start: ", start_s, ", stop: ", stop_s)
            if format(stop_s, "03")+'.png' in files:
                complete_img = np.array(
                    [
                        read_DL_slice(root+"/"+img)
                        for img in files if img.split(".")[-1] == "png"
                        and img.split(".")[0]
                        in [
                            format(i, "03")
                            for i in range(start_s, stop_s + 1)
                        ]
                    ]
                )
                
                if complete_img.shape[0] < 3:
                    print("error: ", filename, " has shape ", complete_img.shape)
                else:
                    filename = scan_name+"_"+format(img_nr, "03")
                    filename_set.add(filename)
                    write_image(out_path+filename+'.mha', complete_img)

with open(out_path+'filenames.pkl', 'w+') as f:
    pickle.dump(filename_set, f)
