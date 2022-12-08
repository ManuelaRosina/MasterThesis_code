from matplotlib import pyplot as plt
from tiger.io import read_image, write_image, read_dicom, ImageMetadata
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from collections import Counter

output_path = "/output/"
Data_folder = 'Data/Trac'
output_image_data_path = output_path+'Data/CT_scan_slices/'
id_information = pd.read_csv(output_path+Data_folder+'/matched_lesions.csv',index_col=0)
ct_scan_conversion_df = pd.read_csv(output_path+Data_folder+'/scan_information.csv',index_col=0)
output_data_path = output_path+'Data/CT_scan_slices/'
input_image_folder = '' #TODO: what path is this on cluster

"""
convert an image number to world z coordinate given the number and the filepath of the ct scan (e.g 2/st002/se011)
"""
def convert_ima_to_world(ima, file):
    m_numbers = ct_scan_conversion_df.loc[[file]]['WorldMatrix'][0].replace('[', '').replace(']', ' ').replace('\n', '').split(' ')
    m_numbers = [float(x) for x in m_numbers if x != '']
    WorldMatrix = np.array(m_numbers).reshape(4,4)
    total_slice = ct_scan_conversion_df.loc[[file]]['total_slice_nr'][0]
    vector = np.array([0,0,total_slice-ima,1]) #total_slice-ima
    worldZ = (WorldMatrix @ vector)[2]
    
    return worldZ

"""
convert world coordinate to ima
"""
def convert_world_to_ima(world, file):
    m_numbers = ct_scan_conversion_df.loc[[file]]['WorldMatrix'][0].replace('[', '').replace(']', ' ').replace('\n', '').split(' ')
    m_numbers = [float(x) for x in m_numbers if x != '']
    WorldMatrix = np.array(m_numbers).reshape(4,4)
    total_slice = ct_scan_conversion_df.loc[[file]]['total_slice_nr'][0]
    #vector = np.array([0,0,total_slice-ima,1]) #total_slice-ima
    #worldZ = (WorldMatrix @ vector)[2]
    
    A = np.linalg.inv(WorldMatrix)
    x = np.append(world, [1])
    #print(x)
    #print(A @ x)
    worldZ = A @ x
    worldZ[2] = total_slice-worldZ[2]
    
    return worldZ

"""
get the spathing between slices of a scan
"""
def get_slice_thickness(file):
    return id_information.loc[id_information['SeriesPath'] == file]['SliceThickness'].values.tolist()[0]

"""
given a ct scan file retrun all coordinates with the sizes
"""
def get_coordinates(dataframe):
    coordinates = dataframe[['worldX', 'worldY', 'worldZ']].values
    sizes = dataframe[['Size']].values.tolist()
    sizes = [[float(y) for y in x[0][1:-1].split(',')] for x in sizes]
    sentence = dataframe[['Sentence']].values.tolist()
    sentence = [x[0] for x in sentence]
    idxs = dataframe.index.tolist()
    types = dataframe['Type'].tolist()
    types = [[(int(y.replace('(','').replace(')','').split(', ')[0]), int(y.replace('(','').replace(')','').split(', ')[1])) for y in x[1:-1].split('), (')] for x in types]
    return coordinates, sizes, sentence, idxs, types
    
    
# frunction from https://d2l.ai/chapter_computer-vision/bounding-box.html slighly adapted
def box_center_to_corner(cx, cy, sizes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    #cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if len(sizes) == 1:
        w, h = sizes[0], sizes[0]
    if len(sizes) == 2:
        w, h = sizes[0], sizes[1]
    if len(sizes) == 3:
        w, h = sizes[0], sizes[1] # not sure this is the right choice, depends on axis of the box
        
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    box = (x1, y1, x2, y2) #tf.stack((x1, y1, x2, y2), axis=-1)
    return box
    
    
"""
load a ct scan
"""
def load_file(file, path=None):
    if path:
        image, header = read_dicom(path+file+'/dicom.zip')
    else:
        image, header = read_dicom('../Data/CT_scans/'+file+'/dicom.zip') #TODO: change
    voxel_spacing = header["spacing"]
    
    return image, header, voxel_spacing
    
    
"""
save a view of a lesion. The filename is the serie id plus an extention for the view 
('.1' : top view, '.2': front view, '.3': side view)
"""
def save_slice(image_slice, series, view, slice_nr, path_to_file, header):
    slice_information = dict()
    id_extension = '.' + str(slice_nr)
    if view == 'top view':
        id_extension += '.1'
    elif view == 'front view':
        id_extension += '.2'
    elif view == 'side view':
        id_extension += '.3'
        
    filename = series + id_extension
    slice_information['filename'] = filename
    slice_information['path_to_file'] = path_to_file
        
    write_image(path_to_file+filename+'.mha', image_slice, header=header, strip_metadata=False)
    
    return filename
    
    
def get_slices(image, file, coordinates, sizes):
    top_view_lesion_coordinates = []
    top_view_bboxes = []
    
    front_view_lesion_coordinates = []
    front_view_bboxes = []
    front_view_images = []

    side_view_lesion_coordinates = []
    side_view_bboxes = []
    side_view_images = []
    
    for c, size in zip(coordinates, sizes):
        converted_center = convert_world_to_ima(c, file)
        total_slice = ct_scan_conversion_df.loc[[file]]['total_slice_nr'][0]
        
        # Top view
        box = box_center_to_corner(c[0], c[1], size)
        converted = convert_world_to_ima(np.array([box[0], box[1], c[2]]), file)
        converted_lower = convert_world_to_ima(np.array([box[2], box[3], c[2]]), file)
        width, height = abs(converted_lower[1]-converted[1]),  abs(converted_lower[0]-converted[0])
            
        top_view_lesion_coordinates.append((converted_center[0], converted_center[1]))
        top_view_bboxes.append([(converted[0], converted[1]), (converted_lower[0], converted_lower[1])]) #Not sure about the order
        
        
    top_view = image[:,:,int(total_slice-converted_center[2])-1:int(total_slice-converted_center[2])+2]
    
    top_view_dict = {'center coordinates': top_view_lesion_coordinates, 'bboxes': top_view_bboxes}
    
    return top_view, top_view_dict
    
    
def get_slices_of_interest(file):
    image, header, voxel_spacing = load_file(file, path=input_image_folder)
    coordinate_df = id_information[id_information['SeriesPath'] == file].groupby('Slice IMA')
    data_information = []
    for ima, group in coordinate_df:
        coordinates, sizes, sentence, idxs, types = get_coordinates(group)
        series = group['seriesuid'].tolist()[0]
        counts = Counter(sentence)
        
        i = 0
        for sent, number in counts.items():
            coord = coordinates[i:i+number]
            s = sizes[i:i+number]
            top_view, top_view_dict = get_slices(image, file, coord, s)
            filename = save_slice(top_view, series, 'top view', ima, output_image_data_path, header)
            top_view_dict['filename'] = filename
            top_view_dict['sentence'] = sent
            top_view_dict['targets'] = types[0]
            top_view_dict['serie'] = series
            top_view_dict['ima'] = ima
            top_view_dict['path'] = output_image_data_path
            top_view_dict['number of boxes'] = len(top_view_dict['bboxes'])
            data_information.append(top_view_dict)
            i = i+number
            
    return data_information

    
def main():
    slice_data = []
    #print(ct_scan_conversion_df.info())
    for p in ct_scan_conversion_df['SeriesPath.1']:
        slice_data_dicts = get_slices_of_interest(p)
        slice_data.extend(slice_data_dicts)
        print(p)
    #slice_data_dicts = get_slices_of_interest('2/st002/se011')
    #print(slice_data_dicts)
    slice_data.to_csv(output_data_path+'slice_data.csv')
    
    
if __name__ == "__main__":
    print("Entering Python script")
    main()
