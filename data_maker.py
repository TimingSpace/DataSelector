import numpy as np
import cv2
import sys
import visdom
import pandas as pd
from PIL import Image
class DataSelector():
    def __init__(self,image_list_path,motion_output_name):#,pose_file_path,motion_output_name):
        self.image_list_path = image_list_path
        #self.pose_file_path = pose_file_path
        self.output_name = motion_output_name
    def select_process(self):
        # sequence prepare
        #pose_data = np.loadtxt(self.pose_file_path)

        image_paths =pd.read_csv(self.image_list_path)

        ## begin output
        count_for_name = 0
        result_pose_data = []
        for i_im in range(0,len(image_paths)-1):
            image_name_curr = image_paths.ix[i_im,0]
            image_name_next = image_paths.ix[i_im+1,0]
            image_curr = cv2.imread(image_name_curr)
            image_next = cv2.imread(image_name_next)
            image_curr_gray = Image.fromarray(image_curr)
            image_next_gray = Image.fromarray(image_next)
            image = np.zeros((image_curr.shape[0],image_curr.shape[1],4),dtype=np.uint8)
            image[:,:,0] = image_curr_gray.convert('L')
            image[:,:,1] = image_next_gray.convert('L')
            image[:,:,2] = image_next_gray.convert('L')
            image[:,:,3] = image_curr_gray.convert('L')
            save_image_name = self.output_name+'ntsd_flash'+str(count_for_name).zfill(5)+'.png'
            cv2.imwrite(save_image_name,image)
            count_for_name = count_for_name +1
            '''
            pose_curr_m = np.matrix(np.eye(4))
            pose_next_m = np.matrix(np.eye(4))
            pose_curr = pose_data[i_im,:]
            pose_next = pose_data[i_im+1,:]
            pose_curr_m[0:3,0:4] = pose_curr.reshape(3,4)
            pose_next_m[0:3,0:4] = pose_next.reshape(3,4)
            motion = pose_curr_m.I*pose_next_m
            result_pose_data=np.append(result_pose_data,motion[0:3,0:4].reshape(1,12))
            print(i_im)
        result_pose_data = result_pose_data.reshape(count_for_name,12)
        np.savetxt('motion_'+self.motion_output_name+'.txt',result_pose_data)
'''



if __name__ == '__main__':
    image_list_path = sys.argv[1]
    motion_output_name = sys.argv[2]
    data_selector = DataSelector(image_list_path,motion_output_name)
    data_selector.select_process()




