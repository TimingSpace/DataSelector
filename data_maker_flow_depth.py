import numpy as np
import cv2
import sys
import visdom
import pandas as pd
from PIL import Image
class DataSelector():
    def __init__(self,flow_list_path,depth_list_path,pose_file_path,motion_output_name):
        self.flow_list_path = flow_list_path
        self.depth_list_path = depth_list_path
        self.pose_file_path = pose_file_path
        self.motion_output_name = motion_output_name
    def select_process(self):
        # sequence prepare
        pose_data = np.loadtxt(self.pose_file_path)
        data_length = pose_data.shape[0]
        image_paths =pd.read_csv(self.image_list_path)

        ## begin output
        count_for_name = 0
        result_pose_data = []
        for i_im in range(0,data_length-2):
            print(i_im)
            image_name_curr = image_paths.ix[i_im,0]
            image_name_next = image_paths.ix[i_im+1,0]
            image_name_next_2 = image_paths.ix[i_im+2,0]
            image_curr = cv2.imread(image_name_curr)
            image_next = cv2.imread(image_name_next)
            image_next_2 = cv2.imread(image_name_next_2)
            image_curr_gray = Image.fromarray(image_curr)
            image_next_gray = Image.fromarray(image_next)
            image_next_gray_2 = Image.fromarray(image_next_2)
            image = np.zeros((image_curr.shape[0],image_curr.shape[1],3),dtype=np.uint8)
            image[:,:,0] = image_curr_gray.convert('L')
            image[:,:,1] = image_next_gray.convert('L')
            image[:,:,2] = image_next_gray_2.convert('L')
            save_image_name = '/data/datasets/xiangwei/kitti_full_3/kitti_'+self.motion_output_name+'_'+str(count_for_name).zfill(6)+'.png'
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
            count_for_name = count_for_name +1
            print(i_im)
        result_pose_data = result_pose_data.reshape(count_for_name,12)
        np.savetxt('motion_'+self.motion_output_name+'.txt',result_pose_data)
        '''



if __name__ == '__main__':
    image_list_path = sys.argv[1]
    pose_file_path = sys.argv[2]
    motion_output_name = sys.argv[3]
    data_selector = DataSelector(image_list_path,pose_file_path,motion_output_name)
    data_selector.select_process()




