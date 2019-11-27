import numpy as np
import cv2
import sys
import visdom
import pandas as pd
from PIL import Image
class DataSelector():
    def __init__(self,motion_path,motion_output_name):#,pose_file_path,motion_output_name):
        self.motion_path = motion_path
        #self.pose_file_path = pose_file_path
        self.motion_output_name = motion_output_name
    def select_process(self):
        # sequence prepare
        pose_data = np.loadtxt(self.motion_path)


        ## begin output
        count_for_name = 0
        result_pose_data = []
        for i_im in range(0,pose_data.shape[0]-1):

            pose_curr_m = np.matrix(np.eye(4))
            pose_next_m = np.matrix(np.eye(4))
            pose_curr = pose_data[i_im,:]
            pose_next = pose_data[i_im+1,:]
            pose_curr_m[0:3,0:4] = pose_curr.reshape(3,4)
            pose_next_m[0:3,0:4] = pose_next.reshape(3,4)
            motion = pose_curr_m.I*pose_next_m
            result_pose_data=np.append(result_pose_data,motion[0:3,0:4].reshape(1,12))
            print(i_im)
            count_for_name+=1
        result_pose_data = result_pose_data.reshape(count_for_name,12)
        np.savetxt('motion_'+self.motion_output_name+'.txt',result_pose_data)



if __name__ == '__main__':
    motion_path = sys.argv[1]
    motion_output_name = sys.argv[2]
    data_selector = DataSelector(motion_path,motion_output_name)
    data_selector.select_process()




