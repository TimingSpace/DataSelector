import numpy as np
import cv2
import sys
import visdom
import pandas as pd
from PIL import Image
class DataSelector():
    def __init__(self,image_list_path,pose_file_path,seq_len=40):
        self.seq_len = seq_len
        self.image_list_path = image_list_path
        self.pose_file_path = pose_file_path
    def select_process(self):
        # sequence prepare
        seq_len = self.seq_len
        pose_data = np.loadtxt(self.pose_file_path)
        data_length = pose_data.shape[0]
        image_paths =pd.read_csv(self.image_list_path)
        posi_data = pose_data[:,3:12:4]

        # measure the angle of each sequence
        selected_pos=[]
        for i in range(0,data_length-seq_len):
            begin_posi  = posi_data[i,:]
            middle_posi = posi_data[i+seq_len//2,:]
            end_posi    = posi_data[i+seq_len,:]
            vector_1 = begin_posi - middle_posi
            vector_2 = end_posi - middle_posi
            inner_dot =  np.dot(vector_1,vector_2)
            cos_angle = inner_dot/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))
            angle = np.arccos(cos_angle)
            angle = 180*angle/np.pi
            if np.abs(angle-90)<40:
                selected_pos.append(i)
        final_end_pos =[]
        final_length = []
        length = 1
        for i_pos in range(1,len(selected_pos)):
            if selected_pos[i_pos] - selected_pos[i_pos-1] != 1 or i_pos == len(selected_pos)-1:
                if i_pos != len(selected_pos)-1:
                    final_end_pos.append(selected_pos[i_pos-1])
                    final_length.append(length)
                else:
                    final_end_pos.append(selected_pos[i_pos])
                    final_length.append(length+1)
                length = 1
            else:
                length +=1
        final_middle = np.array(final_end_pos) - np.array(final_length)//2
        ## begin output
        count_for_name = 0
        result_pose_data = []
        for i_mid in final_middle:
            for i_im in range(i_mid-seq_len//2,i_mid+seq_len//2-1):
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
                save_image_name = 'result_image/kitti_turn_'+str(count_for_name).zfill(6)+'.png'
                cv2.imwrite(save_image_name,image)

                pose_curr_m = np.matrix(np.eye(4))
                pose_next_m = np.matrix(np.eye(4))
                pose_curr = pose_data[i_im,:]
                pose_next = pose_data[i_im+1,:]
                pose_curr_m[0:3,0:4] = pose_curr.reshape(3,4)
                pose_next_m[0:3,0:4] = pose_next.reshape(3,4)
                motion = pose_curr_m.I*pose_next_m
                result_pose_data=np.append(result_pose_data,motion[0:3,0:4].reshape(1,12))
                count_for_name = count_for_name +1
                print(count_for_name)
        result_pose_data = result_pose_data.reshape(count_for_name,12)
        np.savetxt('turn_motion_00.txt',result_pose_data)
    def select(self):
        # sequence prepare
        seq_len = self.seq_len
        pose_data = np.loadtxt(self.pose_file_path)
        data_length = pose_data.shape[0]
        image_paths =pd.read_csv(self.image_list_path)
        posi_data = pose_data[:,3:12:4]

        # measure the angle of each sequence
        selected_pos=[]
        for i in range(0,data_length-seq_len):
            begin_posi  = posi_data[i,:]
            middle_posi = posi_data[i+seq_len//2,:]
            end_posi    = posi_data[i+seq_len,:]
            vector_1 = begin_posi - middle_posi
            vector_2 = end_posi - middle_posi
            inner_dot =  np.dot(vector_1,vector_2)
            cos_angle = inner_dot/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))
            angle = np.arccos(cos_angle)
            angle = 180*angle/np.pi
            if np.abs(angle-90)<40:
                selected_pos.append(i)
        final_end_pos =[]
        final_length = []
        length = 1
        for i_pos in range(1,len(selected_pos)):
            if selected_pos[i_pos] - selected_pos[i_pos-1] != 1 or i_pos == len(selected_pos)-1:
                if i_pos != len(selected_pos)-1:
                    final_end_pos.append(selected_pos[i_pos-1])
                    final_length.append(length)
                else:
                    final_end_pos.append(selected_pos[i_pos])
                    final_length.append(length+1)
                length = 1
            else:
                length +=1
        final_middle = np.array(final_end_pos) - np.array(final_length)//2
        vis = visdom.Visdom()
        data = [{
            'x':pose_data[:,3].tolist(),
            'y':pose_data[:,11].tolist(),
            'mode':"lines",
            'name':'path',
            'type':'line',
            },{
            'x': pose_data[final_middle,3].tolist(),
            'y': pose_data[final_middle,11].tolist(),
            'type': 'scatter',
            'mode': 'markers',
            'name': 'turn point',
            }]

        win = 'mytestwin'
        env = 'main'

        layout= {
            'title':"Test Plot",
            'xaxis':{'title':'x1'},
            'yaxis':{'title':'x2'}
            }
        opts = {}

        vis._send({'data': data, 'win': win, 'eid': env, 'layout': layout, 'opts': opts})
        ## begin output
        for i_mid in final_middle:
            image_paths.ix[i_mid-seq_len//2:i_mid+seq_len/2-1,0].to_csv('result/image_00_sublist_'+str(i_mid)+'.txt')
            np.savetxt('result/motion_sub_'+str(i_mid)+'.txt',pose_data[i_mid-seq_len//2:i_mid+seq_len//2,:])

## flow cluster







if __name__ == '__main__':
    image_list_path = sys.argv[1]
    pose_file_path = sys.argv[2]
    data_selector = DataSelector(image_list_path,pose_file_path)
    data_selector.select_process()




