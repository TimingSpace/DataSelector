import numpy as np
import sys
import visdom
import pandas as pd
class DataSelector():
    def __init__(self,image_list_path,pose_file_path,seq_len=40):
        self.seq_len = seq_len
        self.image_list_path = image_list_path
        self.pose_file_path = pose_file_path
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
            middle_posi = posi_data[i+seq_len/2,:]
            end_posi    = posi_data[i+seq_len,:]
            vector_1 = begin_posi - middle_posi
            vector_2 = end_posi - middle_posi
            inner_dot =  np.dot(vector_1,vector_2)
            cos_angle = inner_dot/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))
            angle = np.arccos(cos_angle)
            angle = 180*angle/np.pi
            if np.abs(angle-90)<40:
                print angle
                selected_pos.append(i)
        print selected_pos
        print len(selected_pos)
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
        print final_end_pos,final_length
        final_middle = np.array(final_end_pos) - np.array(final_length)/2
        print final_middle
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
            image_paths.ix[i_mid-seq_len/2:i_mid+seq_len/2-1,0].to_csv('result/image_00_sublist_'+str(i_mid)+'.txt')
            np.savetxt('result/motion_sub_'+str(i_mid)+'.txt',pose_data[i_mid-seq_len/2:i_mid+seq_len/2,:])

## flow cluster







if __name__ == '__main__':
    image_list_path = sys.argv[1]
    pose_file_path = sys.argv[2]
    data_selector = DataSelector(image_list_path,pose_file_path)
    data_selector.select()




