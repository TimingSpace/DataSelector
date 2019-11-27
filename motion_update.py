import sys
import numpy as np

def main():
    file_name = sys.argv[1]
    data = np.loadtxt(file_name)
    number = data.shape[0]
    dimension = data.shape[1]
    data_new = np.zeros((number-1,dimension*2))
    for i in range(number-1):
        data_new[i,0:dimension] = data[i]
        data_new[i,dimension:2*dimension] = data[i+1]
    np.savetxt(sys.argv[2],data_new)

if __name__ == '__main__':
    main()
