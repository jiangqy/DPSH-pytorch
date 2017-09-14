import DPSH_CIFAR_10 as dpsh
import pickle
from datetime import datetime

def DPSH_CIFAR_10_demo():
    lamda = 10
    param = {}
    param['lambda'] = lamda

    gpu_ind = 7
    bits = [12, 24, 32, 48]
    for bit in bits:
        filename = 'log/DPSH_' + str(bit) + 'bits_CIFAR-10' + '_' + datetime.now().strftime("%y-%m-%d-%H-%M-%S") + '.pkl'
        param['filename'] = filename
        print('---------------------------------------')
        print('[#bit: %3d]' % (bit))
        result = dpsh.DPSH_algo(bit, param, gpu_ind)
        print('[MAP: %3.5f]' % (result['map']))
        print('---------------------------------------')
        fp = open(result['filename'], 'wb')
        pickle.dump(result, fp)
        fp.close()

if __name__=="__main__":
    DPSH_CIFAR_10_demo()

