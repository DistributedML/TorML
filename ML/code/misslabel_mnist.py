import numpy as np
import sys
import os

''' Mislabels preprocessed mnist data
Takes input
    - MNIST dataset. eg 4
    - misslabel : eg 9
 and saves updated dataset as "bad_mnist4_9.npy"
'''

def main(argv):
    print(argv)
    dataset = "mnist"
    dataset +=  argv[0]
    print(dataset)
    data = np.load(os.path.join('../ML', "data/mnist/", dataset + '.npy'))
    data[:, -1] += int(argv[1]) - int(argv[0])
    save_file = "mnist_bad_" + argv[0] + argv[1]
    np.save(save_file, data)




if __name__ == "__main__":
    main(sys.argv[1:])
