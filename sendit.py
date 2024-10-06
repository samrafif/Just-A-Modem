from PIL import Image
import argparse

import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="The input filename")
args = parser.parse_args()

if __name__ == "__main__":
    file_in = args.filename
    img = Image.open(file_in)

    numpy_array = np.array(img)
    original_shape = numpy_array.shape
    print(numpy_array.reshape(-1).shape, ",".join([str(a) for a in original_shape]))