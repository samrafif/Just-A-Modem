from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="The input filename")
parser.add_argument("-o", "--output", help="Output filename [OPTIONAL]", default="")
args = parser.parse_args()

if __name__ == "__main__":
    file_in = args.filename
    img = Image.open(file_in)

    # TODO: STUPHED
    file_out = f"{file_in.split('.')[0]}.bmp"
    img.save(file_out)