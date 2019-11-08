import whatimage
import pyheif
import os
import io
from PIL import Image


def decode_image(f):
    fmt = whatimage.identify_image(f)
    i = pyheif.read_heif(f)

    # Extract metadata etc
    # for metadata in i.metadata or []:
    #     if metadata['type'] == 'Exif':

    # Convert to other file format like jpeg
    s = io.BytesIO()
    pi = Image.frombytes(
        mode=i.mode, size=i.size, data=i.data)

    pi.save(s, format="jpeg")


if __name__ == '__main__':
    for r, d, f in os.walk('../../../data/pawn/'):
        if '.heic' in f:
            decode_image(f)
