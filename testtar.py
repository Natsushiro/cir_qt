import tarfile
import cv2
import numpy as np
from PIL import Image
import io
import os
#import GroundingDINO.groundingdino.datasets.transforms as T
#
#def load_image(image_data):
#    # load image
#    image_pil = Image.open(io.BytesIO(image_data))  # load image
#
#    transform = T.Compose(
#        [
#            T.RandomResize([800], max_size=1333),
#            T.ToTensor(),
#            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#        ]
#    )
#    image, _ = transform(image_pil, None)  # 3, h, w
#    return image_pil, image

#with tarfile.open("../data/conceptualcaptions/cc3m_train/00000.tar") as tar:
#    for member in tar.getmembers():
#        if member.name.endswith(".jpg"):
#            image_data = tar.extractfile(member).read()
#            image_pil, image = load_image(image_data)
#
#            image_pil.save("./testtar/%s.mask.jpg" % member.name.split(".")[0])
#
#            nparr = np.frombuffer(image_data, np.uint8)
#            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#            print(image_pil.size)
#            break
new_tar = "00000new.tar"
ex = "../Grounded-Segment-Anything/00000.tar"
with tarfile.open(new_tar, "w") as new_tar:
    with tarfile.open(ex) as tar:
        folder = "../data/testtar"
        for member in tar.getmembers():
            new_tar.addfile(member, tar.extractfile(member))
        for root, _, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder)
                new_tar.add(file_path, arcname=arcname)