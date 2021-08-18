import cv2
import os
import pandas as pd
import pydicom

from collections import defaultdict
from tqdm import tqdm
from utils import load_dicom, create_directory


DATADIR = '../data/ricord/dicoms/'

info_dict = defaultdict(list)
for root,dirs,files in tqdm(os.walk(DATADIR)):
    for fi in files:
        if 'dcm' in fi:
            fp = os.path.join(root, fi)
            dcm = pydicom.dcmread(fp, stop_before_pixels=True)
            info_dict['PatientID'].append(dcm.PatientID)
            info_dict['StudyInstanceUID'].append(dcm.StudyInstanceUID)
            info_dict['SeriesInstanceUID'].append(dcm.SeriesInstanceUID)
            info_dict['SOPInstanceUID'].append(dcm.SOPInstanceUID)
            info_dict['ViewPosition'].append(dcm.ViewPosition)
            info_dict['filename'].append(fp)


df = pd.DataFrame(info_dict)
df['prefix'] = df.filename.apply(lambda x: '_'.join(x.replace(DATADIR, '').replace('MIDRC-RICORD-1C-', '').split('/')[:-1]))

PNGSDIR = '../data/ricord/pngs'
create_directory(PNGSDIR)

for rownum, row in info_dict.iterrows():
    dcm = load_dicom(row.filename, fake_rgb=False)
    new_filename = row.filename.replace(DATADIR, '').replace('MIDRC-RICORD-1C-', '')
    new_filename = new_filename.split('/')

