import os
import json
import shutil

src = 'data/BaileyLabels/BaileyLabelSet/'
dst = 'data/BaileyLabels/imagefolder/'

metadata = json.load(open('data/BaileyLabels/metadata.json'))

for key in metadata:
    y = metadata[key]['genus']
    os.makedirs(dst + y, exist_ok=True)
    shutil.copy(src + key, dst + y + '/' + key)