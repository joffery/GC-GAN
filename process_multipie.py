import os
import re

# 正则参数说明

regex = r"""(\d{3})_(\d{2}_\d{2})_(051|050|140)_(\d{2}).png"""
pattern = re.compile(regex)

save_dir = "D:\workspace\common_database\multipie_processed\multipie_nearfrontal_faces_original_withMore_lightness"
multipie_path = 'D:\workspace\common_database\CMU_Multi-PIE\Multi-Pie\data'

for dirpath, dirnames, filenames in os.walk(multipie_path):
    for filename in filenames:
        if pattern.match(filename):
            source = os.path.join(dirpath,filename)
            target = os.path.join(save_dir,filename)
            open(target,'wb+').write(open(source,'rb').read())

