import json
import os

import pandas as pd

groups = os.listdir('out')

papers = []

keyword_groups = {
    'mmclassification': ['mmclassification', 'mmcls'],
    'PaddleClas': ['PaddleClas'],

    'mmselfsup': ['mmselfsup'],
    'PLSC': ['PLSC'],

    'mmdetection': ['mmdetection', 'mmdet', 'mmrotate', 'mmyolo'],
    'detectron2/detr': ['detecron2', 'detr'],
    'PaddleDetection': ['PaddleDetection'],
    
    'MMDetection3D': ['MMDetection3D', 'mmdet3d'],
    'Paddle3D': ['Paddle3D'],

    'MMSegmentation': ['MMSegmentation', 'mmseg'],
    'PaddleSeg': ['PaddleSeg'],

    'MMAction': ['MMAction'],
    'PaddleVideo': ['PaddleVideo'],

    'mmpose': ['mmpose'],

    'MMEditing': ['MMEditing', 'mmedit','mmgeneration'],
    'PaddleGAN': ['PaddleGAN'],
    
    'mmocr': ['mmocr'],
    'PaddleOCR': ['PaddleOCR'],

    'mmflow': ['mmflow'],
    'mmrazor': ['mmrazor'],

    'mmcv': ['mmcv'],
    'mmengine': ['mmengine'],
    

    'openmmlab': ['openmmlab', 'open mmlab', 'open-mmlab'],
    'mediapipe': ['mediapipe'],
    'paddle': ['paddle'],
    'mindspore': ['mindspore'],
}


for group in groups:
    with open(f'out/{group}') as f:
        data = json.load(f)
        papers.extend(data)

# print(papers)


def _valid(paper):

    if tag in paper:
        return True
    else:
        return False



# excel_list = dict()

# for tag, keywords in keyword_groups.items():

#     paper_list = list(filter(_valid, papers))
#     excel_list[tag] = paper_list
    
#     df = pd.DataFrame.from_dict(paper_list)
#     df.to_csv(f'/out_excel{tag}.csv', index=True, header=True)


df = pd.DataFrame.from_dict(papers)
df.to_csv(f'stats.csv', index=True, header=True)




# srun -p mm_model -n 1024 -c 1 --ntasks-per-node 36 python rank.py

print(len(groups))

