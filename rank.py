import glob
import os.path as osp
import pickle
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Optional

import pandas as pd
import pdfplumber
import tqdm
import wget
from pdfminer.pdfparser import PDFSyntaxError

import multiprocessing

import logging


import os

# os.remove("log.txt")

cores = multiprocessing.cpu_count()
processes=2
print(cores)
@dataclass
class paper_info:
    title: str
    conference: str
    year: int
    authors: str
    abstract: Optional[str] = None
    pdf_path: Optional[str] = None
    pdf_url: Optional[str] = None
    code_url: Optional[str] = None


def load_paper_info(index_path: str = 'index'):
    """Load paper information from csv files which can be downloaded from
    https://aicarrier.feishu.cn/sheets/shtcnGhSBiEUVqnHQBPtshy6Tse and should
    be organized as:

    index
    ├── 顶会论文数据库-AAAI.csv
    ├── 顶会论文数据库-CVPR.csv
    ├── 顶会论文数据库-ECCV.csv
    ...
    """
    papers = []

    for fn in glob.glob(osp.join(index_path, '*.csv')):
        print(f'load paper index from {fn}')
        df = pd.read_csv(fn)

        for _, item in tqdm.tqdm(df.iterrows(), total=len(df) - 1):
            paper = paper_info(title=item['title'],
                               conference=item['conference'],
                               year=item['year'],
                               authors=item['authors'],
                               abstract=item['abstract'])

            if isinstance(item['pdf_url'], str):
                paper.pdf_url = item['pdf_url']

            if isinstance(item['code_url'], str):
                paper.code_url = item['code_url']

            paper.pdf_path = osp.join('data',
                                      f'{paper.conference}{paper.year}',
                                      f'{paper.title}.pdf')

            papers.append(paper)

    print(f'load {len(papers)} papers in total.')
    return papers


def _download(args):
    idx, total, paper = args
    url = paper.pdf_url
    path = paper.pdf_path
    try:
        print(f'{idx}/{total}')
        wget.download(url, path, bar=None)
        return None
    except:  # noqa
        return paper


def download_missing_pdf(papers):

    missing_list = [
        paper for paper in papers if not osp.isfile(paper.pdf_path)
    ]
    print(f'found {len(missing_list)} missing papers.')

    with Pool(processes=processes) as p:
        total = len(missing_list)
        tasks = [(i, total, paper) for i, paper in enumerate(missing_list)]
        failed_list = [r for r in p.map(_download, tasks) if r is not None]

    if failed_list:
        print(f'failed to download {len(failed_list)} papers.')
        with open('failed_list.pkl', 'wb') as f:
            pickle.dump(failed_list, f)


def search_kwgroups_in_pdf(pdf_path: str,
                           keyword_groups: dict[str, list[str]],
                           case_sensitive=False) -> list[int]:
    """Search a keyword groups in a pdf file. One keyword group is considered
    hit if at least one keyword in this group is found in the pdf.

    Args:
        pdf_path (str): path to the pdf file
        keyword_groups (dict[str, list[str]]): A list of keyword groups. Each
            group is a list of keywords
        case_sensitive (bool): Whether consider letter case
    
    Returns:
        dict[str, bool]: The indicators of each keypoint group.
    """

    if not case_sensitive:
        keyword_groups = {
            k: [kw.lower() for kw in group]
            for k, group in keyword_groups.items()
        }

    result = {k: False for k in keyword_groups.keys()}
    if osp.isfile(pdf_path):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for _, page in enumerate(pdf.pages, 1):
                    if all(result.values()):
                        break

                    text = page.extract_text()
                    if not case_sensitive:
                        text = text.lower()

                    for name, group in keyword_groups.items():
                        if result[name]:
                            continue
                        else:
                            for kw in group:
                                if kw in text:
                                    result[name] = True
                                    break
        except PDFSyntaxError:
            print(f'fail to parse: {pdf_path}')

    return result


def _search_in_pdf(args):
    idx, total, keyword_groups, pdf_path = args
    # print(f'{idx}/{total}')
    fs_log = open('log.txt', 'a+')
    fs_log.write(f'{idx}/{total}\n')
    fs_log.close()

    return search_kwgroups_in_pdf(pdf_path, keyword_groups)


def main():
    # load paper information
    papers = load_paper_info()
    # download_missing_pdf(papers)

    # search in title/abstract
    def _valid(paper):
        pos_kws = [
            'pose estimation',
            'pose regression',
            'keypoint',
            'landmark',
            'pose tracking',

            'camera pose',
        ]

        neg_kws = [
            'object pose estimation',
            '6d',
            '6 dof',
            'camera pose',
            'pose and shape',
            'shape and pose',
        ]

        text = paper.title.lower()
        # if isinstance(paper.abstract, str):
        #     text = text + ' ' + paper.abstract.lower()

        for kw in neg_kws:
            if kw in text:
                return False

        for kw in pos_kws:
            if kw in text:
                return True

        return False

    # papers = list(filter(_valid, papers))

    # search in PDF
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

    total = len(papers)
    print("processes:" ,processes)

    print('total totaltotaltotaltotaltotaltotaltotaltotaltotaltotaltotaltotaltotaltotaltotaltotal')
    tasks = [(i, total, keyword_groups, paper.pdf_path)
             for i, paper in enumerate(papers)]
    
    print("processes:" ,processes)
    import os
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    local_tasks = tasks[proc_id::ntasks]

    search_results = []

    for task in local_tasks:
        try:
            result = _search_in_pdf(task)
            search_results.append(result)
        except Exception as e:
            print(e)        

    # search_results = _search_in_pdf(local_tasks)
    # search_results = p.map(_search_in_pdf, tasks)

    # with Pool(processes=processes) as p:
        
    #     search_results = p.map(_search_in_pdf, tasks)

    with open(f'outp/mmpose_search_results_{proc_id}.pkl', 'wb') as f:
        pickle.dump(search_results, f)
    

    matched = []
    for paper, result in zip(papers, search_results):
        pos_keys = [k for k in result.keys() if k.startswith('_pose_pos')]
        neg_keys = [k for k in result.keys() if k.startswith('_pose_neg')]

        relevant = False
        for key in pos_keys:
            relevant |= result.pop(key)

        for key in neg_keys:
            relevant &= (~result.pop(key))

        result['relevant'] = relevant
        result = {k: int(v) for k, v in result.items()}
        if any(result.values()):
            matched.append((paper, result))

    for name in matched[0][1].keys():
        count = sum(result[name] for _, result in matched)
        print(name, count)

    # save to csv
    paper_dicts = []
    for paper, result in matched:
        d = paper.__dict__.copy()
        d.update(result)
        paper_dicts.append(d)
    
    import json
    json.dump(paper_dicts, open(f'out/mmpose_matched_{proc_id}.json', 'w'), indent=2)

    # df = pd.DataFrame.from_dict(paper_dicts)
    # df.to_csv(f'mmpose_stats_{proc_id}.csv', index=True, header=True)

    # srun -p mm_model -n 1024 -c 1 --ntasks-per-node 36 python rank.py

if __name__ == '__main__':

    main()
