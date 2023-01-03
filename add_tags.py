import glob

import pandas as pd


def main():
    tagged = dict()

    # collect pose/mmpose related papers
    data = pd.read_csv('stats/mmpose_stats.csv')
    data = data[(data['relevant'] == 1) | (data['mmpose'] == 1) |
                (data['alphapose(code)'] == 1) | (data['openpose(code)'] == 1)
                | (data['detectron2/detr'] == 1) | (data['hrnet (code)'] == 1)
                | (data['videopose3d (code)'] == 1)]

    for i in range(len(data)):
        item = data.iloc[i]
        paper = {
            'title': item['title'],
            'area_tag': 'pose estimation' if item['relevant'] == 1 else None,
            'codebase_tag': None,
        }

        codebases = []
        if item['mmpose'] == 1:
            codebases.append('mmpose')
        if item['alphapose(code)'] == 1:
            codebases.append('alphapose')
        if item['openpose(code)'] == 1:
            codebases.append('openpose')
        if item['detectron2/detr'] == 1:
            codebases.append('detectron2/detr')
        if item['hrnet (code)'] == 1:
            codebases.append('hrnet')
        if item['videopose3d (code)'] == 1:
            codebases.append('videopose3d')
        if codebases:
            paper['codebase_tag'] = '|'.join(codebases)

        tagged[paper['title']] = paper

    # collect action/mmaction2 related papers
    data = pd.read_csv('stats/mmaction2_stats.csv')
    data = data[(data['relevant'] == 1) | (data['mmaction2'] == 1) |
                (data['slowfast'] == 1)]

    for i in range(len(data)):
        item = data.iloc[i]
        paper = {
            'title': item['title'],
            'area_tag':
            'video understanding' if item['relevant'] == 1 else None,
            'codebase_tag': None,
        }

        codebases = []
        if item['mmaction2'] == 1:
            codebases.append('mmaction2')
        if item['slowfast'] == 1:
            codebases.append('slowfast')
        if codebases:
            paper['codebase_tag'] = '|'.join(codebases)

        if paper['title'] in tagged:
            print(f'duplicated paper: {paper["title"]}')
        tagged[paper['title']] = paper

    print(f'Collect tag information from {len(tagged)} papers')

    # add tags to paper index
    for fn in glob.glob('index/*.csv'):
        print(f'Add tags to {fn}')
        fn_out = fn.replace('index', 'index_with_tags')
        data = pd.read_csv(fn)
        area_tags = data['related areas'].to_list()
        codebase_tags = data['codebases'].to_list()

        for i in range(len(data)):
            title = data.iloc[i]['title']
            if title in tagged:
                paper = tagged[title]
                if paper['area_tag']:
                    area_tags[i] = paper['area_tag']
                if paper['codebase_tag']:
                    codebase_tags[i] = paper['codebase_tag']

        data['related areas'] = area_tags
        data['codebases'] = codebase_tags

        data.to_csv(fn_out)


if __name__ == '__main__':
    main()
