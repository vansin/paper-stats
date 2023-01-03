import glob

import pandas as pd


def main():
    tagged = dict()

    # collect pose/mmpose related papers
    data = pd.read_csv('tags/mmpose 顶会论文引用情况 - MMPose.csv')
    data = data[(data['relevant'] == 1) | (data['mmpose'] == 1)]

    for i in range(len(data)):
        item = data.iloc[i]
        paper = {
            'title': item['title'],
            'area_tag': 'pose estimation' if item['relevant'] else None,
            'codebase_tag': 'mmpose' if item['mmpose'] else None,
        }

        tagged[paper['title']] = paper

    # collect action/mmaction2 related papers
    data = pd.read_csv('tags/mmaction2 顶会论文引用情况 - MMAction2.csv')
    data = data[(data['relevant'] == 1) | (data['mmaction2'] == 1)]

    for i in range(len(data)):
        item = data.iloc[i]
        paper = {
            'title': item['title'],
            'area_tag': 'video understanding' if item['relevant'] else None,
            'codebase_tag': 'mmaction2' if item['mmaction2'] else None,
        }

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
