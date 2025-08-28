from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import os
import json
import argparse

def score(cap_list):
    '''
    input: list[[pred1, gt1], [pred2, gt2], ...]
    '''
    ref = dict()
    hypo = dict()
    if len(cap_list) == 0:
        return
    for i, pair in enumerate(cap_list):
        ref[i] = [pair[1]]
        hypo[i] = [pair[0]]
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        # (Meteor(),"METEOR"),
        (Rouge(),"ROUGE_L"),
        (Cider(), "Cider"),
    ]
    final_scores = {}
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score)==list:
                for m,s in zip(method,score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        except:
            print(f'error in {method}')

    print(final_scores)
    return final_scores
    
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--step', type=str, default='')
    
    step = args.parse_args().step
    # 指定文件夹路径
    folder_path = f'/data/liuxy/b2d/llava_carla/checkpoints/0.5B_changelane_road_QA_CoT/output/checkpoint-{step}'

    cap_list = []
    # i=0
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)
            
            # 打开并读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            answer = data['answer']
            predicted_answer = data['predicted_answer']
            cap_list.append([predicted_answer, answer])
            # i+=1
            # if i>10:
            #     break

    
    score_dict = score(cap_list)
    with open(os.path.join(folder_path, "language_metrics.txt"), "w") as f:
        f.write(str(score_dict))
    
    