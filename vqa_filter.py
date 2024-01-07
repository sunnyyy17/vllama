import json
import re

txt_path = '/scratch/slurm-user3/changsun/vllama/new_rectal_mri_report_1100.json'

with open(txt_path, 'r') as file:
    new_rectal_mri_report = json.load(file)

with open('vqat.out', 'r') as file:
    lines = file.read().splitlines()

candidates = [line for line in lines if "Candidate:" in line.split(" ")]
ground_truth = [line for line in lines if "Report:" in line.split(" ")]
print(len(candidates))
print(len(ground_truth))

keys_list = list(new_rectal_mri_report.keys())
keys_list = keys_list[:len(candidates)]

filter_mri_vqa = {}
report_cnt = 0
pair_cnt = 0
for vqa in zip(keys_list, candidates, ground_truth):
    vqa_candidate = vqa[1]

    qa_pairs = re.findall(r'Question: (.*?)Answer: (.*?)(?=Question:|$)', vqa_candidate)
    
    if len(qa_pairs) == 0:
        continue
    filter_mri_vqa[vqa[0]] = []
    report_cnt += 1
    print("==========================================")
    print(vqa[2])
    for pair in qa_pairs:
        question, answer = pair 
        print('Question: ', question)
        print('Answer: ', answer)
        single_qa = {'Question': question, 'Answer': answer}
        filter_mri_vqa[vqa[0]].append(single_qa)
        pair_cnt += 1
    print("==========================================")

print("COMPLETE",'vqa pairs:', pair_cnt, 'report_cnt', report_cnt)
with open('new_rectal_mri_vqa.json', 'w') as jfile:
    json.dump(filter_mri_vqa, jfile, indent=4)
