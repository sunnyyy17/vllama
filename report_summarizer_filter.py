import json
from rouge_score import rouge_scorer

'''
def dictionary_after_key(dictionary, key):
    keys = list(dictionary.keys())
    try:
        index = keys.index(key)
        # Get the part of the dictionary after the specified key
        return {k: dictionary[k] for k in keys[index+1:]}
    except ValueError:
        # Key not found in the dictionary
        return {}
'''

txt_path = '/scratch/slurm-user3/changsun/data/rectal_MRI_label/filtered_rectal_mri_report.json'
with open(txt_path, 'r') as file:
    rectal_mri_report = json.load(file)

formatted_report = {}

for key, value in rectal_mri_report.items():
    value_list = value.split(" ")
    if 'Distance' == value.split(" ")[0]:
        formatted_report[key] = value
    
    elif 'The lowest tumor' == " ".join(value_list[:3]):
        formatted_report[key] = value

    elif 'The distance' == " ".join(value_list[:2]):
        formatted_report[key] = value

print('length of formatted: ', len(list(formatted_report.keys())))
with open('formatted_rectal_mri_report.json', 'w') as json_writer:
    json.dump(formatted_report, json_writer, indent=4)

'''
with open('sum.out', 'r') as file:
    lines = file.read().splitlines()

candidates = [line for line in lines if "Candidate:" in line.split(" ")]
ground_truth = [line for line in lines if "Report:" in line.split(" ")]
print(len(candidates))
print(len(ground_truth))

search_key = '10784600'
rectal_mri_report = dictionary_after_key(rectal_mri_report, search_key)

keys_list = list(rectal_mri_report.keys())
keys_list = keys_list[:len(candidates)]

filter_mri_report = {}
for report in zip(keys_list, candidates, ground_truth):
    filt_report = report[1].split("Q:")[0]
    filt_report = filt_report.split("A: ")
    if len(filt_report) > 1:
        filt_report = filt_report[1]
    else:
        filt_report = filt_report[0]
    
    if "Candidate:  : " in filt_report:
        filt_report = filt_report[14:]
        
    gt_report = report[2]

    print("filter: ", filt_report)
    print("gt: ", gt_report)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    rouge_l_score = scorer.score(gt_report, filt_report)['rougeL'].fmeasure
    print("Rouge-L: ", rouge_l_score)
    if rouge_l_score > 0.2:
        filter_mri_report[report[0]] = filt_report

print("Number of Properly Summarized Reports", len(list(filter_mri_report.keys())))

with open('new_rectal_mri_report_fafter1100.json', 'w') as jfile:
    json.dump(filter_mri_report, jfile, indent=4)
'''
print("COMPLETE")
