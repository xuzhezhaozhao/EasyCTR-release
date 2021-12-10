
import json
import sys
import numpy as np
import pandas as pd
import warnings

if len(sys.argv) != 6:
    print("Usage: <predictfile> <labelfile> <extra_id> <outputfile> <hive_media_type>")
    sys.exit(-1)

warnings.filterwarnings('ignore')

predictfile = sys.argv[1]
labelfile = sys.argv[2]
extra_id = float(sys.argv[3])
outputfile = sys.argv[4]
hive_media_type = int(float(sys.argv[5]))

if hive_media_type == 0:
    pncv_id = 411
else:
    pncv_id = 105

def parse_sample_line(line):
    D = {}
    line = line.strip()
    tokens = line.split(';')
    for token in tokens:
        subs = token.split('|')
        if len(subs) != 2:
            continue

        try:
            fid = float(subs[0])
            value = subs[1]
            D[fid] = value
        except:
            continue

    return D


preds = []
for line in open(predictfile):
    if line == '':
        break
    p = json.loads(line)['predictions'][0]
    preds.append(p)


labels = []
extras = []
pncvs = []
for line in open(labelfile):
    if line == '':
        break
    D = parse_sample_line(line)
    label = float(D[1])
    labels.append(label)
    extras.append(D.get(extra_id, '0'))
    pncvs.append(float(D.get(pncv_id, 0)))

ensembles = [(x + y) * 0.5 for (x, y) in zip(preds, pncvs)]

sum_labels = sum(labels)
sum_preds = sum(preds)
sum_pncvs = sum(pncvs)
sum_ensembles = sum(ensembles)
print('[diff-sum]')
print('labels = {:12.0f}'.format(sum_labels))
print('preds  = {:12.0f}, diff rate = {:.3f}'.format(sum_preds, 1.0*sum_preds/sum_labels))
print('pncvs  = {:12.0f}, diff rate = {:.3f}'.format(sum_pncvs, 1.0*sum_pncvs/sum_labels))
print('ensembles  = {:12.0f}, diff rate = {:.3f}'.format(sum_ensembles, 1.0*sum_ensembles/sum_labels))
print('\n')

assert len(preds) == len(labels)

preds_diff_0 = [max(x, 0.0) for (x, y) in zip(preds, labels) if y == 0]
pncv_diff_0 = [max(x, 0.0) for (x, y) in zip(pncvs, labels) if y == 0]
ensemble_diff_0 = [max(x, 0.0) for (x, y) in zip(ensembles, labels) if y == 0]

diffs_1 = [abs(x - y) / float(y) for (x, y) in zip(preds, labels) if y >= 1]
weighted_diffs_1 = [abs(x - y) for (x, y) in zip(preds, labels) if y >= 1]
labels_1 = [y for y in labels if y >= 1]
pncv_diffs_1 = [abs(x - y) / float(y) for (x, y) in zip(pncvs, labels) if y >= 1]
pncv_weighted_diffs_1 = [abs(x - y) for (x, y) in zip(pncvs, labels) if y >= 1]
ensemble_diffs_1 = [abs(x - y) / float(y) for (x, y) in zip(ensembles, labels) if y >= 1]
ensemble_weighted_diffs_1 = [abs(x - y) for (x, y) in zip(ensembles, labels) if y >= 1]

diffs_5 = [abs(x - y) / float(y) for (x, y) in zip(preds, labels) if y >= 5]
weighted_diffs_5 = [abs(x - y) for (x, y) in zip(preds, labels) if y >= 5]
labels_5 = [y for y in labels if y >= 5]
pncv_diffs_5 = [abs(x - y) / float(y) for (x, y) in zip(pncvs, labels) if y >= 5]
pncv_weighted_diffs_5 = [abs(x - y) for (x, y) in zip(pncvs, labels) if y >= 5]
ensemble_diffs_5 = [abs(x - y) / float(y) for (x, y) in zip(ensembles, labels) if y >= 5]
ensemble_weighted_diffs_5 = [abs(x - y) for (x, y) in zip(ensembles, labels) if y >= 5]

diffs_10 = [abs(x - y) / float(y) for (x, y) in zip(preds, labels) if y >= 10]
weighted_diffs_10 = [abs(x - y) for (x, y) in zip(preds, labels) if y >= 10]
labels_10 = [y for y in labels if y >= 10]
pncv_diffs_10 = [abs(x - y) / float(y) for (x, y) in zip(pncvs, labels) if y >= 10]
pncv_weighted_diffs_10 = [abs(x - y) for (x, y) in zip(pncvs, labels) if y >= 10]
ensemble_diffs_10 = [abs(x - y) / float(y) for (x, y) in zip(ensembles, labels) if y >= 10]
ensemble_weighted_diffs_10 = [abs(x - y) for (x, y) in zip(ensembles, labels) if y >= 10]

diffs_1_10 = [abs(x - y) / float(y) for (x, y) in zip(preds, labels) if y >= 1 and y < 10]
weighted_diffs_1_10 = [abs(x - y) for (x, y) in zip(preds, labels) if y >= 1 and y < 10]
labels_1_10 = [y for y in labels if y >= 1 and y < 10]
pncv_diffs_1_10 = [abs(x - y) / float(y) for (x, y) in zip(pncvs, labels) if y >= 1 and y < 10]
pncv_weighted_diffs_1_10 = [abs(x - y) for (x, y) in zip(pncvs, labels) if y >= 1 and y < 10]
ensemble_diffs_1_10 = [abs(x - y) / float(y) for (x, y) in zip(ensembles, labels) if y >= 1 and y < 10]
ensemble_weighted_diffs_1_10 = [abs(x - y) for (x, y) in zip(ensembles, labels) if y >= 1 and y < 10]

diffs_10_100 = [abs(x - y) / float(y) for (x, y) in zip(preds, labels) if y >= 10 and y < 100]
weighted_diffs_10_100 = [abs(x - y) for (x, y) in zip(preds, labels) if y >= 10 and y < 100]
labels_10_100 = [y for y in labels if y >= 10 and y < 100]
pncv_diffs_10_100 = [abs(x - y) / float(y) for (x, y) in zip(pncvs, labels) if y >= 10 and y < 100]
pncv_weighted_diffs_10_100 = [abs(x - y) for (x, y) in zip(pncvs, labels) if y >= 10 and y < 100]
ensemble_diffs_10_100 = [abs(x - y) / float(y) for (x, y) in zip(ensembles, labels) if y >= 10 and y < 100]
ensemble_weighted_diffs_10_100 = [abs(x - y) for (x, y) in zip(ensembles, labels) if y >= 10 and y < 100]

diffs_100_1000 = [abs(x - y) / float(y) for (x, y) in zip(preds, labels) if y >= 100 and y < 1000]
weighted_diffs_100_1000 = [abs(x - y) for (x, y) in zip(preds, labels) if y >= 100 and y < 1000]
labels_100_1000 = [y for y in labels if y >= 100 and y < 1000]
pncv_diffs_100_1000 = [abs(x - y) / float(y) for (x, y) in zip(pncvs, labels) if y >= 100 and y < 1000]
pncv_weighted_diffs_100_1000 = [abs(x - y) for (x, y) in zip(pncvs, labels) if y >= 100 and y < 1000]
ensemble_diffs_100_1000 = [abs(x - y) / float(y) for (x, y) in zip(ensembles, labels) if y >= 100 and y < 1000]
ensemble_weighted_diffs_100_1000 = [abs(x - y) for (x, y) in zip(ensembles, labels) if y >= 100 and y < 1000]

diffs_1000_2000 = [abs(x - y) / float(y) for (x, y) in zip(preds, labels) if y >= 1000 and y < 2000]
weighted_diffs_1000_2000 = [abs(x - y) for (x, y) in zip(preds, labels) if y >= 1000 and y < 2000]
labels_1000_2000 = [y for y in labels if y >= 1000 and y < 2000]
pncv_diffs_1000_2000 = [abs(x - y) / float(y) for (x, y) in zip(pncvs, labels) if y >= 1000 and y < 2000]
pncv_weighted_diffs_1000_2000 = [abs(x - y) for (x, y) in zip(pncvs, labels) if y >= 1000 and y < 2000]
ensemble_diffs_1000_2000 = [abs(x - y) / float(y) for (x, y) in zip(ensembles, labels) if y >= 1000 and y < 2000]
ensemble_weighted_diffs_1000_2000 = [abs(x - y) for (x, y) in zip(ensembles, labels) if y >= 1000 and y < 2000]

diffs_2000_5000 = [abs(x - y) / float(y) for (x, y) in zip(preds, labels) if y >= 2000 and y < 5000]
weighted_diffs_2000_5000 = [abs(x - y) for (x, y) in zip(preds, labels) if y >= 2000 and y < 5000]
labels_2000_5000 = [y for y in labels if y >= 2000 and y < 5000]
pncv_diffs_2000_5000 = [abs(x - y) / float(y) for (x, y) in zip(pncvs, labels) if y >= 2000 and y < 5000]
pncv_weighted_diffs_2000_5000 = [abs(x - y) for (x, y) in zip(pncvs, labels) if y >= 2000 and y < 5000]
ensemble_diffs_2000_5000 = [abs(x - y) / float(y) for (x, y) in zip(ensembles, labels) if y >= 2000 and y < 5000]
ensemble_weighted_diffs_2000_5000 = [abs(x - y) for (x, y) in zip(ensembles, labels) if y >= 2000 and y < 5000]



diffs_5000 = [abs(x - y) / float(y) for (x, y) in zip(preds, labels) if y >= 5000]
weighted_diffs_5000 = [abs(x - y) for (x, y) in zip(preds, labels) if y >= 5000]
labels_5000 = [y for y in labels if y >= 5000]
pncv_diffs_5000 = [abs(x - y) / float(y) for (x, y) in zip(pncvs, labels) if y >= 5000]
pncv_weighted_diffs_5000 = [abs(x - y) for (x, y) in zip(pncvs, labels) if y >= 5000]
ensemble_diffs_5000 = [abs(x - y) / float(y) for (x, y) in zip(ensembles, labels) if y >= 5000]
ensemble_weighted_diffs_5000 = [abs(x - y) for (x, y) in zip(ensembles, labels) if y >= 5000]

# log diff info
cnt_0_inf = len(labels)
cnt_0 = len([x for x in labels if x == 0])
cnt_1_10 = len(diffs_1_10)
cnt_10_100 = len(diffs_10_100)
cnt_100_1000 = len(diffs_100_1000)
cnt_1000_2000 = len(diffs_1000_2000)
cnt_2000_5000 = len(diffs_2000_5000)
cnt_5000_inf = len(diffs_5000)
print('[diff-sample-cnt]')
print('0:         {:12.0f}, prop = {:.0f}%'.format(cnt_0, 100.0 * cnt_0 / cnt_0_inf))
print('1-10:      {:12.0f}, prop = {:.0f}%'.format(cnt_1_10, 100.0 * cnt_1_10 / cnt_0_inf))
print('10-100:    {:12.0f}, prop = {:.0f}%'.format(cnt_10_100, 100.0 * cnt_10_100 / cnt_0_inf))
print('100-1000:  {:12.0f}, prop = {:.0f}%'.format(cnt_100_1000, 100.0 * cnt_100_1000 / cnt_0_inf))
print('1000-2000: {:12.0f}, prop = {:.0f}%'.format(cnt_1000_2000, 100.0 * cnt_1000_2000 / cnt_0_inf))
print('2000-5000: {:12.0f}, prop = {:.0f}%'.format(cnt_2000_5000, 100.0 * cnt_2000_5000 / cnt_0_inf))
print('5000-inf:  {:12.0f}, prop = {:.0f}%'.format(cnt_5000_inf, 100.0 * cnt_5000_inf / cnt_0_inf))
print('')

cv_cnt_0_inf = sum(labels)
cv_cnt_1_10 = sum(labels_1_10)
cv_cnt_10_100 = sum(labels_10_100)
cv_cnt_100_1000 = sum(labels_100_1000)
cv_cnt_1000_2000 = sum(labels_1000_2000)
cv_cnt_2000_5000 = sum(labels_2000_5000)
cv_cnt_5000_inf = sum(labels_5000)
print('[diff-cv-cnt]')
print('1-10:      {:12.0f}, prop = {:.0f}%'.format(cv_cnt_1_10, 100.0 * cv_cnt_1_10 / cv_cnt_0_inf))
print('10-100:    {:12.0f}, prop = {:.0f}%'.format(cv_cnt_10_100, 100.0 * cv_cnt_10_100 / cv_cnt_0_inf))
print('100-1000:  {:12.0f}, prop = {:.0f}%'.format(cv_cnt_100_1000, 100.0 * cv_cnt_100_1000 / cv_cnt_0_inf))
print('1000-2000: {:12.0f}, prop = {:.0f}%'.format(cv_cnt_1000_2000, 100.0 * cv_cnt_1000_2000 / cv_cnt_0_inf))
print('2000-5000: {:12.0f}, prop = {:.0f}%'.format(cv_cnt_2000_5000, 100.0 * cv_cnt_2000_5000 / cv_cnt_0_inf))
print('5000-inf:  {:12.0f}, prop = {:.0f}%'.format(cv_cnt_5000_inf, 100.0 * cv_cnt_5000_inf / cv_cnt_0_inf))
print('')

print('[diff-reflow-rate]')
print('1-inf:     {:.3f}, std = {:.3f}'.format(np.mean(diffs_1), np.std(diffs_1)))
print('5-inf:     {:.3f}, std = {:.3f}'.format(np.mean(diffs_5), np.std(diffs_5)))
print('10-inf:    {:.3f}, std = {:.3f}'.format(np.mean(diffs_10), np.std(diffs_10)))
print('0:         {:.3f}, std = {:.3f}'.format(np.mean(preds_diff_0), np.std(preds_diff_0)))
print('1-10:      {:.3f}, std = {:.3f}'.format(np.mean(diffs_1_10), np.std(diffs_1_10)))
print('10-100:    {:.3f}, std = {:.3f}'.format(np.mean(diffs_10_100), np.std(diffs_10_100)))
print('100-1000:  {:.3f}, std = {:.3f}'.format(np.mean(diffs_100_1000), np.std(diffs_100_1000)))
print('1000-2000: {:.3f}, std = {:.3f}'.format(np.mean(diffs_1000_2000), np.std(diffs_1000_2000)))
print('2000-5000: {:.3f}, std = {:.3f}'.format(np.mean(diffs_2000_5000), np.std(diffs_2000_5000)))
print('5000-inf:  {:.3f}, std = {:.3f}'.format(np.mean(diffs_5000), np.std(diffs_5000)))
print('')

print('[diff-reflow-weighted_rate]')
print('1-inf:     {:.3f}, std = {:.3f}'.format(sum(weighted_diffs_1)/(sum(labels_1)+0.001), np.std(weighted_diffs_1)))
print('5-inf:     {:.3f}, std = {:.3f}'.format(sum(weighted_diffs_5)/(sum(labels_5)+0.001), np.std(weighted_diffs_5)))
print('10-inf:    {:.3f}, std = {:.3f}'.format(sum(weighted_diffs_10)/(sum(labels_10)+0.001), np.std(weighted_diffs_10)))
print('1-10:      {:.3f}, std = {:.3f}'.format(sum(weighted_diffs_1_10)/(sum(labels_1_10)+0.001), np.std(weighted_diffs_1_10)))
print('10-100:    {:.3f}, std = {:.3f}'.format(sum(weighted_diffs_10_100)/(sum(labels_10_100)+0.001), np.std(weighted_diffs_10_100)))
print('100-1000:  {:.3f}, std = {:.3f}'.format(sum(weighted_diffs_100_1000)/(sum(labels_100_1000)+0.001), np.std(weighted_diffs_100_1000)))
print('1000-2000: {:.3f}, std = {:.3f}'.format(sum(weighted_diffs_1000_2000)/(sum(labels_1000_2000)+0.001), np.std(weighted_diffs_1000_2000)))
print('2000-5000: {:.3f}, std = {:.3f}'.format(sum(weighted_diffs_2000_5000)/(sum(labels_2000_5000)+0.001), np.std(weighted_diffs_2000_5000)))
print('5000-inf:  {:.3f}, std = {:.3f}'.format(sum(weighted_diffs_5000)/(sum(labels_5000)+0.001), np.std(weighted_diffs_5000)))
print('')

print('[diff-pncv-rate]')
print('1-inf:     {:.3f}, std = {:.3f}'.format(np.mean(pncv_diffs_1), np.std(pncv_diffs_1)))
print('5-inf:     {:.3f}, std = {:.3f}'.format(np.mean(pncv_diffs_5), np.std(pncv_diffs_5)))
print('10-inf:    {:.3f}, std = {:.3f}'.format(np.mean(pncv_diffs_10), np.std(pncv_diffs_10)))
print('0:         {:.3f}, std = {:.3f}'.format(np.mean(pncv_diff_0), np.std(pncv_diff_0)))
print('1-10:      {:.3f}, std = {:.3f}'.format(np.mean(pncv_diffs_1_10), np.std(pncv_diffs_1_10)))
print('10-100:    {:.3f}, std = {:.3f}'.format(np.mean(pncv_diffs_10_100), np.std(pncv_diffs_10_100)))
print('100-1000:  {:.3f}, std = {:.3f}'.format(np.mean(pncv_diffs_100_1000), np.std(pncv_diffs_100_1000)))
print('1000-2000: {:.3f}, std = {:.3f}'.format(np.mean(pncv_diffs_1000_2000), np.std(pncv_diffs_1000_2000)))
print('2000-5000: {:.3f}, std = {:.3f}'.format(np.mean(pncv_diffs_2000_5000), np.std(pncv_diffs_2000_5000)))
print('5000-inf:  {:.3f}, std = {:.3f}'.format(np.mean(pncv_diffs_5000), np.std(pncv_diffs_5000)))
print('')

print('[diff-pncv-weighted_rate]')
print('1-inf:     {:.3f}, std = {:.3f}'.format(sum(pncv_weighted_diffs_1)/(sum(labels_1)+0.001), np.std(pncv_weighted_diffs_1)))
print('5-inf:     {:.3f}, std = {:.3f}'.format(sum(pncv_weighted_diffs_5)/(sum(labels_5))+0.001, np.std(pncv_weighted_diffs_5)))
print('10-inf:    {:.3f}, std = {:.3f}'.format(sum(pncv_weighted_diffs_10)/(sum(labels_10)+0.001), np.std(pncv_weighted_diffs_10)))
print('1-10:      {:.3f}, std = {:.3f}'.format(sum(pncv_weighted_diffs_1_10)/(sum(labels_1_10)+0.001), np.std(pncv_weighted_diffs_1_10)))
print('10-100:    {:.3f}, std = {:.3f}'.format(sum(pncv_weighted_diffs_10_100)/(sum(labels_10_100)+0.001), np.std(pncv_weighted_diffs_10_100)))
print('100-1000:  {:.3f}, std = {:.3f}'.format(sum(pncv_weighted_diffs_100_1000)/(sum(labels_100_1000)+0.001), np.std(pncv_weighted_diffs_100_1000)))
print('1000-2000: {:.3f}, std = {:.3f}'.format(sum(pncv_weighted_diffs_1000_2000)/(sum(labels_1000_2000)+0.001), np.std(pncv_weighted_diffs_1000_2000)))
print('2000-5000: {:.3f}, std = {:.3f}'.format(sum(pncv_weighted_diffs_2000_5000)/(sum(labels_2000_5000)+0.001), np.std(pncv_weighted_diffs_2000_5000)))
print('5000-inf:  {:.3f}, std = {:.3f}'.format(sum(pncv_weighted_diffs_5000)/(sum(labels_5000)+0.001), np.std(pncv_weighted_diffs_5000)))
print('')

print('[diff-ensemble-rate]')
print('1-inf:     {:.3f}, std = {:.3f}'.format(np.mean(ensemble_diffs_1), np.std(ensemble_diffs_1)))
print('5-inf:     {:.3f}, std = {:.3f}'.format(np.mean(ensemble_diffs_5), np.std(ensemble_diffs_5)))
print('10-inf:    {:.3f}, std = {:.3f}'.format(np.mean(ensemble_diffs_10), np.std(ensemble_diffs_10)))
print('0:         {:.3f}, std = {:.3f}'.format(np.mean(ensemble_diff_0), np.std(ensemble_diff_0)))
print('1-10:      {:.3f}, std = {:.3f}'.format(np.mean(ensemble_diffs_1_10), np.std(ensemble_diffs_1_10)))
print('10-100:    {:.3f}, std = {:.3f}'.format(np.mean(ensemble_diffs_10_100), np.std(ensemble_diffs_10_100)))
print('100-1000:  {:.3f}, std = {:.3f}'.format(np.mean(ensemble_diffs_100_1000), np.std(ensemble_diffs_100_1000)))
print('1000-2000: {:.3f}, std = {:.3f}'.format(np.mean(ensemble_diffs_1000_2000), np.std(ensemble_diffs_1000_2000)))
print('2000-5000: {:.3f}, std = {:.3f}'.format(np.mean(ensemble_diffs_2000_5000), np.std(ensemble_diffs_2000_5000)))
print('5000-inf:  {:.3f}, std = {:.3f}'.format(np.mean(ensemble_diffs_5000), np.std(ensemble_diffs_5000)))
print('')

print('[diff-ensemble-weighted_rate]')
print('1-inf:     {:.3f}, std = {:.3f}'.format(sum(ensemble_weighted_diffs_1)/(sum(labels_1)+0.001), np.std(ensemble_weighted_diffs_1)))
print('5-inf:     {:.3f}, std = {:.3f}'.format(sum(ensemble_weighted_diffs_5)/(sum(labels_5))+0.001, np.std(ensemble_weighted_diffs_5)))
print('10-inf:    {:.3f}, std = {:.3f}'.format(sum(ensemble_weighted_diffs_10)/(sum(labels_10)+0.001), np.std(ensemble_weighted_diffs_10)))
print('1-10:      {:.3f}, std = {:.3f}'.format(sum(ensemble_weighted_diffs_1_10)/(sum(labels_1_10)+0.001), np.std(ensemble_weighted_diffs_1_10)))
print('10-100:    {:.3f}, std = {:.3f}'.format(sum(ensemble_weighted_diffs_10_100)/(sum(labels_10_100)+0.001), np.std(ensemble_weighted_diffs_10_100)))
print('100-1000:  {:.3f}, std = {:.3f}'.format(sum(ensemble_weighted_diffs_100_1000)/(sum(labels_100_1000)+0.001), np.std(ensemble_weighted_diffs_100_1000)))
print('1000-2000: {:.3f}, std = {:.3f}'.format(sum(ensemble_weighted_diffs_1000_2000)/(sum(labels_1000_2000)+0.001), np.std(ensemble_weighted_diffs_1000_2000)))
print('2000-5000: {:.3f}, std = {:.3f}'.format(sum(ensemble_weighted_diffs_2000_5000)/(sum(labels_2000_5000)+0.001), np.std(ensemble_weighted_diffs_2000_5000)))
print('5000-inf:  {:.3f}, std = {:.3f}'.format(sum(ensemble_weighted_diffs_5000)/(sum(labels_5000)+0.001), np.std(ensemble_weighted_diffs_5000)))
print('')

total_diffs = [abs(x - y) / float(y+0.1) for (x, y) in zip(preds, labels)]
total_pncv_diffs = [abs(x - y) / float(y+0.1) for (x, y) in zip(pncvs, labels)]
total_ensemble_diffs = [abs(x - y) / float(y+0.1) for (x, y) in zip(ensembles, labels)]

preds_labels = pd.DataFrame({
    '1_label': labels,
    '2_pred': preds,
    '3_pncv': pncvs,
    '4_ensemble': ensembles,
    '5_extra': extras,
    '6_diff': total_diffs,
    '7_pncv_diff': total_pncv_diffs,
    '8_ensemble_diff': total_ensemble_diffs
}).sort_values(by='1_label')
preds_labels.to_csv(outputfile + '.by_label', index=False, float_format='%.3f')

preds_labels = preds_labels.sort_values(by='2_pred')
preds_labels.to_csv(outputfile + '.by_pred', index=False, float_format='%.3f')

preds_labels = preds_labels.sort_values(by='6_diff')
preds_labels.to_csv(outputfile + '.by_diff', index=False, float_format='%.3f')

preds_labels = preds_labels.sort_values(by='7_pncv_diff')
preds_labels.to_csv(outputfile + '.by_pncv_diff', index=False, float_format='%.3f')
