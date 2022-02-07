#!/usr/bin/env python
# !-*-coding:utf-8 -*-
from scipy import stats
import pandas as pd
import krippendorff

if __name__ == '__main__':

    human_annotation = pd.read_excel('human_evaluation_1.xlsx')
    human1 = human_annotation["volunteer1"].tolist()
    human2 = human_annotation["volunteer2"].tolist()
    human3 = human_annotation["volunteer3"].tolist()
    human4 = human_annotation["volunteer4"].tolist()
    human5 = human_annotation["volunteer5"].tolist()
    human_label = [human1, human2, human3, human4, human5]
    print("Krippendorff's alpha for ordinal metric: {}".format(
        krippendorff.alpha(reliability_data=human_label, level_of_measurement='ordinal')))
    for i in range(5):
        for j in range(i + 1, 5):
            print("Kendall’s tau (Volunteer#{} and Volunteer{}): {}".format(i, j,
                                                                      stats.kendalltau(human_label[i], human_label[j])))
