## BLEU Variant
The calculation of BLEU variant and visualization of 
different smoothing function cn be found in [RQ1.ipynb](RQ1.ipynb)

## Human Evaluation
To find which BLEU is correlated with the human perception the most, we conduct the human evaluation.
### Human agreement
To verify the agreement among the annotators, we calculate the 
Krippendorff’s alpha  and Kendall rank correlation coefficient (Kendall’s Tau) values
```
cd human_evaluation
python cal_human_agreement.py
```
output is in [human_aggrement_log](human_evaluation/human_aggrement_log)

    Krippendorff's alpha for ordinal metric: 0.9379082476938598
    Kendall’s tau (Volunteer#0 and Volunteer1): KendalltauResult(correlation=0.8818756360626245, pvalue=4.799153353422063e-74)
    Kendall’s tau (Volunteer#0 and Volunteer2): KendalltauResult(correlation=0.8754440515790167, pvalue=5.969465932428485e-73)
    Kendall’s tau (Volunteer#0 and Volunteer3): KendalltauResult(correlation=0.8779315598626052, pvalue=2.6902016561319257e-72)
    Kendall’s tau (Volunteer#0 and Volunteer4): KendalltauResult(correlation=0.8629706588961966, pvalue=2.419564747256711e-70)
    Kendall’s tau (Volunteer#1 and Volunteer2): KendalltauResult(correlation=0.9915302032792017, pvalue=2.884458316322982e-94)
    Kendall’s tau (Volunteer#1 and Volunteer3): KendalltauResult(correlation=0.9233121983682175, pvalue=8.267448969494009e-81)
    Kendall’s tau (Volunteer#1 and Volunteer4): KendalltauResult(correlation=0.927744951026552, pvalue=4.5594118567074983e-82)
    Kendall’s tau (Volunteer#2 and Volunteer3): KendalltauResult(correlation=0.9191889302619408, pvalue=4.688924475447896e-80)
    Kendall’s tau (Volunteer#2 and Volunteer4): KendalltauResult(correlation=0.9264126186328877, pvalue=8.720259936215406e-82)
    Kendall’s tau (Volunteer#3 and Volunteer4): KendalltauResult(correlation=0.9214534425556004, pvalue=9.200538875489352e-80)
The value of Krippendorff’s alpha is 0.93, and the values of pairwise Kendall’s Tau range from 0.87 to 0.99, which indicates that there is a high degree of agreement between the 5 annotators and the scores are reliable.
### Correlation Coefficient
 we use Kendall’s rank correlation coefficient $\tau$ and Spearman correlation coefficient $\rho$ to measure the correlation between the human evaluation and each BLEU variant.
 
 ```
cd human_evaluation
python calculate_correlation.py
```

output is in [correlatio_coefficient_log](human_evaluation/correlatio_coefficient_log_1)

    ****************************************************************************************************
    Aggregation way: arithmetic_mean
    --------------------------------------------------
    corpus size 1
    **************************************************
    BLEU-DCOM
    KendalltauResult(correlation=0.32795815163032604 pvalue=0.0) 
    SpearmanrResult(correlation=0.6895326527944815, pvalue=0.0)
    BLEU-FC
    KendalltauResult(correlation=0.3279401480296059 pvalue=0.0) 
    SpearmanrResult(correlation=0.6895231915250678, pvalue=0.0)
    BLEU-DC
    KendalltauResult(correlation=0.54433974794959 pvalue=0.0) 
    SpearmanrResult(correlation=0.7507003707094376, pvalue=0.0)
    BLEU-CN
    KendalltauResult(correlation=0.47533314662932585 pvalue=0.0) 
    SpearmanrResult(correlation=0.6613170674767227, pvalue=0.0)
    BLEU-NCS
    KendalltauResult(correlation=0.3702766953390678 pvalue=0.0) 
    SpearmanrResult(correlation=0.5330935169669554, pvalue=0.0)
    BLEU-RC
    KendalltauResult(correlation=0.32795815163032604 pvalue=0.0) 
    SpearmanrResult(correlation=0.6895326527944815, pvalue=0.0)
    ......