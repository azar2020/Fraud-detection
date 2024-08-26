import pandas as pd
import csv

# Here Test_MUP_NPI, is a portion of the entire MUP_NPI data where we consiuder out test data
# I added 'ISFRAUD' column to the dataframe to indicate positive rows.
# Here, for example, my file name is Test_MUP_NPI with size 321863

Test_MUP_NPI = pd.read_csv("MUP_NPI_TS.csv",quotechar='"')

TotalTestPositive=0
TotalTestRows = len(Test_MUP_NPI)


# Here we count the total number of positive rows
for _,r in Test_MUP_NPI.iterrows():
    if r.loc['ISFRAUD']==True:
        TotalTestPositive+=1

# The Weighted evaluation simply assign a cost weight to each incorrectly predicted rows
# Here FN_cost is the False Negative Cost and FP_Cost is the False Positive Cost.
# A is a boolean array of size TotalTestRows which indicates the results
def WeightedEvaluation(A,FN_Cost=200,FP_Cost=1):
    TotalWeights = (TotalTestRows-TotalTestPositive)*FP_Cost + TotalPositive*FN_Cost
    Count = TotalWeights
    for i in range(TotalTestRows):
        if Test_MUP_NPI.iloc[i]['ISFRAUD']==True and A[i]==False:
            Count -= FN_Cost
        elif Test_MUP_NPI.iloc[i]['ISFRAUD']==False and A[i]==True:
            Count -= FP_Cost
    return (Count/TotalWeights)*100


# Total Cost-Sensitive Cost calculation based on the cost sensitive paper

def TotalCost(A):
    I=4000
    K=20
    C=0    
    for i in range(TotalTestRows):
        yi = Test_MUP_NPI.iloc[i]['ISFRAUD']
        C += yi*A[i]*I+K*yi*(1-A[i])*Test_MUP_NPI.iloc[i]['Tot_Drug_Cst']+\
             (1-yi)*A[i]*I
    return C/1000000
    
# An example

A = [False]*TotalTestRows
print (TotalCost(A))
A = [True]*TotalTestRows
print (TotalCost(A))
