#%% 

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import re
import random
from itertools import combinations
from sklearn.utils import resample
from sklearn.cluster import AgglomerativeClustering
from scipy import spatial
#%% Import data 
with open("/Users/florienmuntinga/Desktop/Msc Econometrics/B Computer Science/Paper /TVs-all-merged.json") as jsonFile:
    file = json.load(jsonFile)   

#%% Make data set such that I am able to work with it  
datalist = []
for i in range(len(list(file.values()))):
    if len(list(file.values())[i]) == 1:
        datalist.append(list(file.values())[i][0])
    else:
        for j in range(len(list(file.values())[i])):
            datalist.append(list(file.values())[i][j])
datalist = pd.DataFrame(datalist)
datalist = datalist.drop(['url', 'featuresMap'], axis = 1) #not needed

#%% Cleaning data: extract from data set (raw) & clean per variable. 
#Input: data (datalist), output: clean data for variable ModelIDs, Titles, KeyTitles, tvBrand, tvSize, tvShop.
def dataclean(data):
    
    
    #Titles: the main variable of my paper 
    RawTitles = (data['title'])
    #cleaning
    for i in range(len(RawTitles)):
            RawTitles[i] = RawTitles[i].replace('"', 'inch')
            RawTitles[i] = RawTitles[i].replace('\"', 'inch')
            RawTitles[i] = RawTitles[i].replace('inches', 'inch')
            RawTitles[i] = RawTitles[i].replace('-inch', 'inch')
            RawTitles[i] = RawTitles[i].replace(' inch', 'inch')
            RawTitles[i] = RawTitles[i].replace(' hz', 'hz')
            RawTitles[i] = RawTitles[i].replace('hertz', 'hz')
            RawTitles[i] = RawTitles[i].replace('Hertz', 'hz')
            RawTitles[i] = RawTitles[i].replace('-', '')
            RawTitles[i] = RawTitles[i].replace('/', '')
            RawTitles[i] = RawTitles[i].replace(':', '')
            RawTitles[i] = RawTitles[i].replace('–', '')
            RawTitles[i] = RawTitles[i].replace(';','')
            RawTitles[i] = RawTitles[i].replace('+', '')
            RawTitles[i] = RawTitles[i].replace('(', '')
            RawTitles[i] = RawTitles[i].replace(')', '')
            RawTitles[i] = RawTitles[i].replace('[', '')
    
    Titles = []
    for i in range(len(data)):
        temp2 = re.sub(r'[^\w\s]','', RawTitles[i]) 
        temp2 = temp2.lower()
        temp2 = temp2.split()
        
        Titles.append(temp2)
    
    #Only the unique words of all titles, needed for binary matrix 
    UniqueTitles = []
    for i in range(len(data)):
        for j in range(len(Titles[i])):
            if Titles[i][j] not in UniqueTitles:
                UniqueTitles.append(Titles[i][j])
    
    #Now I substract some variables which are given in titles (except for shop). 
    #These are used for the candidate matrix and I make sure these vectors only exists of numbers.
    AllBrands =  ["philips", "supersonic", "sharp", "samsung", 
               "toshiba", "hisense", "sony", "lg",  "sanyo",
               "coby", "panasonic", "rca", "vizio", "naxa",
               "sansui", "viewsonic", "avue", "insignia",
               "sunbritetv", "magnavox", "jvc", "haier", 
               "optoma", "nec", "proscan", "venturer", 
               "westinghouse", "pyle", "dynex", "magnavox", 
               "sceptre", "tcl", "mitsubishi", "open box", 
               "curtisyoung", "compaq", "hannspree", 
               "upstar", "azend", "seiki", "craig",
               "contex", "affinity", "hiteker", "epson", 
               "elo", "pyle", "hello kitty", "gpx", "sigmac", 
               "venturer", "elite"]
    
    TVBrand = np.zeros(len(Titles))
    for i in range(len(Titles)):
        for j in range(len(AllBrands)):
            if AllBrands[j] in Titles[i]:
                TVBrand[i] = j
    
    AllSizes = [0] * 100
    for i in range(0, len(AllSizes)):
        AllSizes[i] = str(i)+'inch'        

    TVSize = np.zeros(len(Titles))
    for i in range(len(AllSizes)):
        for j in range(len(Titles)):
            if AllSizes[i] in Titles[j]:
                TVSize[j] = i
    
    AllShops = ["bestbuy.com", "newegg.com", "amazon.com", "thenerds.com"]
    TVShoptemp = data['shop']
    
    TVShop = np.zeros(len(TVShoptemp))
    for i in range(len(TVShoptemp)):
        for j in range(len(AllShops)):
            if AllShops[j] in TVShoptemp[i]:
                TVShop[i] = j 

    #ModelIDS with this I know which are the real duplicates in the end 
    IDs = datalist['modelID']
    for i in range(len(IDs)):
        IDs[i] = IDs[i].lower() 

    
    return Titles, UniqueTitles, TVBrand, TVSize, TVShop, IDs

#Alldata = dataclean(datalist)
#Titles = Alldata[0]
#UniqueTitles = Alldata[1]
#TVBrand = Alldata[2]
#TVSize = Alldata[3]
#TVShop = Alldata[4]
#ModelIDs  = Alldata[5]
#%% Binary matrix containing 1 if that uniquetitle word is in the title 

def DefBinaryMatrix(Titles, UniqueTitles):
    BinaryMatrix = np.zeros((len(UniqueTitles), len(Titles)))
    for k in range(len(UniqueTitles)):
        for i in range(len(Titles)):
            if UniqueTitles[k] in Titles[i]:
                BinaryMatrix[k,i] = 1
    return BinaryMatrix

#BinaryMatrix = DefBinaryMatrix(Titles, UniqueTitles)

#%% Signature matrix with Min-Hashing
def DefSignatureMatrix(BinaryMatrix, UniqueTitles, R, B):
    
    N = B * R
    
    #Creating vector with random numbers needed for the hashfunction 
    def RandomIntegerVector(N):
        RandomInteger = []
        for i in range(N):
            r_int = random.randint(0,N)
            RandomInteger.append(r_int)
        return RandomInteger
    
    #Two random vectors, input for hashfucntion
    randomA = RandomIntegerVector(N)
    randomB = RandomIntegerVector(N)
    
    p = len(UniqueTitles)
    
    def HashFunction(x,a,b):
            return (a*x + b)%p

    def MinHashing(data, perm, randomA, randomB):
        
        SignatureMatrix = np.ones((N, len(data[0]))) * sys.maxsize
        for r in range(len(data)):
            hashvalue = []
            for k in range(perm):
                hashvalue.append(HashFunction(r,randomA[k],randomB[k]))
            
            for c in range(len(data[0])):
                if data[r][c] == 0:
                    continue
                for i in range(perm):
                    if SignatureMatrix[i][c] > hashvalue[i]:
                        SignatureMatrix[i][c] = hashvalue[i]
    
        return SignatureMatrix
    
    return MinHashing(BinaryMatrix, N, randomA, randomB)

#SignatureMatrix = DefSignatureMatrix(BinaryMatrix, UniqueTitles, 4, 25)

#%% (Hash)Function to append bands after one another 
def AppendBands(SignatureMatrix,R,B):
    
    N = B * R 
    z = np.arange(0,N, R)

    BandsTemp0 = {}
    for i in range(0,len(z)-1):
        BandsTemp0["band{0}".format(i)]=SignatureMatrix[z[i]:z[i+1]][:]
        BandsTemp0["band{0}".format(B-1)]=SignatureMatrix[z[B-1]:][:]

    BandsMatrix = []
    for i in range(0,B):
        BandsTemp = []
        for j in range(len(SignatureMatrix[1,:])):
            BandsTemp1 = BandsTemp0['band'+str(i)][:,j] 
            BandsTemp2 = [str(int) for int in BandsTemp1]
            
            for k in range(len(BandsTemp2)):
                BandsTemp2[k] = BandsTemp2[k].replace('.0','')
            
            BandsTemp3 = ''.join(BandsTemp2)
            BandsTemp.append(BandsTemp3)
            
        BandsMatrix.append(BandsTemp)
    
    return BandsMatrix

#BandsMatrix = AppendBands(SignatureMatrix,4,25)

#%% Creating candidate matrix  
    
def DefCandidateMatrix(Titles, BandsMatrix, R, B):
    
    CandidateMatrix = np.zeros((len(Titles), len(Titles)), dtype = np.int64)
    
    for i in range(len(Titles) - 1):
        for j in range(i+1, len(Titles)):
            for c in range(0,B):
                if (BandsMatrix[c][i] == BandsMatrix[c][j]):
                    CandidateMatrix[i,j] = 1
                    CandidateMatrix[j,i] = 1
    
    return CandidateMatrix

#CandidateMatrix = DefCandidateMatrix(Titles, BandsMatrix, 4, 25)

#%% Dissimilarity matrix  
def DefDissimilarityMatrix(Alldata, CandidateMatrix, BinaryMatrix, measure):
    
    TVBrand = Alldata[2]
    TVSize = Alldata[3]
    TVShop = Alldata[4]
    
    DissimilarityMatrix = CandidateMatrix.astype('float64')
    DissimilarityMatrix[DissimilarityMatrix==0] = 100000
    
    #product cannot be its own duplicate
    for i in range(0,len(DissimilarityMatrix)): 
        DissimilarityMatrix[i,i] = 0
    
    def JaccardDisimilarity(A,B):
        intersection = np.logical_and(A,B)
        union = np.logical_or(A,B)
        distance = 1 - (intersection.sum() / float(union.sum()))
        return distance
    
    def CosineDisimilarity(A, B):
        return spatial.distance.cosine(A, B)
    
    #given certain dissimilarity measure, change 1's into distances
    if (measure == 'jaccard'):
        for i in range(0,len(DissimilarityMatrix)):
            for j in range(i+1,len(DissimilarityMatrix)):
                if DissimilarityMatrix[i,j]==1 and TVShop[i] != TVShop[j] and TVBrand[i]==TVBrand[j] and  TVSize[j]==TVSize[j]:
                    DissimilarityMatrix[i,j] = JaccardDisimilarity(BinaryMatrix[:,i], BinaryMatrix[:,j])
                    DissimilarityMatrix[j,i] = JaccardDisimilarity(BinaryMatrix[:,i], BinaryMatrix[:,j])              
        
    elif (measure == 'cosine'):
        for i in range(0,len(DissimilarityMatrix)):
            for j in range(i+1,len(DissimilarityMatrix)):
                if DissimilarityMatrix[i,j]==1 and TVShop[i] != TVShop[j] and TVBrand[i]==TVBrand[j] and  TVSize[j]==TVSize[j]:
                    DissimilarityMatrix[i,j] = CosineDisimilarity(BinaryMatrix[:,i], BinaryMatrix[:,j])
                    DissimilarityMatrix[j,i] = CosineDisimilarity(BinaryMatrix[:,i], BinaryMatrix[:,j])      
                    
    return DissimilarityMatrix

#DissimilarityMatrix = DefDissimilarityMatrix(Alldata, CandidateMatrix, BinaryMatrix, 'jacard')

#%% Clustering function #goed, IDS kan weg boven? 

def ClusterFunction(DissimilarityMatrix, t, datalist):
    cluster = AgglomerativeClustering(n_clusters=None, affinity='precomputed', 
                                         linkage='complete', distance_threshold = t)
    
    clustering = cluster.fit(DissimilarityMatrix)
    labels = clustering.labels_
    
    PredictedDuplicates = []
    for i in range(0, clustering.n_clusters_):
        clusprods = np.where(labels == i)[0]
        if (len(clusprods)>1):
            PredictedDuplicates.extend(list(combinations(clusprods, 2)))
    
    IDs = datalist['modelID']
    for i in range(len(IDs)):
        IDs[i] = IDs[i].lower()
        
    RealDuplicates = []
    for modelID in IDs:
        if modelID not in RealDuplicates:
            dubs = np.where(IDs == modelID)[0]
            if (len(dubs)>1):
                RealDuplicates.extend(list(combinations(dubs, 2)))

    set_real = set(RealDuplicates)
    RealDuplicates = list(set_real)
    
    return PredictedDuplicates, RealDuplicates

#Duplicates = ClusterFunction(DissimilarityMatrix, 0.8,datalist); 
#PredictedDuplicates = Duplicates[0]; 
#RealDuplicates = Duplicates[1]

#%% Performance function
def Performance(PredictedDuplicates, RealDuplicates, CandidateMatrix, data):
    
    NPredDuplicates = len(PredictedDuplicates)
    NRealDuplicates = len(RealDuplicates)
    
    nTP=0; nFP=0
    for i in range(0,NPredDuplicates):
        if PredictedDuplicates[i] in RealDuplicates:
            nTP+=1
        else:
            nFP+=1
    
    nFN = NRealDuplicates - nTP
    
    nComps = np.count_nonzero(CandidateMatrix) * 0.5 
    nCompsPossible = len(data)*(len(data)-1) * 0.5
    
    comparisonFrac = nComps/nCompsPossible
    
    PQ = nTP/nComps
    PC = nTP/NRealDuplicates
    
    Precision = nTP/(nTP + nFP)
    Recall = nTP/(nTP + nFN)
    
    F1 = (2* Precision * Recall) / (Precision + Recall)
    F1Star = (2 * PQ * PC)/(PQ + PC)    
    
    return PQ, PC, F1, F1Star, comparisonFrac
#PerformanceOutcome = Performance(PredictedDuplicates, RealDuplicates, CandidateMatrix)
#%% Function that performs everything

def OverallFunction(data, R, B, treshold, measure):
    
    #Call data which is used 
    Alldata = dataclean(data)
    Titles = Alldata[0]
    UniqueTitles = Alldata[1]

    #Call all matrices 
    BinaryMatrix = DefBinaryMatrix(Titles, UniqueTitles)
    SignatureMatrix = DefSignatureMatrix(BinaryMatrix, UniqueTitles, R, B)
    BandsMatrix = AppendBands(SignatureMatrix, R, B)
    CandidateMatrix = DefCandidateMatrix(Titles, BandsMatrix, R, B)
    DissimilarityMatrix = DefDissimilarityMatrix(Alldata, CandidateMatrix, BinaryMatrix, measure)
    
    # Clustering, extracting real duplicates and predicted duplicates
    Duplicates = ClusterFunction(DissimilarityMatrix, treshold, data) 
    PredictedDuplicates = Duplicates[0] 
    RealDuplicates = Duplicates[1]
 
    # Performance measures
    PerformanceOutcome = Performance(PredictedDuplicates, RealDuplicates, CandidateMatrix,data)
    
    return PerformanceOutcome

#OverallFunction(datalist, 1, 100, 0.65, 'jaccard')

#%% Bootstrap

def bootstrap(data, integer):
    bootstrap = resample(data, replace=True, n_samples=len(data), random_state = integer)
    train = bootstrap.drop_duplicates()
    test = pd.concat([data, train]).drop_duplicates(keep=False)    
    return  train, test

train1, test1 =  bootstrap(datalist, 1)[0].reset_index(drop=True), bootstrap(datalist, 1)[1].reset_index(drop=True)
train2, test2 =  bootstrap(datalist, 2)[0].reset_index(drop=True), bootstrap(datalist, 2)[1].reset_index(drop=True)
train3, test3 =  bootstrap(datalist, 3)[0].reset_index(drop=True), bootstrap(datalist, 3)[1].reset_index(drop=True)
train4, test4 =  bootstrap(datalist, 4)[0].reset_index(drop=True), bootstrap(datalist, 4)[1].reset_index(drop=True)
train5, test5 =  bootstrap(datalist, 5)[0].reset_index(drop=True), bootstrap(datalist, 5)[1].reset_index(drop=True)

def Average(r,b,threshold, measure):
    F11 = OverallFunction(train1, r, b, threshold, measure)[2]
    F12 = OverallFunction(train2, r, b, threshold, measure)[2]
    F13 = OverallFunction(train3, r, b, threshold, measure)[2]
    F14 = OverallFunction(train4, r, b, threshold, measure)[2]
    F15 = OverallFunction(train5, r, b, threshold, measure)[2]
    F1_average = (F11+F12+F13+F14+F15)/5 
    return F11, F12, F13, F14, F15, F1_average 

#Average(2, 50, 0.8, 'cosine') #PQ, PC, F1, F1Star, comparisonFrac

#%% 

def EndScores(r, b, threshold, measure):
    results = np.zeros((5, 5))
    results[:,0] = OverallFunction(test1, r, b, threshold, measure)
    results[:,1] = OverallFunction(test2, r, b, threshold, measure)
    results[:,2] = OverallFunction(test3, r, b, threshold, measure)
    results[:,3] = OverallFunction(test4, r, b, threshold, measure)
    results[:,4] = OverallFunction(test5, r, b, threshold, measure)
    
    avg_results = results.mean(axis=1)
    return results, avg_results

#%% Jaccard results 

results1jac = EndScores(1,700,0.67,'jaccard') 
results2jac = EndScores(2, 350, 0.67, 'jaccard')
results3jac = EndScores(3, 235, 0.67, 'jaccard')
results4jac = EndScores(4, 175, 0.67, 'jaccard')
results5jac = EndScores(5, 140, 0.67, 'jaccard')
results6jac = EndScores(6, 120, 0.67, 'jaccard')
results7jac = EndScores(7, 100, 0.67, 'jaccard')
results8jac = EndScores(8, 90, 0.67, 'jaccard')
results9jac = EndScores(10, 70, 0.67, 'jaccard')

scoresjac = pd.DataFrame(columns=['R1B700', 'R2B530', 'R3B235', 'R4B175', 'R5B140',
                               'R6B120', 'R7B100', 'R8B90', 'R10B70'], 
                      index=['PQ', 'PC', 'F1', 'F1*', 'comparisonFrac'])

scoresjac['R1B100'] = results1jac[1]
scoresjac['R2B50'] = results2jac[1]
scoresjac['R3B34'] = results3jac[1]
scoresjac['R4B25'] = results4jac[1]
scoresjac['R5B20'] = results5jac[1]
scoresjac['R6B16'] = results6jac[1]
scoresjac['R7B15'] = results7jac[1]
scoresjac['R8B13'] = results8jac[1]
scoresjac['R10B10'] = results9jac[1]

#%% Cosine results

results1cos = EndScores(1,700,0.49,'cosine') 
results2cos = EndScores(2, 350, 0.49,'cosine')
results3cos = EndScores(3, 235, 0.49,'cosine')
results4cos = EndScores(4, 175, 0.49,'cosine')
results5cos = EndScores(5, 140, 0.49,'cosine')
results6cos = EndScores(6, 120, 0.49,'cosine')
results7cos = EndScores(7, 100, 0.49,'cosine')
results8cos = EndScores(8, 90, 0.49,'cosine')
results9cos = EndScores(10, 70, 0.49,'cosine')

scorescos = pd.DataFrame(columns=['R1B100', 'R2B50', 'R3B34', 'R4B25', 'R5B20',
                               'R6B16', 'R7B15', 'R8B13', 'R10B10'], 
                      index=['PQ', 'PC', 'F1', 'F1*', 'comparisonFrac'])

scorescos['R1B100'] = results1cos[1]
scorescos['R2B50'] = results2cos[1]
scorescos['R3B34'] = results3cos[1]
scorescos['R4B25'] = results4cos[1]
scorescos['R5B20'] = results5cos[1]
scorescos['R6B16'] = results6cos[1]
scorescos['R7B15'] = results7cos[1]
scorescos['R8B13'] = results8cos[1]
scorescos['R10B10'] = results9cos[1]


#%% Graphs PQ, PC, F1 and F1Star versus comparisonFrac

#Pair quality plot
plt.plot(scoresjac.iloc[4], scoresjac.iloc[0], 'black')
plt.plot(scorescos.iloc[4], scorescos.iloc[0], 'red')
plt.xlabel('Fraction of comparisons')
plt.ylabel('Pair quality')
plt.show

#Pair completeness plot
plt.plot(scoresjac.iloc[4], scoresjac.iloc[1], 'black')
plt.plot(scorescos.iloc[4], scorescos.iloc[1], 'red')
plt.xlabel('Fraction of comparisons')
plt.ylabel('Pair completeness')
plt.show

#F1 plot
plt.plot(scoresjac.iloc[4], scoresjac.iloc[2], 'black')
plt.plot(scorescos.iloc[4], scorescos.iloc[2], 'red')
plt.xlabel('Fraction of comparisons')
plt.ylabel('F1-measure')
plt.show

#F1* plot
plt.plot(scoresjac.iloc[4], scoresjac.iloc[3], 'black')
plt.plot(scorescos.iloc[4], scorescos.iloc[3], 'red')
plt.xlabel('Fraction of comparisons')
plt.ylabel('F1*-measure')
plt.show










