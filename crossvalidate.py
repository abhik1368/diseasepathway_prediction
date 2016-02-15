    __author__ = 'abhikseal'

import pandas as pd
import numpy as np
import scipy
import scipy.io
import scipy.sparse
from scipy.sparse import hstack
from scipy.sparse import vstack
#from numpy import linalg as LA
import scipy.linalg
import networkx as nx
from sklearn.preprocessing import normalize
import time
import mmap
import random

def read_edgelist2mat(filename):
    """

    :param filename:
    :return:
    """
    fh=open(filename, 'rU')
    G=nx.read_edgelist(fh,delimiter=' ',comments='#', nodetype=int,data=(('weight',float),))
    fh.close()
    return nx.to_numpy_matrix(G)

#def read(filename):
#    with open(filename, "rb") as f:
#        mm = mmap.mmap(f.fileno(), 0,prot=mmap.PROT_READ)
#        lines = mm.readline().splitlines()
#        matrix = []
#        for line in lines:
#            if line != "":
#                matrix.append(map(float, line.split("\t")))
#    return matrix

def lapnormadj(A):

    import scipy
    import numpy as np
    from scipy.sparse import csgraph
    n,m = A.shape
    d1 = A.sum(axis=1).flatten()
    d2 = A.sum(axis=0).flatten()
    d1_sqrt = 1.0/scipy.sqrt(d1)
    d2_sqrt = 1.0/scipy.sqrt(d2)
    d1_sqrt[scipy.isinf(d1_sqrt)] = 0
    d2_sqrt[scipy.isinf(d2_sqrt)] = 0
    la = np.zeros(shape=(n,m))

    for i in range(0,n):
        for j in range(0,m):
          la[i,j] = A[i,j]/(d1_sqrt[i]*d2_sqrt[j])

    #D1 = scipy.sparse.spdiags(d1_sqrt, [0], n,m, format='coo')
    #D2 = scipy.sparse.spdiags(d2_sqrt, [0], n,m, format='coo')


    return  scipy.sparse.coo_matrix(la)

def laplaceNorm(A):
    """

    :param mat: Adjacency matrix
    :return: laplacian normalized matrix
    """
    import scipy
    from scipy.sparse import csgraph
    n,m = A.shape
    diags = A.sum(axis=1).flatten()
    diags_sqrt = 1.0/scipy.sqrt(diags)
    diags_sqrt[scipy.isinf(diags_sqrt)] = 0
    #print diags_sqrt
    DH = scipy.sparse.spdiags(diags_sqrt, [0], m,n, format='coo')
    return  scipy.sparse.coo_matrix(DH * A * DH)


def threshold(similarity,t=0.3):

    similarity[similarity < t] = 0

    return scipy.sparse.coo_matrix(similarity)

def graphknn(similarity,K = 15):

    m  = similarity.as_matrix()
    A = (np.ones(m.shape)-np.eye(m.shape[0]))*m
    A = A.astype(float)
    # and sparsifying by only keeping the n most similar entries

    for i in range(A.shape[0]):
        s = np.sort(A[i])
        s = s[::-1] #reverse order
        A[i][A[i]<s[K]] = 0

    # This makes it symmetrical
    A = np.fmax(A, A.T)

    return scipy.sparse.coo_matrix(A)



def transition(omimn,omim,pd,PPI,tissue,pc,cp,oid):

    #transition(omimd,pr_di,Tissue,pr_co,co_path,omimID)
    """
        Generate the transition matrix
        :param omim:
        :param pd:
        :param ppi:
        :param pc:
        :param cp:
        :return:
        """

    # Reading Tissue based Protein Protein Interaction data
    #file = '/Users/abhikseal/DTPProject/PPIdata/%d.ppi.csv' % tissue
    #print filename
    #PPI = pd.read_csv(filename,delimiter=",",index_col=0).as_matrix()
    #PPI = read_edgelist2mat(file)
    #PPI = scipy.sparse.coo_matrix(PPI)

    PPIm = scipy.sparse.coo_matrix(normalize(laplaceNorm(PPI),norm="l1",axis=1))
    #print "Loaded PPI Matrix of shape : ", PPIm.shape

    # Convert the matrices to scipy sparse matrix
    Indx = omim.columns.get_loc(str(oid))
    pd[pd.columns[Indx]] = 0

    #omimn = threshold(o,t)

    pdn = normalize(lapnormadj(pd.as_matrix()),norm="l1",axis=1)
    pcn = normalize(lapnormadj(pc.as_matrix()),norm="l1",axis=1)
    cpn = normalize(lapnormadj(cp.as_matrix()),norm="l1",axis=1)

    #pdn = scipy.sparse.coo_matrix(pd.values)
    #pcn = scipy.sparse.coo_matrix(pc.values)
    #cpn = scipy.sparse.coo_matrix(cp.values)

    #omimn = normalize(omimn,norm='l1',axis=0)

    # Generate empty matrices for transition matrix
    DC = scipy.sparse.coo_matrix((omimn.shape[1], pcn.shape[0]))
    DP = scipy.sparse.coo_matrix((omimn.shape[1] ,cpn.shape[0]))
    PPath = scipy.sparse.coo_matrix((pcn.shape[1],cpn.shape[0]))
    CC = scipy.sparse.coo_matrix((pcn.shape[0],pcn.shape[0]))
    PaPa = scipy.sparse.coo_matrix((cpn.shape[0],cpn.shape[0]))

    r1 = hstack([omimn,pdn,DC,DP])
    r2 = hstack([pdn.T,PPIm,pcn.T,PPath])
    r3 = hstack([DC.T,pcn,CC,cpn.T])
    r4 = hstack([DP.T,PPath.T,cpn,PaPa])
    trans = vstack([r1,r2,r3,r4])
    return trans

def spnorm(a):
    return np.sqrt(((np.power(a.data,2)).sum()))


def rwr(transition,PT,r=0.7):

    #Stop criteria
    stop = 1e-07
    PO = PT
    #Tr  =  normalize(transition, norm='l1', axis=0)
    Tr = transition
    while True:

        PX = (1-r)* Tr.T * PT + (r * PO)

        #delta =  (LA.norm(PX,axis=0) - LA.norm(PT,axis=0))
        #print LA.norm(PX,axis=0)
        #print LA.norm(PT,axis=0)
        #print delta

        delta =  spnorm(PX) - spnorm(PT)

        if delta < stop :
            #print delta ,"\n"
            break

        PT = PX
    #fMat = normalize(PT, norm='l1', axis=0)
    return PT



def main():

    # Total number of Nearest neighbours to check
    knn  = [0.3,0.4,0.5]


    # Read the test file

    test = pd.read_csv("/home/abseal/PHD_Thesis2/testcase_1.csv")
    omimName = test['disease'].unique()

    # Reading the files from the disk

    print " Reading data from the files for threshold  ... \n"

    start_time = time.time()

    omimd = pd.read_csv("/home/abseal/PHD_Thesis2/omimmat.txt",delimiter="\t",index_col=0)
    pr_di = pd.read_csv("/home/abseal/PHD_Thesis2/protein_disease_m.csv",header=None)
    pr_co = pd.read_csv("/home/abseal/PHD_Thesis2/protein_complex_m.csv",header=None)
    co_path =  pd.read_csv("/home/abseal/PHD_Thesis2/complex_pathway_m.csv",header=None)
    print("--- File Readings  %s seconds ---" % (time.time() -  start_time))
    for k in knn:
        omim = threshold(omimd,k)
        # calling laplace after KNN or omim matrix
        omimn = normalize(laplaceNorm(omim),norm="l1",axis=0)
        print "Working on graph thres : %.2f" %k
        dataResult = pd.read_csv("/home/abseal/PHD_Thesis2/Data/AllNames.csv",delimiter=',',index_col=0)
        # Start the timing for each graphs
        s_time = time.clock()
        for i in range(0,len(omimName)):

            #Get the tissue, genes and pathwya from the dataframe
            s_time = time.time()
            omimID = omimName[i]
            subfr = test[test['disease'] == omimID]
            Tissue = subfr['tissue'].unique()
            #genes = subfr[subfr['tissue'] == Tissue[0]]['entrezid'].unique()
            pathway = subfr[subfr['tissue'] == Tissue[0]]['pathway'].unique()
            randPath = random.choice(pathway)

            #print " Tissue ID selected & for pathway : " , Tissue[0],randPath
            filename = '/home/abseal/PHD_Thesis2/PPIdata/%d.ppi.csv' % Tissue[0]
            PPI = pd.read_csv(filename,delimiter=",",index_col=0).as_matrix()
            T  =  transition(omimn,omimd,pr_di,PPI,Tissue[0],pr_co,co_path,omimID)
            Indx1 = omimd.columns.get_loc(str(omimID))
            pathInd = co_path.columns.get_loc(str(randPath))
            Indx2 = 5080+9998+1826+int(pathInd)-1
            PT = np.zeros((T.shape[0],1))
            PT[Indx1] = 1 * 0.5
            PT[Indx2] = 1 * 0.5
            print " Running for OMIN disease " , omimID
            fPredict = rwr(T,PT,0.7)
            fPredict = normalize(fPredict, norm='l1', axis=0)
            dataResult[str(omimID)] = fPredict
            del(T)
        end = time.clock()
        print("--- RWR execution on %s run for  %s seconds ---" % (k,(end -  s_time)))

        print (" Writing Results file ..\n ")
        filename = "/home/abseal/PHD_Thesis2/Results/set1/laplace_testresults_thres__0.5_%.2f.csv" %k
        print "File name :  Results shape" , filename , dataResult.shape
        dataResult.to_csv(filename, sep=',')

if __name__ == "__main__":

    main()
    print ("\nAll files written to disk")

