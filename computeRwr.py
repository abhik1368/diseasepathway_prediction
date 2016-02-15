__author__ = 'abhikseal'

import pandas as pnd
import numpy as np
import scipy
import scipy.io
import scipy.sparse
from scipy.sparse import hstack
from scipy.sparse import vstack
import scipy.linalg
import sys
from sklearn.preprocessing import normalize
import time
import argparse
import os.path

# def read_edgelist2mat(filename):
#     """
#
#     :param filename:
#     :return:
#     """
#     import networkx as nx
#     fh=open(filename, 'rU')
#     G=nx.read_edgelist(fh,delimiter=' ',comments='#', nodetype=int,data=(('weight',float),))
#     fh.close()
#     return nx.to_numpy_matrix(G)
#
# def read(filename):
#     """
#
#     :param filename:
#     :return:
#     """
#     #lines = open(filename, 'rb',4194304).read().splitlines()
#     with open(filename, "r") as f:
#         mm = mmap.mmap(f.fileno(), 0,prot=mmap.PROT_READ)
#         lines = mm.readline().splitlines()
#         matrix = []
#         for line in lines:
#             if line != "":
#                 matrix.append(map(float, line.split("\t")))
#     return matrix

def lapnormadj(A):

    """
    Function to perform Laplacian Normalization on m x n matrix
    :param A: Adjacency Matrix
    :return: Laplace normalised matrix
    """

    import scipy
    import numpy as np
    from scipy.sparse import csgraph
    n,m = A.shape
    d1 = A.sum(axis=1).flatten()
    d2 = A.sum(axis=0).flatten()
    d1_sqrt = 1.0/scipy.sqrt(d1)
    d2_sqrt = 1.0/scipy.sqrt(d2)
    d1_sqrt[scipy.isinf(d1_sqrt)] = 10000
    d2_sqrt[scipy.isinf(d2_sqrt)] = 10000
    la = np.zeros(shape=(n,m))

    for i in range(0,n):
        for j in range(0,m):
          la[i,j] = A[i,j]/(d1_sqrt[i]*d2_sqrt[j])

    #D1 = scipy.sparse.spdiags(d1_sqrt, [0], n,m, format='coo')
    #D2 = scipy.sparse.spdiags(d2_sqrt, [0], n,m, format='coo')

    la[la < 1e-5] = 0

    return  scipy.sparse.coo_matrix(la)

def laplaceNorm(A):
    """
    Function to perform laplcian normalization of mxm matrix
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

def graphknn(similarity,K = 15):

    m  = similarity.as_matrix().astype(float)
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

def threshold(similarity,t=0.3):

    similarity[similarity < t] = 0

    return scipy.sparse.coo_matrix(similarity)

def transition(omim,pd,tissue,pc,cp,norm="laplace"):

    """
    Generate the transition matrix for the Bipartite Graph
    :param omim: Disease Similarity Matrix
    :param pd: Protein Disease matrix
    :param ppi: Protein-Protein Interaction Matrix
    :param pc:Protein Complex Matrix
    :param cp: Complex Pathway Matrix
    :return: Normalized Transition Matrix
    """

    # Reading Tissue based Protein Protein Interaction data
    #file = "/Users/abhikseal/DTPProject/PPIdata/%s.part" % tissue
    #file = "/Users/abhikseal/DTPProject/PPIdata/%s.part" % tissue

    try:
        file = '/Users/abhikseal/PHD_Thesis2/PPIdata/%d.ppi.csv' %tissue
        PPI = scipy.sparse.coo_matrix(pnd.read_csv(file,delimiter=",",index_col=0).as_matrix())
        print "Loaded PPI Matrix for tissue : %d " %tissue

    except (TypeError,Exception) :
        sys.exit("Unable to locate the PPI data file for tissue %d" %tissue)

    if norm == "laplace":

        omimn = normalize(laplaceNorm(omim),norm="l1",axis=0)
        pdn = normalize(lapnormadj(pd.as_matrix()),norm="l1",axis=1)
        pcn = normalize(lapnormadj(pc.as_matrix()),norm="l1",axis=1)
        cpn = normalize(lapnormadj(cp.as_matrix()),norm="l1",axis=1)
        PPI = normalize(laplaceNorm(PPI),norm="l1",axis=0)

    elif norm == "row":
        omimn = normalize(omim,norm="l1",axis=0)
        pdn = normalize(pd.as_matrix(),norm="l1",axis=0)
        pcn = normalize(pc.as_matrix(),norm="l1",axis=0)
        cpn = normalize(cp.as_matrix(),norm="l1",axis=0)
        PPI = normalize(PPI,norm="l1",axis=0)

    elif norm == "column":
        omimn = normalize(omim,norm="l1",axis=1)
        pdn = normalize(pd.as_matrix(),norm="l1",axis=1)
        pcn = normalize(pc.as_matrix(),norm="l1",axis=1)
        cpn = normalize(cp.as_matrix(),norm="l1",axis=1)
        PPI = normalize(PPI,norm="l1",axis=1)

    else :
        sys.exit("Enter the correct name for normalisation")

    DC = scipy.sparse.coo_matrix((omim.shape[1], pcn.shape[1]))

    DP = scipy.sparse.coo_matrix((omim.shape[1] ,cpn.shape[1]))
    PPath = scipy.sparse.coo_matrix((pcn.shape[0],cpn.shape[1]))
    CC = scipy.sparse.coo_matrix((pcn.shape[1],pcn.shape[1]))
    PaPa = scipy.sparse.coo_matrix((cpn.shape[1],cpn.shape[1]))

    r1 = hstack([omimn,pdn.T,DC,DP])
    r2 = hstack([pdn,PPI,pcn,PPath])
    r3 = hstack([DC.T,pcn.T,CC,cpn])
    r4 = hstack([DP.T,PPath.T,cpn.T,PaPa])

    trans = vstack([r1,r2,r3,r4])

    return trans

def spnorm(a):
    """
    :param a: Sparse Matrix
    :return: a sparse matrix
    """
    return np.sqrt(((np.power(a.data,2)).sum()))

def rwr(transition,PT,r=0.7):
    """

    :param transition: Get the spare Transition matrix
    :param PT: Intialization Vector
    :param r: restart probability
    :return: Numpy Matrix of predicted scores
    """
    #Stop criteria
    stop = 1e-07
    PO = PT
    Tr  =  transition

    while True:

        PX = (1-r)* Tr.T * PT + (r * PO)
        delta =  spnorm(PX) - spnorm(PT)

        if delta < stop :
            break

        PT = PX
    #fMat = normalize(PT, norm='l1', axis=0)
    OM = PT[0:5080]
    OM  = normalize(OM, norm='l1', axis=0)
    PP = PT[5080:15078]
    PP = normalize(PP, norm='l1', axis=0)
    CP = PT[15078:16904]
    CP  = normalize(CP, norm='l1', axis=0)
    PAT = PT[16904:19435]
    PAT  = normalize(PAT, norm='l1', axis=0)
    P = np.concatenate((OM,PP,CP,PAT),axis=0)

    return P

def convertFrame(results):
    """
    Append the results to the names file and return the full predicted results
    :param results: get the matrix of results
    :return:
    """
    # Read the names file containing all field names
    try:

        allname = pnd.read_csv("/Users/abhikseal/PHD_Thesis2/Data/AllNames.csv",delimiter=',',index_col=0)
        allname['prediction'] = results
        return allname
    except (RuntimeError,ValueError,TypeError,Exception) :
        sys.exit("All names files not found\n")

def create_net(df,DID,top=10):
    """

    :param df:
    :return: return dataframe of scores of omim ids, complex ids, and pathway ids
    """
    # Getting sorted data for each different types of data

    omim= df[0:5080].sort(['prediction'],ascending=[0])
    ppi = df[5080:15078].sort(['prediction'],ascending=[0])
    complex = df[15078:16904].sort(['prediction'],ascending=[0])
    pathway = df[16904:19435].sort(['prediction'],ascending=[0])
    pieces = [omim[0:top],ppi[0:top],complex[0:top],pathway[0:top]]
    net = pnd.concat(pieces)
    net['source'] = DID
    return net

def runPrediction(fileName,**param):
    """

    :param DID: Disease ID
    :param fileName: output filename to write
    :param norm:normalisation to perform on the matrices
    """

    print " Reading data from the files ..."
    start_time = time.time()

    pr_di = pnd.read_csv("/Users/abhikseal/PHD_Thesis2/protein_disease_m.csv",header=None)
    pr_co = pnd.read_csv("/Users/abhikseal/PHD_Thesis2/protein_complex_m.csv",header=None)
    co_path =  pnd.read_csv("/Users/abhikseal/PHD_Thesis2/complex_pathway_m.csv",delimiter=",",index_col=0)
    Omim_Data = pnd.read_table("/Users/abhikseal/PHD_Thesis2/omimmat.txt",index_col=0)
    #omim =  scipy.sparse.coo_matrix(Omim_Data.as_matrix())

    dnames = list(Omim_Data.index)

    omim = graphknn(Omim_Data,25)

    print("--- File Readings  %s seconds ---" % (time.time() -  start_time))

    # Get the parameters passed
    contents = param.get('contents')
    DID = param.get('input')
    Norm = param.get('norm')
    restart = param.get('restart')
    top = param.get('top')

    # Reading the disease tissue matrix
    dt = pnd.read_csv("/Users/abhikseal/PHD_Thesis2/Data/fullDiseaseTissue.csv",delimiter=',',index_col=None)

    if DID is not None:
        if int(DID) in dnames:
            f =  dt[dt['disease'] == DID]
            tissue = int(f['tissue'])
            print "Got the tissue ID : %d "  % tissue
            Indx = Omim_Data.columns.get_loc(str(DID))
            T  =  transition(omim,pr_di,tissue,pr_co,co_path,norm=Norm)
            del (omim,pr_di,pr_co,co_path)
            PT = np.zeros((T.shape[0],1))
            PT[Indx] = 1
            pr = rwr(T,PT,r=restart)
            print " RWR executed for Disease ID : %d" %DID
            del T
            predict = convertFrame(pr)
            net = create_net(predict,DID,top)
            net.to_csv(fileName, sep='\t',header=False)
        else :
            print " The OMIM Disease ID is not in the database ..\n"
            sys.exit()

    if contents is not None:
        for line in contents:
            DID = line.strip()
            print "Computing RWR for Disease ID : %s " %DID
            f =  dt[dt['disease'] == int(DID)]
            tissue = int(f['tissue'])
            print "Got the tissue ID : %d "  % tissue
            Indx = Omim_Data.columns.get_loc(str(DID))
            # Calling the transition matrix function
            T  =  transition(omim,pr_di,tissue,pr_co,co_path,norm=Norm)
            PT = np.zeros((T.shape[0],1))
            PT[Indx] = 1
            pr = rwr(T,PT,r=restart)
            #pr = normalize(pr, norm='l1', axis=0)
            print " RWR executed for Disease ID : %s " %DID

            del T
            #pr= normalize(pr, norm='l1', axis=0)
            predict = convertFrame(pr)
            net = create_net(predict,DID,top)
            with open(fileName, 'a') as f:
                net.to_csv(f, sep='\t',header=False)
            #net.to_csv(fileName, sep='\t')
        f.close()
    #del (omim,pr_di,pr_co,co_path)

if __name__ == "__main__":

    """
    Using command line arguments

    """

    parser = argparse.ArgumentParser(description="Disease Pathway Prediciton using Random Walk with Restart")
    parser.add_argument("--input", type = int, help="input the OMIM disease ID.")
    parser.add_argument("--file",help="Input txt file with Disease IDs. Each Diseases IDs should be in a new row .")
    parser.add_argument("--restart",type=float,required=True,default=0.7,help="Restart Probability for RWR")
    parser.add_argument("--out", required=True, help="Output file of predictions in '.csv' format.")
    parser.add_argument("--norm", help="laplace , Row and Column",default="laplace")
    parser.add_argument("--top",help="Get the top results",default=10,type=int)

    args = vars(parser.parse_args())
    print parser.parse_args()

    start_time = time.time()
    if os.path.isfile(args['out']):
        print " Removing Old %s file " %args['out']
        os.remove(args['out'])

    if args['file'] is None and args['input'] is None:
        print "Enter either a Disease ID or a txt file with Disease IDs"
        sys.exit()

    if args['file'] is None:
        runPrediction(args['out'],restart=args['restart'],top=args['top'],norm=args['norm'],input=args['input'])
        sys.exit("Output file %s generated .." %args['out'])
    else :
        try:
            read_file = open(args['file'])
            contents = read_file.readlines()
            read_file.close()
        except:
            print "Something went wrong with the file read!"


        runPrediction(args['out'],restart=args['restart'],top=args['top'],norm=args['norm'],contents=contents)
        print ("Output file %s generated ..")

    sys.exit("---Time executed  %s seconds ---" % (time.time() -  start_time))