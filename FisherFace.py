import os
import numpy as np
from PIL import Image
from numpy import linalg
import math
import numpy

def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)

    [row, col] = x.shape
    r = np.zeros((1,col))

    for i in range(col):
        r[0,i] = linalg.norm(x[:,i])
    return r

def myLDA(A,Labels):
    # function [W,m]=myLDA(A,Label)
    # computes LDA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # m: mean of each projection


    classLabels = np.unique(Labels)
    classNum = len(classLabels)
    dim,datanum = A.shape
    totalMean = np.mean(A,1)
    partition = [np.where(Labels==label)[0] for label in classLabels]
    classMean = [(np.mean(A[:,idx],1),len(idx)) for idx in partition]

    #compute the within-class scatter matrix
    W = np.zeros((dim,dim))
    for idx in partition:
        W += np.cov(A[:,idx],rowvar=1)*len(idx)

    #compute the between-class scatter matrix
    B = np.zeros((dim,dim))
    for mu,class_size in classMean:
        offset = mu - totalMean
        B += np.outer(offset,offset)*class_size

    #solve the generalized eigenvalue problem for discriminant directions
    import scipy.linalg as linalg
    import operator
    ew, ev = linalg.eig(B,W+B)
    sorted_pairs = sorted(enumerate(ew), key=operator.itemgetter(1), reverse=True)
    selected_ind = [ind for ind,val in sorted_pairs[:classNum-1]]
    LDAW = ev[:,selected_ind]
    Centers = [np.dot(mu,LDAW) for mu,class_size in classMean]
    Centers = np.transpose(np.array(Centers))
    return LDAW,Centers, classLabels

def myPCA(A):
    # function [W,LL,m]=mypca(A)
    # computes PCA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # LL: eigenvalues
    # m: mean of columns of A

    # Note: "lambda" is a Python reserved word


    # compute mean, and subtract mean from every column
    [r,c] = A.shape
    m = np.mean(A,1) #这个求平均值的形式是什么 参数1  每行求出一个平均值，也就是每个样本的均值，得到的均值和样本维数相同

    A = A - np.transpose(np.tile(m, (c,1)))#每列减去均值

    B = np.dot(np.transpose(A), A) #ATA
    [d,v] = linalg.eig(B)#特征值特征向量 
    # v is in descending sorted order

    # compute eigenvectors of scatter matrix
    W = np.dot(A,v) #乘法W  小trick？ 
    Wnorm = ComputeNorm(W) 

    W1 = np.tile(Wnorm, (r, 1)) #r行 每行一个Wnorm
    W2 = W / W1 #标准化
    
    LL = d[0:-1] #特征值矩阵

    W = W2[:,0:-1]      #omit last column, which is the nullspace
    
    return W, LL, m


def read_faces(directory):
    # function faces = read_faces(directory)
    # Browse the directory, read image files and store faces in a matrix
    # faces: face matrix in which each colummn is a colummn vector for 1 face image
    # idLabels: corresponding ids for face matrix

    A = []  # A will store list of image vectors
    Label = [] # Label will store list of identity label
 
    # browsing the directory
    for f in os.listdir(directory):
        if not f[-3:] =='bmp':
            continue
        infile = os.path.join(directory, f)
        im = Image.open(infile)
        im_arr = np.asarray(im)
        im_arr = im_arr.astype(np.float32)

        # turn an array into vector
        im_vec = np.reshape(im_arr, -1)
        A.append(im_vec)
        name = f.split('_')[0][-1]
        Label.append(int(name))

    faces = np.array(A)
    faces = np.transpose(faces)
    idLabel = np.array(Label)

    return faces,idLabel



#PCA
K = 30
face_sample_num = 12
t_faces,t_idLabel = read_faces("./train")#；路径
#print(np.shape(t_idLabel))
a,b = np.shape(t_faces)
print(a)
print(b)
W, LL, m = myPCA(t_faces)    
print(np.shape(m))
W = W[:,:K]#取k个 
print(np.shape(W))
WT = np.transpose(W)
print(np.shape(np.tile(m,(b,1))))
y = np.dot(WT,(t_faces-np.transpose(np.tile(m, (b,1)))))#Y=WT(x-m)
h,num = np.shape(y)
print(np.shape(y))#30*120
num = (int)(num / face_sample_num)
z = np.zeros((h,num))#numpy matrix

for i in range(num):
    z[:,i] = np.transpose(np.mean(y[:,i*face_sample_num:(i+1)*face_sample_num],1))
za,zb=z.shape
print(za,zb)#30*10 10个人的模板 

#print(z)

#lda
K1 = 90
W0, LL, m = myPCA(t_faces)  
W1 = W0[:,:K1]
W1t = np.transpose(W1)
x = np.dot(W1t,(t_faces-np.transpose(np.tile(m, (b,1)))))
LDAW, Centers, classLabels = myLDA(x,t_idLabel) #centers 就是那个维度里的各种中心点


ca,cb = Centers.shape

#begin test
faces,idLabel = read_faces("./test")
a,b = faces.shape
y_pca = np.dot(WT,(faces-np.transpose(np.tile(m, (b,1)))))#得到test的pca特征

Wft = np.transpose(LDAW)
y_lda = np.dot(np.dot(Wft,W1t),(faces-np.transpose(np.tile(m, (b,1)))))#得到test的lda特征
print(y_lda)
#pcatest
C = np.zeros((num,num))

for i in range(b):

    difflist = []

    for j in range(num):
        difflist.append(numpy.linalg.norm(y_pca[:,i]-z[:,j]))
    
    k = difflist.index(min(difflist))
    C[int(i/12),k] += 1

print(C)

print(np.trace(C)/b)#confusion matrix

#ldatest

C = np.zeros((num,num))
for i in range(b):
    difflist = []
    for j in range(num):
        difflist.append(numpy.linalg.norm(y_lda[:,i]-Centers[:,j]))
    k = difflist.index(min(difflist))
    C[int(i/12),k] += 1

print(C)

print(np.trace(C)/b)



#feature fusion 混合模板
alpha = 0.5
Y = np.zeros((za+ca,zb))
Y[:za,:] = alpha*z
Y[za:za+ca,:] = Centers*(1-alpha)
print(np.shape(Y))
print(Y)
Y_fusion = np.zeros((za+ca,b))
Y_fusion[:za,:] = y_pca*alpha
Y_fusion[za:za+ca,:]= y_lda*(1-alpha)
print(Y_fusion)

#test_fusion
C = np.zeros((num,num))

for i in range(b):
    difflist = []
   
    for j in range(num):
        difflist.append(numpy.linalg.norm(Y_fusion[:,i]-Y[:,j]))
    k = difflist.index(min(difflist))
    C[int(i/12),k] += 1
print(C)

print(np.trace(C)/b)


from matplotlib import pyplot as plt

#PLOT

accuracy = []
alist = []
alpha = 0.1
while alpha<=0.9:
    alist.append(alpha)

    Y = np.zeros((za+ca,zb))
    Y[:za,:] = z*alpha
    Y[za:za+ca,:] = Centers*(1-alpha)

    C = np.zeros((num,num))

    for i in range(b):
        value_list = []
        for j in range(num):
            value_list.append(numpy.linalg.norm(Y_fusion[:,i]-Y[:,j]))
        k = value_list.index(min(value_list))
        C[int(i/12),k] += 1

    accuracy.append(np.trace(C)/b)

    alpha += 0.1

print(accuracy)

print(alist)


plt.plot(alist,accuracy,'or-')
plt.xlabel("alpha")
plt.ylabel("accuracy")
#plt.savefig('xcxplot1.png')
plt.show()
