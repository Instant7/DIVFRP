#Author: Wen-Kai Huang
#SMIE, Sun-yat sen University
#huang.wenkai@foxmail.com
#blog: http://www.cnblogs.com/instant7
#2014.12

from numpy import *
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    f = file('./R15.txt')
    xArr = []
    for l in f.readlines():
        if len(l) < 3:
            break
        l = l.strip()
        l = l.split('\t')
        xLineArr = []
        for i in [0,1]:
            xLineArr.append(float(l[i]));
        xArr.append(xLineArr)
    f.close()
    return mat(xArr)

def calcRP(points):
    '''calc the furthest point to the mean of points'''
    mean = points.mean(0)
    diff = points - mean
    dis = [sqrt((i.A**2).sum()) for i in diff]
    return points[dis.index(max(dis))]

def calcRank(points, refPoint):
    '''sort points according to their distant to refPoint. (Ascend)'''
    diffMat = points - refPoint
    dis = mat([sqrt((i.A**2).sum()) for i in diffMat])
    order = dis.argsort().A[0]
    rank = mat(points.A[order])
    return rank

def getClusterPoints(pointsIndex, dataMat):
    '''from dataMat get points for a specify cluster'''
    points = []
    for i in pointsIndex:
        points.append(dataMat[i,:].A[0].tolist())
    points = mat(points)
    return points

def calcDev(cluster):
    '''calc deviation of a cluster'''
    clusterString = cluster.tostring()
    if calcDev.__dict__.has_key('history'):
        if calcDev.history.has_key(clusterString):
            return calcDev.history[clusterString]
    else:
        calcDev.history = {}
    points = cluster
    points = mat(points)
    mean = points.mean(0)
    diff = points - mean
    dev = sum([sqrt((i.A**2).sum()) for i in diff]) / len(points)
    calcDev.history[clusterString] = dev
    return dev

def chooseNextSplit(clusterAssign):
    '''choose cluster for next split''' 
    devs = [calcDev(cluster) for cluster in clusterAssign]
    return devs.index(max(devs))

def calcD(rank, refPoint):
    '''calc differences between elements of all neighboring pairs in Rank'''
    D = []
    diffMat = rank - refPoint
    dis = [sqrt((i.A**2).sum()) for i in diffMat]
    for i in range(len(dis) - 1):
        D.append(dis[i+1] - dis[i])
    return D

def splitCluster(clusterIndex, clusterAssign):
    '''do cluster bipartitions'''
    points = mat(clusterAssign[clusterIndex])
    refPoint = calcRP(points)
    rank = calcRank(points, refPoint)
    D = calcD(rank, refPoint)
    splitIndex = D.index(max(D))
    Ci1 = rank[0:splitIndex+1].A
    Ci2 = rank[splitIndex+1:].A
    #Uncommete to seen split process
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(Ci1[:,0], Ci1[:,1], c='b')
    #ax.scatter(Ci2[:,0], Ci2[:,1], c='r')
    #ax.scatter(refPoint.A[:,0], refPoint.A[:,1], c='w')
    #plt.show()
    clusterAssign.append(Ci1)
    clusterAssign[clusterIndex] = Ci2

def calcEffectiveError(points):
    '''calc effective error of a cluster'''
    pointsString = points.tostring()
    if calcEffectiveError.__dict__.has_key('history'):
        if calcEffectiveError.history.has_key(pointsString):
            return calcEffectiveError.history[pointsString]
    else:
        calcEffectiveError.history = {}
    points = mat(points)
    mean = points.mean(0)
    diff = points - mean
    effectiveError = sum([sqrt((i.A**2).sum()) for i in diff])
    calcEffectiveError.history[pointsString] = effectiveError
    return effectiveError

def calcSumOfError(clusterAssign):
    '''calc sum-of-error (Je)'''
    sumOfError = 0
    for cluster in clusterAssign:
        sumOfError += calcEffectiveError(cluster)
    return sumOfError

def divisive(dataMat):
    '''divisive algorithm, split cluster untile each cluster has only one object '''
    m, n = shape(dataMat)
    clusterAssign = [dataMat.A]
    sumOfError = []
    while len(clusterAssign) != m:
        sumOfError.append(calcSumOfError(clusterAssign))
        nextSplit = chooseNextSplit(clusterAssign)
        try:
            splitCluster(nextSplit, clusterAssign)
        except:
            break #a cluster have two or more same point and all clusters have only one kind of point
    sumOfError.append(calcSumOfError(clusterAssign))
    return sumOfError

def calcDiff(sumOfError):
    '''calc difference between neighborhoods in sum-of-error'''
    d = [sumOfError[i-1] - sumOfError[i] for i in range(1, len(sumOfError))]
    return d

def fastSum(d):
    '''fast sum function, for calc sum(d) and sum(subD) only'''
    try:
        return fastSum.history[d]
    except:
        pass
    sumVal = sum(d)
    try:
        fastSum.history[d] = sumVal
    except:
        fastSum.history = {}
        fastSum.history[d] = sumVal
    return sumVal

def avgd(j, e, d, peakList, lam):
    '''j is index of element consider to select in d, e is last point in peakList's index in d'''
    if avgd.__dict__.has_key('history') and avgd.history['lam'] == lam:
        if avgd.history.has_key(j):
            return avgd.history[j]
    else:
        avgd.history = {}
        avgd.history['lam'] = lam
    if len(peakList) == 0 or j == 0: #exception 1: if m = 0
        ans = fastSum(d) / float(len(d))
        avgd.history[j] = ans
        return ans
    else:
        if j == e + 1:#exception2: if m != 0 and j = e + 1
            jLast = e; eLast = d.index(peakList[-1])
            ans = avgd(jLast, eLast, d, peakList, lam)
            avgd.history[j] = ans
            return ans
        else:#normal
            subD = d[e+1:j]
            ans = fastSum(subD) / float(len(subD))
            avgd.history[j] = ans
            return ans

def choosePeakList(d, lam):
    '''choose peak list from diff. lam stands for lambda, the parameter'''
    peakList = []
    goodSplit = []
    e = 0
    for i in range(len(d)):
        if d[i] >= lam * avgd(i, e, d, peakList, lam):
            peakList.append(d[i])
            goodSplit.append(i)#split that make new true clusters. index from 0
            e = i
    return peakList, goodSplit

def calcLBD(dataMat):
    '''calc LBD for this dataMat. The LBD is an extend version of original paper. 
    I add splitTimes of each true cluster number k for f'''
    sumOfError = divisive(dataMat)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(array(range(len(sumOfError))), array(sumOfError), )
    #ax.set_title('Sum of error')
    #plt.xlabel('Bipartition order i')
    #plt.ylabel('Je(i)')
    #plt.show()
    d = calcDiff(sumOfError)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(array(range(1, len(d)+1)), array(d), )
    #ax.set_title('Difference of neighboring Sum of error')
    #plt.xlabel('split times')
    #plt.ylabel('Je(i-1) - Je(i)')
    #plt.show()
    d = tuple(d)
    omega = 1.2; sigma = 0.1
    LBD = []
    lam = omega
    while True:
        peakList, goodSplit = choosePeakList(d, lam)
        k = len(peakList) + 1
        if k == 1:
            break
        elif k == 2 and LBD[-7][1] == k:
            break
        else:
            LBD.append([lam, k, goodSplit])
            lam += sigma
    return LBD

def chooseCandidateK(LBD):
    '''choose candidate K witch satisfy delta lambda > gamma'''
    gamma = 0.5
    maxK = LBD[0][1]
    LBDCpy = LBD[:]
    LBDCpy.reverse()
    LBDRev = LBDCpy
    candidateKwithSplit = []
    allK = []
    allKwithSplit = []
    for lbd in LBD:
        k = lbd[1]; split = lbd[2]
        if k not in allK:
            allK.append(k)
            allKwithSplit.append([k, split])
    allKwithSplit.sort(cmp = lambda x, y : cmp(x[0], y[0]), reverse=True)#sort by k
    for kwithSplit in allKwithSplit:
        k = kwithSplit[0]
        for i in LBD:
            if i[1] == k:
                lambdaI = i[0]
                break
        for i in LBDRev:
            if i[1] == k:
                lambdaJ = i[0]
                break
        deltaLambda = lambdaJ - lambdaI
        if deltaLambda > gamma:
            candidateKwithSplit.append(kwithSplit)
    return candidateKwithSplit

def divisiveToKClusters(dataMat, k):
    '''split dataMat to k  clusters'''
    m, n = shape(dataMat)
    clusterAssign = [dataMat.A]
    for t in range(k - 1): #split k - 1 times
        nextSplit = chooseNextSplit(clusterAssign)
        splitCluster(nextSplit, clusterAssign)
    return clusterAssign

def calcq(cluster):
    '''calc |Ci| * Jce(Ci) of a cluster'''
    return calcEffectiveError(cluster) * len(cluster)

def calcQandSize(clusterAssign):
    '''calc Q(SC) and size of all corresponding clusters'''
    QandSize = [[calcq(cluster), len(cluster)] for cluster in clusterAssign]
    QandSize.sort(cmp = lambda x, y : cmp(x[0], y[0]))#sort by Q
    return QandSize

def splitSpuriousCluster(clusterIndex, clusterAssign, spuriousClusterByD):
    '''do cluster bipartitions when split is not necessary'''
    points = mat(clusterAssign[clusterIndex])
    refPoint = calcRP(points)
    rank = calcRank(points, refPoint)
    D = calcD(rank, refPoint)
    splitIndex = D.index(max(D))
    Ci1 = rank[0:splitIndex+1].A
    Ci2 = rank[splitIndex+1:].A
    qCi1 = calcq(Ci1)
    qCi2 = calcq(Ci2)
    if qCi1 < qCi2:
        spuriousClusterByD.append(Ci1)
        clusterAssign[clusterIndex] = Ci2
    else:
        spuriousClusterByD.append(Ci2)
        clusterAssign[clusterIndex] = Ci1

def calcSpuriousClustersNum(kwithSplit, dataMat):
    alpha = 2; beta = 3
    clusterAssign = [dataMat.A]
    spuriousClusterByD = []
    for t in range(kwithSplit[1][-1] + 1): # split enough time accroding to d
        nextSplit = chooseNextSplit(clusterAssign)
        if t in kwithSplit[1]:
            splitCluster(nextSplit, clusterAssign)
        else:
            splitSpuriousCluster(nextSplit, clusterAssign, spuriousClusterByD)
    QandSize = calcQandSize(clusterAssign)
    maxSize = 0
    for i in range(kwithSplit[0]):
        if QandSize[i][1] > maxSize:
            maxSize = QandSize[i][1] 
    spuriousClustersNum = 0
    if  beta * QandSize[0][1] < maxSize:
        for i in range(1, len(QandSize)):
            if  QandSize[i][0] > alpha * QandSize[i-1][0] and \
                maxSize <= beta * QandSize[i][1]:
                spuriousClustersNum = i
                break
    return spuriousClustersNum


def chooseBestK(candidateKwithSplit, dataMat):
    QandSizeForAllK = []
    for kIndex in range(len(candidateKwithSplit)):
        kwithSplit = candidateKwithSplit[kIndex]
        spuriousClustersNum = calcSpuriousClustersNum(kwithSplit, dataMat)
        if spuriousClustersNum == 0:
            return kwithSplit
        elif kIndex == len(candidateKwithSplit) - 1:
            return kwithSplit
        else:
            nextKwithSplit = candidateKwithSplit[kIndex+1]
            if kwithSplit[0] - spuriousClustersNum >= nextKwithSplit[0]:
                return kwithSplit

def drawClusters(clusters, name = '', xlabel = '', ylabel = ''):
    cm = plt.get_cmap("prism")
    colors = [cm(float(i)/(len(clusters))) for i in xrange(len(clusters))]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
    'd', 'v', 'h', '>', '<', '1', '+', '3', '.', '*', '4', ',','2']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for clusterIndex in range(len(clusters)):
        points = clusters[clusterIndex]
        points = mat(points)
        x = points[:,0].A
        y = points[:,1].A
        markerStyle = scatterMarkers[clusterIndex % len(scatterMarkers)]
        ax.scatter(x, y, s=40, c=colors[clusterIndex] * len(x), marker=markerStyle)
    if name != '':
        ax.set_title(name)
    if xlabel != '':
        plt.xlabel(xlabel)
    if ylabel != '':
        plt.ylabel(ylabel)

def redistributeCluster(clusterAssign, spuriousClusterByD, spuriousClusterByQ):
    '''redistrute clusters'''
    spuriousClusters = spuriousClusterByD
    spuriousClusters.extend(spuriousClusterByQ)
    drawClusters(clusterAssign, 'Real Clusters')
    #plt.show()
    drawClusters(spuriousClusters, 'Spurious Clusters')
    finalClusters = [[] for i in range(len(clusterAssign))]
    means = [i.mean(0) for i in clusterAssign]
    pointGroups = clusterAssign[:]
    pointGroups.extend(spuriousClusters)
    for cluster in pointGroups:
        for point in cluster:
            dis = [sum((point - m)**2) for m in means]
            finalClusters[dis.index(min(dis))].append(point)
    for i in range(len(finalClusters)):
        finalClusters[i] = array(finalClusters[i])
    return finalClusters

def finalClusters(kwithSplit, dataMat):
    alpha = 2; beta = 3
    clusterAssign = [dataMat.A]
    spuriousClusterByD = []
    for t in range(kwithSplit[1][-1] + 1): # split enough time accroding to d
        nextSplit = chooseNextSplit(clusterAssign)
        if t in kwithSplit[1]:
            splitCluster(nextSplit, clusterAssign)
        else:
            splitSpuriousCluster(nextSplit, clusterAssign, spuriousClusterByD)
    QSizeCluster = [[calcq(cluster), len(cluster), cluster] for cluster in clusterAssign]
    QSizeCluster.sort(cmp = lambda x, y : cmp(x[0], y[0]))#sort by Q
    maxSize = 0
    for i in range(kwithSplit[0]):
        if QSizeCluster[i][1] > maxSize:
            maxSize = QSizeCluster[i][1] 
    spuriousClusterByQ = []
    if  beta * QSizeCluster[0][1] < maxSize:
        spuriousClusterByQ.append(QSizeCluster[0][2])
        for i in range(1, len(QSizeCluster)):
            if  QSizeCluster[i][0] > alpha * QSizeCluster[i-1][0] and \
                maxSize <= beta * QSizeCluster[i][1]:
                clusterAssign = [QSizeCluster[index][2] for index in range(i, len(QSizeCluster))]
                break
            spuriousClusterByQ.append(QSizeCluster[i][2])
    return redistributeCluster(clusterAssign, spuriousClusterByD, spuriousClusterByQ)

def DIVFRP(dataMat):
    LBD = calcLBD(dataMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pureLBD = [lbd[0:2] for lbd in LBD]
    ax.scatter(mat(pureLBD).A[:,0], mat(pureLBD).A[:,1])
    ax.set_title('LBD')    
    plt.xlabel('Lambda')
    plt.ylabel('Cluster number')
    #plt.show()

    candidateKwithSplit = chooseCandidateK(LBD)
    print 'candidateK'
    print [i[0] for i in candidateKwithSplit]

    bestKwithSplit = chooseBestK(candidateKwithSplit, dataMat)

    print 'bestK'
    print bestKwithSplit[0] - calcSpuriousClustersNum(bestKwithSplit, dataMat)

    clusters = finalClusters(bestKwithSplit, dataMat)
    return clusters

if __name__ == '__main__':
    import time
    startTime = time.time()
    dataMat = loadDataSet()
    #Uncomment if normalization is necessary
    #minVals = dataMat.min(0)
    #maxVals = dataMat.max(0)
    #ran = maxVals - minVals
    #dataMat[:, 0] = (dataMat[:, 0] - minVals[0, 0]) / ran[0, 0]
    #dataMat[:, 1] = (dataMat[:, 1] - minVals[0, 1]) / ran[0, 1]
    clusters = DIVFRP(dataMat)
    endTime = time.time()
    print 'time cost:', endTime - startTime, 's'

    drawClusters(clusters, 'After Redistribute')
    plt.show()
    
