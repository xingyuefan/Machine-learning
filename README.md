# Machine-learning
from numpy import *
import random
import numpy as np
from sklearn.cluster import KMeans
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
import csv
import pandas as pd
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

#计算两个点之间的欧氏距离
def calcuDistance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))

#根据距离质心的大小定义用户目前所处的等级
def UserRank(AC_num, ua_centroids):
    min = 100000
    userRank = mini = -1
    rows = shape(ua_centroids)[0]
    for i in range(rows):
        if calcuDistance(AC_num, ua_centroids[i][0]) < min:
            min = calcuDistance(AC_num, ua_centroids[i][0])
            mini = i
    return mini+1


#统计每个用户等级的人数
def UserRank_cnt():
    cntuserrank1 = 0
    cntuserrank2 = 0
    cntuserrank3 = 0
    cntuserrank4 = 0
    cntuserrank5 = 0
    rows = shape(UAKMat)[0]
    for i in range(rows):
        x = UserRank(UAKarray[i][0], ua_centroids)
        if x == 1:
            cntuserrank1 += 1
        elif x == 2:
            cntuserrank2 += 1
        elif x == 3:
            cntuserrank3 += 1
        elif x == 4:
            cntuserrank4 += 1
        else:
            cntuserrank5 += 1
    return cntuserrank1,cntuserrank2,cntuserrank3,cntuserrank4,cntuserrank5


#根据AC题量的多少给不同题目自定义难度等级
def ProbRank_Cnt(AKMat, pa_centroids):
    cntprobrank1 = 0
    cntprobrank2 = 0
    cntprobrank3 = 0
    cntprobrank4 = 0
    cntprobrank5 = 0
    rows1 = shape(AKMat)[0]
    rows2 = shape(pa_centroids)[0]
    ARKMat = AKMat
    ARKMat = np.insert(ARKMat, 1, values=0, axis=1)
    for i in range(rows1):  # i=0
        min = 1000000
        for j in range(rows2):
            if calcuDistance(AKMat[i], pa_centroids[j]) < min:
                min = calcuDistance(AKMat[i], pa_centroids[j])
                minJ = j
        ARKMat[i, 1] = 5-minJ
        if minJ == 0:
            cntprobrank5 += 1
        elif minJ == 1:
            cntprobrank4 += 1
        elif minJ == 2:
            cntprobrank3 += 1
        elif minJ == 3:
            cntprobrank2 += 1
        elif minJ == 4:
            cntprobrank1 += 1
    return ARKMat,cntprobrank1,cntprobrank2,cntprobrank3,cntprobrank4,cntprobrank5


#根据不同难度等级所带的标记不同，统计出不同难度等级的题量有多少
def showRecommend(APARKMat):
    ProbRank_Cnt(AKMat, pa_centroids)
    centroids0 = []
    centroids1 = []
    centroids2 = []
    centroids3 = []
    centroids4 = []
    row = shape(APARKMat)[0]
    for i in range(row):
        if APARKMat[i, 2] == 1:
            centroids0.append(APARKMat[i, 0])
        elif APARKMat[i, 2] == 2:
            centroids1.append(APARKMat[i, 0])
        elif APARKMat[i, 2] == 3:
            centroids2.append(APARKMat[i, 0])
        elif APARKMat[i, 2] == 4:
            centroids3.append(APARKMat[i, 0])
        elif APARKMat[i, 2] == 5:
            centroids4.append(PARKMat[i, 0])
    print('难度等级  个数  题号')
    print('难度等级1:', cntprobrank1, centroids0)
    print('难度等级2:', cntprobrank2, centroids1)
    print('难度等级3:', cntprobrank3, centroids2)
    print('难度等级4:', cntprobrank4, centroids3)
    print('难度等级5:', cntprobrank5, centroids4)


if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    # print('原始导入数据Problem_AC表：')
    PAMat = pd.read_csv(".\\Problem_AC.csv", sep=',', header=None)
    rows = shape(PAMat)[0]
    # print(PAMat)
    PAMat = mat(PAMat)
    # print('PAMat\'s shape:', PAMat.shape)
    # print(PAMat)
    # print('选取AC列用来做Kmeans聚类的矩阵')
    AKMat = PAMat[:, 1]

    #轮廓系数（Silhoute Coefficient Score）来确定最佳K值
    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    sc_scores = []
    for t in clusters:
        kmeans_model = KMeans(n_clusters=t).fit(AKMat)
        sc_score = silhouette_score(AKMat, kmeans_model.labels_, metric='euclidean')
        sc_scores.append(sc_score)
    # 绘制轮廓系数与不同类簇数量的关系曲线
    # print('簇个数（the number of clusters）:', clusters)
    # print('轮廓系数（Silhoute Coefficient Score）:', sc_scores)
    plt.plot(clusters, sc_scores, '*-')
    plt.xlabel('簇个数（the number of clusters）')
    plt.ylabel('轮廓系数（Silhoute Coefficient Score）')
    plt.title(u'用轮廓系数则来确定最佳的K值')
    plt.savefig("Silhoute Coefficient.png")
    plt.show()

    # 肘部法（The elbow method）则来确定最佳的K值
    K = range(1, 10)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(AKMat)
        meandistortions.append(sum(np.min(cdist(AKMat, kmeans.cluster_centers_, "euclidean"), axis=1)) / AKMat.shape[0])
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('簇个数（the number of clusters）')
    plt.ylabel(u'平均畸变程度')
    plt.title(u'用肘部法则来确定最佳的K值')
    plt.savefig("The elbow method.png")
    plt.show()

    #确定最佳K值后，人工对质心贴标签，确定对应的题目难度系数（等级为1，2，3，4，5且等级依次递增)
    # print('构造关于题目难度系数的聚类器')
    estimator1 = KMeans(n_clusters=5)
    # print('estimator1:', estimator1)
    estimator1.fit(AKMat)  # 聚类
    pa_centroids = estimator1.cluster_centers_  # 获取聚类中心
    # print('pa_centroids：')
    # print(pa_centroids)
    pa_centroids = pa_centroids[np.lexsort(-pa_centroids[:, ::-1].T)]
    print('不同难度等级题目的质心（被AC量）：')
    print(pa_centroids)
    # print('pa_centroid[0]：', pa_centroids[0])  #pa_centroid[0]： [19.10315985]
    # print('pa_centroids[1][0]:', pa_centroids[1][0])  #pa_centroids[1][0]: 180.88509316770254
    cntprobrank1 = cntprobrank2 = cntprobrank3 = cntprobrank4 = cntprobrank5 = 0
    ARKMat, cntprobrank1, cntprobrank2, cntprobrank3, cntprobrank4, cntprobrank5 = ProbRank_Cnt(AKMat, pa_centroids)
    # print('ARKMat:', ARKMat)
    # print('cntprobrank1:', cntprobrank1)
    PARKMat =  np.hstack((PAMat[:,0], ARKMat))#行合并
    # print('PARKMat:', PARKMat)
    PARKArray = array(PARKMat)
    # PARKMat = PARKMat[PARKMat[:, 1].argsort()]  # 按照第2列对行升序排序
    PARKArray = PARKArray[argsort(-PARKArray[:, 1])]  # 按照第2列对行降序排序
    # print('PARKArray:')
    # print(PARKArray)
    PARKMat = mat(PARKArray)
    # print('PARKMat:')
    # print(PARKMat)

    # 确定最佳K值后，人工对质心贴标签，确定对应的用户等级（等级为1，2，3，4，5且等级依次递增)
    # print('原始导入数据User_AC表：')
    UAMat = pd.read_csv(".\\User_AC.csv", sep=',', header=None)
    # print(UAMat)
    UAMat = mat(UAMat)
    # print('UAMat\'s shape:', UAMat.shape)
    # print(UAMat)
    # print('选取AC列用来做Kmeans聚类的矩阵')
    UAKMat = UAMat[:, 1]

    # print('构造关于确定用户等级的聚类器')
    estimator2 = KMeans(n_clusters=5)
    # print('estimator2:', estimator2)
    estimator2.fit(UAKMat)  # 聚类
    ua_centroids = estimator2.cluster_centers_  # 获取聚类中心
    # print('user\'s centroids：')
    # print(ua_centroids)
    ua_centroids = ua_centroids[np.lexsort(ua_centroids[:, ::-1].T)]
    print('不同用户等级质心（用户AC量）：')
    print(ua_centroids)
    # print('user\'s centroids[0]：', ua_centroids[0])  # ua_centroid[0]： [8.37009123]
    # print('user\'s centroids[1][0]:', ua_centroids[1][0])  # ua_centroids[1][0]: 63.30537019383742
    UAKMat = UAMat[:, 1]
    UAKarray = array(UAKMat)
    # print('UAKArray[0]:', UAKarray[0])
    # print('UAKArray[0][0]:', UAKarray[0][0])

    # 确定最佳K值后，绘制题目难度系数聚类柱状图
    x = [pa_centroids[0, 0], pa_centroids[1, 0], pa_centroids[2, 0], pa_centroids[3, 0], pa_centroids[4, 0]]
    y = [cntprobrank5, cntprobrank4, cntprobrank3, cntprobrank2, cntprobrank1]
    plt.bar(x, y, color="green", width=250)
    plt.xlabel("题号的被AC数")
    plt.ylabel("题的个数")  # 绘制聚类柱状图
    plt.title("对题目聚类成不同难度等级的柱状图")
    plt.savefig("p_histogram.png")
    plt.show()

    # 确定最佳K值后，绘制用户难度系数聚类柱状图
    cntuserrank1, cntuserrank2, cntuserrank3, cntuserrank4, cntuserrank5 = UserRank_cnt()
    xx = [ua_centroids[0, 0], ua_centroids[1, 0], ua_centroids[2, 0], ua_centroids[3, 0], ua_centroids[4, 0]]
    yy = [cntuserrank1, cntuserrank2, cntuserrank3, cntuserrank4, cntuserrank5]
    plt.bar(xx, yy, color="green", width=40)
    plt.xlabel("用户的AC数")
    plt.ylabel("人数")  # 绘制聚类柱状图
    plt.title("对用户聚类成不同级别的柱状图")
    plt.savefig("u_histogram.png")
    plt.show()

    # print('showUserRank_Cnt:')
    print('用户等级   人数')
    print('用户等级1：', cntuserrank1)
    print('用户等级2：', cntuserrank2)
    print('用户等级3：', cntuserrank3)
    print('用户等级4：', cntuserrank4)
    print('用户等级5：', cntuserrank5)

    APARKMat = np.insert(PARKMat, 3, values=1, axis=1)
    # print('APARKMat:', APARKMat)
    APARKArray = array(APARKMat)
    # print('APARKArray:', APARKArray)
    showRecommend(APARKMat)

    # #从此处修改
    # print('请输入您的Solved题量:',end=' ')
    # AC_num = int(input())  # Python3.x 中 input() 函数接受一个标准输入数据，返回为 string 类型。
    # if AC_num == -1:
    #     exit()
    # userRank = UserRank(AC_num, ua_centroids)
    # # print('您目前所处用户等级为%d' % userRank)

    # #同等难度等级题目推荐
    # recom_problem_id1 = []
    # for i in range(rows):
    #     if APARKArray[i, 2] == userRank:
    #         recom_problem_id1.append(APARKArray[i, 0])
    # # print('题目是：',recom_problem_id1)
    # # print('等级是：', recom_problem_rank1)
    # if len(recom_problem_id1)>=10:
    #     slice1 = random.sample(recom_problem_id1, 10)
    #     print('推荐同级：%d等级的题目: %s' % (userRank, slice1))
    # else:
    #     print('推荐同级：%d等级的题目: %s' % (userRank, recom_problem_id1))
    #
    # # 更高难度等级题目推荐
    # recom_problem_id2 = []
    # for i in range(rows):
    #     if APARKArray[i, 2] == userRank+1:
    #         recom_problem_id2.append(APARKArray[i, 0])
    # # print('题目是：', recom_problem_id2)
    # # print('等级是：', recom_problem_rank2)
    # if len(recom_problem_id2) >= 10:
    #     slice2 = random.sample(recom_problem_id2, 10)
    #     print('推荐高一个等级：%d等级的题目: %s' % (userRank+1, slice2))
    # else:
    #     print('推荐高一个等级：%d等级的题目: %s' % (userRank+1, recom_problem_id2))


    #从此处开始修改
    # 用户根据提示完成相应的操作
    print('请输入您的Solved题量(-1表示退出系统):', end=' ')
    AC_num = int(input())  # Python3.x 中 input() 函数接受一个标准输入数据，返回为 string 类型。
    if AC_num == -1:
        exit()
    userRank = UserRank(AC_num, ua_centroids)
    while AC_num != -1:
        cnt1_before = AC_num
        print('您目前所处用户等级为%d' % userRank)
        j = 0
        recom_problem_id_level = []
        recom_ac_num_level = []
        recom_problem_rank_level = []
        for i in range(rows):
            if APARKArray[i, 2] == userRank:
                if APARKArray[i, 3] == 1:
                    APARKArray[i, 3] = 0
                    recom_problem_id_level.append(APARKArray[i, 0])
                    recom_ac_num_level.append(APARKArray[i, 1])
                    recom_problem_rank_level.append(APARKArray[i, 2])
                    # print(APARKArray[i, 0], APARKArray[i, 1], APARKArray[i, 2])
                    j += 1
                    if j == 10:
                        break

        if j == 10:
            # flag = 1
            print('同级推荐')
            print('题号:',recom_problem_id_level)
            print('AC数:',recom_ac_num_level)
            print('题目难度等级:',recom_problem_rank_level)
            n1 = np.random.randint(0, 11)  # 返回闭区间 [a, b) 范围内的整数值
            AC_num = AC_num + n1  # 模拟随机答对10道题目中的多少道题目
        else:
            # print('%d等级题目不足10个，只有%d个' % (userRank, j))
            print('题号:', recom_problem_id_level)
            print('AC数:', recom_ac_num_level)
            print('题目难度等级:', recom_problem_rank_level)
            n2 = np.random.randint(0, j + 1)
            AC_num = AC_num + n2
        cnt1_later = AC_num
        cnt1_sub_sum = cnt1_later - cnt1_before
        print('这次做对了%d道题，总共的AC数：%d'%(cnt1_sub_sum,AC_num))


        #高一个级别题目：
        k = 0
        recom_problem_id_high = []
        recom_ac_num_high = []
        recom_problem_rank_high = []
        for i in range(rows):
            if APARKArray[i, 2] == userRank + 1:
                if APARKArray[i, 3] == 1:
                    APARKArray[i, 3] = 0
                    recom_problem_id_high.append(APARKArray[i, 0])
                    recom_ac_num_high.append(APARKArray[i, 1])
                    recom_problem_rank_high.append(APARKArray[i, 2])
                    # print(APARKArray[i, 0], APARKArray[i, 1], APARKArray[i, 2])
                    k += 1
                    if k == 10:
                        break
        cnt2_before = cnt1_later
        if k == 10:
            print('高一级题目推荐')
            print('题号:', recom_problem_id_high)
            print('AC数:', recom_ac_num_high)
            print('题目难度等级:', recom_problem_rank_high)
            n3 = np.random.randint(0, 11)  # 返回闭区间 [a, b) 范围内的整数值
            AC_num = AC_num + n3  # 模拟随机答对10道题目中的多少道题目
        else:
            # print('%d等级题目不足10个，只有%d个' % (userRank+1, k))
            print('题号:', recom_problem_id_high)
            print('AC数:', recom_ac_num_high)
            print('题目难度等级:', recom_problem_rank_high)
            n4 = np.random.randint(0, k + 1)
            AC_num = AC_num + n4
        cnt2_later = AC_num
        cnt2_sub_sum = cnt2_later - cnt2_before
        print('这次做对了%d道题，总共的AC数：%d' % (cnt2_sub_sum, AC_num))
        print('请输入您的Solved题量:(-1表示退出系统)', end=' ')
        AC_num = int(input())  # Python3.x 中 input() 函数接受一个标准输入数据，返回为 string 类型。
        if AC_num == -1:
            exit()
        userRank = UserRank(AC_num, ua_centroids)
