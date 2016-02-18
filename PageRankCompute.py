
from pyspark import SparkContext, SparkConf
import math
import numpy as np
import argparse
import csv



def matrixMapper(data):
    row = int(data[0])
    col = int(data[1])
    val = float(data[2])
    colGroup = int(math.ceil((col+1)/float(GRP_SIZE)))
    rowGroup = int(math.ceil((row+1)/float(GRP_SIZE)))
    return ((rowGroup, colGroup), (row, col, val))

def vectorGroupMapper(data):
    val = float(data[1])
    group = int(math.ceil((int(data[0])+1)/float(GRP_SIZE)))
    vecList = []
    for i in range(1, numGrpA + 1):
        vecList.append(((i, group), (int(data[0]),float(val))))
    return vecList

def test(data):
    print 'key:' + str(data[0]) + ' value1: '+ str(list(data[1][0])) + ' value2: '+ str(data[1][1])
    return data


def multCombine(data):
    row_id = data[0][0]
    col_id = data[0][1]
    matrixValues = list(data[1][0])
    vecValues = list(data[1][1])
    matrix = np.zeros((GRP_SIZE, GRP_SIZE))
    for each_value in matrixValues:
        row = int(each_value[0]  - (row_id - 1)*GRP_SIZE)
        col = int(each_value[1] - (col_id -1)*GRP_SIZE)
        matrix[row][col] = each_value[2]
    vector = np.zeros(GRP_SIZE)
    for each in vecValues:
        index = int(each[0]-(col_id-1)*GRP_SIZE)
        vector[index] = float(each[1])
    prod = np.dot(matrix, vector)
    rowProd = []
    rows = (prod.shape)[0]
    for i in range(0, rows):
        row_no = i + rows*(row_id - 1)
        rowProd.append((row_no, prod[i]))
    return rowProd




def computeDifference(oldVector, newVector):
    joinedRDD = oldVector.union(newVector).reduceByKey(lambda x, y: abs(x-y)).map(lambda x: x[1]).reduce(lambda x, y: x+y)
    return joinedRDD/numRows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Page Rank Calculator',
                                     epilog='Given a transition Matrix and Document vector it computes page rank of each documents', add_help='How to use',
                                     prog='./bin/pyspark PageRankCompute.py -i <transitionMatrix> -v <documentVector> [ -e <epsilon value> -b <beta value>-s <group_size> -m <executor_memory in GB> -p <parallelism>]')
    parser.add_argument("--mat", type=str, required=True,
                        help="Input Transition Matrix file.")
    parser.add_argument("--vec", type=str, required=True,
                        help="Input Vector file.")

    # Optional parameters.
    parser.add_argument("-s", "--size", type=int, default=200,
                        help="Each Group size.. default 200")
    parser.add_argument("-e", "--epsilon", type=float, default=0.000001,
                                            help="epsilon value .. default: 0.000001")
    parser.add_argument("-b", "--beta", type=float, default=0.85,
                                            help="beta value .. default: 0.85")
    parser.add_argument("-m", "--memory", type=int, default=4,
                        help="Spark Executor Memory in GB .. default: 4GB")
    parser.add_argument("-p", "--proc", type=int, default=4,
                                            help="number of CPU cores .. default: 4")



    args = vars(parser.parse_args())

    GRP_SIZE = int(args['size'])

    #ABS_DIFF = 0.000001
    #BETA = 0.85
    ABS_DIFF = args['epsilon']
    BETA = args['beta']

    conf = SparkConf()
    conf.setMaster("local["+ str(args['proc']) + "]")
    conf.setAppName("MatVec")
    conf.set("spark.executor.memory", str(args['memory']) + "g")
    sc = SparkContext(conf=conf)


    matFilePath = args['mat']
    vecFilePath = args['vec']

    #calculating no of rows
    vectorData = sc.textFile(vecFilePath).map(lambda s: s.split(" ")).map(lambda x: (int(x[0]), float(x[1]))).cache()
    taxVector = vectorData.map(lambda x: (x[0], (1-BETA) * x[1]))
    matrixData = sc.textFile(matFilePath).map(lambda s: s.split(" ")).cache()
    numRows = vectorData.map(lambda s: int(s[0])).reduce(max) + 1
    numGrpA = numRows/GRP_SIZE
    matrixMap = matrixData.map(matrixMapper).groupByKey()

    for i in range(0, 50):
        vectorMap = vectorData.flatMap(vectorGroupMapper).groupByKey()
        joinedRDD = matrixMap.join(vectorMap)
        vectorProd = joinedRDD.flatMap(multCombine).reduceByKey(lambda x, y: x+y)\
            .map(lambda x: (x[0], BETA*x[1]))
        #print vectorProd.collect()
        vectorProd = vectorProd.union(taxVector).reduceByKey(lambda x, y: x+y)
        diff = computeDifference(vectorData, vectorProd)
        if diff <= ABS_DIFF:
            break
        vectorData = vectorProd


    output = vectorProd.sortBy(lambda x: x[1], False).zipWithIndex().map(lambda x: (x[1], x[0][0], x[0][1])).collect()
    #print output

    with open('pageRank.csv', 'w') as csvfile:
        for each_entry in output:
            linewriter = csv.writer(csvfile, delimiter=' ')
            linewriter.writerow([each_entry[0], each_entry[1], each_entry[2]])







