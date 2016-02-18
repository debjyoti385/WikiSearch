#from nltk.corpus import stopwords
from random import randint
import pickle
from pyspark import SparkContext, SparkConf
import os
import math
import numpy as np
import argparse
from collections import Counter
stop = set([u'|', u"=",u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now',  u"a", u"about", u"above", u"after", u"again", u"against", u"all", u"am", u"an", u"and", u"any", u"are", u"aren't", u"as", u"at", u"be", u"because", u"been", u"before", u"being", u"below", u"between", u"both", u"but", u"by", u"can't", u"cannot", u"could", u"couldn't", u"did", u"didn't", u"do", u"does", u"doesn't", u"doing", u"don't", u"down", u"during", u"each", u"few", u"for", u"from", u"further", u"had", u"hadn't", u"has", u"hasn't", u"have", u"haven't", u"having", u"he", u"he'd", u"he'll", u"he's", u"her", u"here", u"here's", u"hers", u"herself", u"him", u"himself", u"his", u"how", u"how's", u"i", u"i'd", u"i'll", u"i'm", u"i've", u"if", u"in", u"into", u"is", u"isn't", u"it", u"it's", u"its", u"itself", u"let's", u"me", u"more", u"most", u"mustn't", u"my", u"myself", u"no", u"nor", u"not", u"of", u"off", u"on", u"once", u"only", u"or", u"other", u"ought", u"our", u"ours    ourselves", u"out", u"over", u"own", u"same", u"shan't", u"she", u"she'd", u"she'll", u"she's", u"should", u"shouldn't", u"so", u"some", u"such", u"than", u"that", u"that's", u"the", u"their", u"theirs", u"them", u"themselves", u"then", u"there", u"there's", u"these", u"they", u"they'd", u"they'll", u"they're", u"they've", u"this", u"those", u"through", u"to", u"too", u"under", u"until", u"up", u"very", u"was", u"wasn't", u"we", u"we'd", u"we'll", u"we're", u"we've", u"were", u"weren't", u"what", u"what's", u"when", u"when's", u"where", u"where's", u"which", u"while", u"who", u"who's", u"whom", u"why", u"why's", u"with", u"won't", u"would", u"wouldn't", u"you", u"you'd", u"you'll", u"you're", u"you've", u"your", u"yours", u"yourself", u"yourselves"])


word2id={}
id2file = {}
outfile =""
directory = ""

def lsa( ):
    from sparsesvd import sparsesvd
    from numpy import array
    import scipy.sparse as sp
    # calculate svd and perform lsa
    print "########     READING TERM DOC MATRIX #########"
    termDocEntries = pickle.load(open(outfile  +"/tdm.p" ,"rb"))
    id2title = pickle.load(open(outfile + "/id_file.p","rb"))
    word2id = pickle.load(open(outfile + "/word_id.p","rb"))
    fileCount = len(id2title)
    #fileCount = 60000
    vocab_size = len(word2id)
    print "########     READING COMPLETE        #########"
    I = array([ i for ((i,j),v) in termDocEntries] )
    J = array([ j for ((i,j),v) in termDocEntries] )
    V = array([ v for ((i,j),v) in termDocEntries] )
    shape = (fileCount, vocab_size)
    print "Dimension of TDM is : ", shape
    print "########     STARTING LSA            #########"
    termDocMatrix = sp.csc_matrix( (V,(I,J)), shape= (fileCount, vocab_size ), dtype=np.float32)

    UT , S, V = sparsesvd(termDocMatrix, 300) 
    (m1,m2) =  UT.T.shape

    S1 = np.zeros((m2,m2), dtype=np.float32)
    for i in range(m2):
        S1[i][i] = S[i]
    US = np.dot(UT.T, S1)
    print m1, m2
    (n1,n2) = V.shape

    pickle.dump( US , open( outfile + "/u_sigma.p", "wb" ) )
    pickle.dump( V.T , open( outfile + "/v.p", "wb" ) )
    print "########     LSA COMPLETE        #########"

def run_query( queryWords, count = 50):
    # Start Query Part 
    US = pickle.load(open(outfile  +"/u_sigma.p" ,"rb"))
    VT = pickle.load(open(outfile  +"/v.p" ,"rb"))
    word2id = pickle.load(open(outfile + "/word_id.p","rb"))
    id2file = pickle.load(open(outfile + "/id_file.p","rb"))
    id2file = {int(k):v for k,v in id2file.iteritems() }
    #queryWord = raw_input('Enter your search query [-1 to stop]: ')
    print "Query word :" + queryWords
#    print VT[0].shape
#    print VT.shape
#    print US.shape
    indexList = []
    for queryWord in queryWords.split():
        queryWord = unicode(queryWord.lower(), "utf-8")
        index = word2id.get(queryWord, -1 )
        if index == -1:
            print "word not found ", queryWord
        else:
            indexList.append(index)
            print "index of word found " + str(index)
    print indexList
    if len(indexList) < 1:
        print "SORRY!! NO RELEVANT WORD FOUND "
        return 
    V= VT[indexList,:].T
    resultMatrix = np.dot(US, V)
    resultVector = resultMatrix.sum(axis=1)
    #print resultMatrix
    id2rank = sc.textFile(directory + "/pageRank.csv").map( lambda x : ( int(x.split(" ")[1]), int(x.split(" ")[0])) )
    relevantDocs =  sc.parallelize(resultVector).zipWithIndex().sortBy(lambda x : x[0], False).take(count)
    resultRDD = sc.parallelize(relevantDocs).map( lambda (x,y): (y,x))

    result = resultRDD.join(id2rank).sortBy(lambda x: x[1][1]).collect()


    
    #print V.T[2]
    #print n1, n2
    print 
    print "###################################################################################################################"
    print " Query : ", queryWords.upper()
    print "###################################################################################################################"
    print "|\t","%-10s"% "PAGE RANK", "\t|\t%-50s\t" % unicode("      TITLE    ").encode("utf-8"),"|\t%-12s\t" % " RELEVANCE INDEX"
    print "###################################################################################################################"
    for i in result:
        print "|\t","%10i"% i[1][1], "\t|\t%-50s\t" % unicode(id2file[i[0]]).encode("utf-8"),"|\t%12f\t" % i[1][0]
    #print US    
    print "###################################################################################################################"
    return 

def countFrequency( data ):
    fileid = data[0]
    #fileid = fileID.value[filename]
    #stop = stopwords.words('english')
    words = [word.lower() for word in data[1].split() if word.lower() not in stop and len(word)> 2]
    counter  = Counter(words)
    result = [((fileid,k), v) for k, v in counter.iteritems() if v > 2] 
    return result


def splitLine(word):
    elements= word.split("\t",1)
    if len(elements) <2 or elements[1] == None:
	return (elements[0], "  garbage value ")
    return (elements[0],elements[1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Term Document Similarity',
                                     epilog='Term-document Similarity.. with TF-IDF and SVD \n Example: \n spark-submit --master yarn --num-executors 10 --driver-memory 6g --executor-memory 8g --executor-cores 1 --queue general term_document_cluster.py -i hdfs://elephant/user/u0992708/wikidateset -n 40000 -r query -o ~/output -w "sample query" -c 30', add_help='How to use',
                                     prog='spark-submit term_document.py  -i <inputDirectory> -o <outputDirectory> -n <vocab_size> -r < all | lsa | query > -w <word> [  -m <executor_memory in GB> -p <parallelism>]')
    parser.add_argument("-i", "--dir", required=True,
                        help="Input Directory where pageContent.csv pageRank.csv and pageID.csv is present")
    parser.add_argument("-o", "--outfile", type=str, required=True,
                    help=" Output Directory where it stores all the meta information and later used by query ")
    parser.add_argument("-n", "--size",type=int, default=30000,
                        help=" Specify your Dictoionary/Vocab size")

    parser.add_argument("-r", "--run", type=str, default="query",
                    help="Specify which section of program to run <tdm | t | lsa | l | query | q > \n\t\t Step 1:  TDM: Constructs Term-Document Matrix \n\t\t Step 2: LSA: Construct SVD Components for LSA \t\n\n Step 3: Query")

    parser.add_argument("-w", "--word", type=str, required=True,
                    help=" Enter your search string ")

    parser.add_argument("-c", "--count", type=int, default=25,
            help=" Specify number of documents to show as output.. default: 25   ")

    # Optional parameters.
    parser.add_argument("-m", "--memory", type=int, default=4,
                        help="Spark Executor Memory in GB .. default: 4GB")

    parser.add_argument("-p", "--proc", type=int, default=4,
                    help="number of CPU cores .. default: 4")
    args = vars(parser.parse_args())

    #conf = SparkConf()
    #conf.setMaster("local["+ str(args['proc']) + "]")
    #conf.setAppName("TermDocument")
    #conf.set("spark.executor.memory", str(args['memory']) + "g")
    #sc = SparkContext(conf=conf)
    sc = SparkContext()

    vocab_size = int( args['size'])
    directory = args['dir'] 
    outfile = args['outfile']
    isBuild = args['run'].lower()
    queryWord = args['word']
    count = args['count']
    if isBuild == "all" or isBuild == "a" or isBuild== "tdm" or isBuild=="t":
        fileRDD =  sc.textFile(directory + "/pageContent.csv").map(splitLine) 
        fileCount = fileRDD.count()
        #fileID = sc.broadcast( { k: v for (k,v) in fileRDD.keys().zipWithIndex().collect()})
        #id2file =  { v: k  for (k,v) in fileRDD.keys().zipWithIndex().collect()}
        frequencyRDD = fileRDD.flatMap( countFrequency )
        
        topWords = frequencyRDD.map(lambda ((f,w),v): (w,v)).reduceByKey(lambda x,y: x+y).sortBy(lambda x: x[1], False ).map(lambda x: x[0]).take( vocab_size )
        topWordsRDD = sc.broadcast(topWords)
        
        termDocumentFreq = frequencyRDD.filter(lambda ((f,w),v):  w in topWordsRDD.value )
        termDocumentFreq.cache()

        idfRDD = termDocumentFreq.map(lambda ((f,w),v): (w, 1) ).reduceByKey(lambda x,y: x+ y ).map( lambda (word,count) : (word, math.log(fileCount/ count)) )
        idfRDD.cache()
        bidfs = sc.broadcast({ k:v for (k,v) in idfRDD.collect() })
        word2id = { k:v for (k,v) in  idfRDD.map(lambda (w,v): w ).zipWithIndex().collect() }
        btermIds = sc.broadcast(  word2id  )

        fileMaxWordCountList = termDocumentFreq.map(lambda ((f,w),v): (f, (v,w)) ).reduceByKey(lambda x,y: x if x[0] > y[0] else y ).collect()
        fileMaxWordCount = sc.broadcast( { element[0]:element[1][0] for element in fileMaxWordCountList })

        tfRDD = termDocumentFreq.map( lambda ((f,w),v): ((f,w), float(0.5) + float(v)/ fileMaxWordCount.value[f]  )  )
        termDocEntries  = tfRDD.map( lambda ((f,w),v): ((f,  btermIds.value[w] ), v * bidfs.value[w] )).collect()


        if not os.path.exists(outfile):
            os.makedirs(outfile)
	id2titleRDD = sc.textFile(directory + "/pageID.csv").map(lambda x : (x.split(" ",1)[0],x.split(" ",1)[1] ))
	id2title = { k:v for (k,v) in id2titleRDD.collect() }
        pickle.dump( termDocEntries , open( outfile + "/tdm.p", "wb" ) )
        pickle.dump( id2title , open( outfile + "/id_file.p", "wb" ) )
        pickle.dump( word2id , open( outfile + "/word_id.p", "wb" ) )
        print "#########                  TERM DOCUMENT MATRIX CREATED                                    ############"
        
        #lsa()
        #run_query(queryWord,count)

    elif isBuild == "lsa" or isBuild=="l":
        lsa()
        #run_query( queryWord , count)
    elif isBuild == "query" or isBuild == "q":
        run_query( queryWord , count)
    else:
        print "Please use option -r < all | lsa | query  > or < a | l | q > "
