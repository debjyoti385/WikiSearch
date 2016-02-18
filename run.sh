/bin/bash

export PYSPARK_PYTHON=/uufs/chpc.utah.edu/sys/installdir/anaconda/2.2.0/2.7.9/bin/python
export PATH=/uufs/chpc.utah.edu/sys/installdir/anaconda/2.2.0/2.7.9/bin/:$PATH

# IF DATA IS IN CHUNK and chunk01 IS IN CURRENT FOLDER
# RUN following to obtain individual xml files 
csplit --digits=8 --quiet --prefix=page- --suffix=%d.xml chunk01 "/<\/page>/+1" "{200000}"

# ASSUMING WE ARE IN THE CURRENT FOLDER WHER WE HAVE DATA 
# IN DATA IN data/*.xml PATH
# RUN

python wiki_extractor_2.py -l -a data/page-00001000.xml

# The above outputs Lemmatized PageContent, pageLinks graph, PageTitle mapping
# Lets copy them to HDFS for further processing

hadoop fs -copyFromLocal *.csv /user/u0992708/wikidataset/
hadoop fs -copyFromLocal *.txt /user/u0992708/wikidataset/

# compute pagerank 
spark-submit --master yarn --num-executors 10 --driver-memory 4g --executor-memory 8g --executor-cores 1 --queue general PageRankCompute.py --mat hdfs://elephant/user/u0992708/wikidataset/transitionMatrix.txt --vec hdfs://elephant/user/u0992708/wikidataset/vector.txt

# CopyPageRank to HDFS
hadoop fs -copyFromLocal pageRank.csv /user/u0992708/wikidataset/


# Find Term-doc relevancy with -r all
# TDM
spark-submit --master yarn --num-executors 10 --driver-memory 6g --executor-memory 8g --executor-cores 1 --queue general term_document_cluster.py -i hdfs://elephant/user/u0992708/wikidataset -n 30000 -r tdm -o ~/final_result_all -w "sample query" -c 30

# Since cluster does not have scipy.. sparsesvd can't run here in cluster
# for LSA and QUERY  change -r option  [ -r lsa ]  or [ -r query ]
# LSA
spark-submit --master yarn --num-executors 10 --driver-memory 6g --executor-memory 8g --executor-cores 1 --queue general term_document_cluster.py -i hdfs://elephant/user/u0992708/wikidataset -n 30000 -r lsa -o ~/final_result_all -w "sample query" -c 30

# QUERY
spark-submit --master yarn --num-executors 10 --driver-memory 6g --executor-memory 8g --executor-cores 1 --queue general term_document_cluster.py -i hdfs://elephant/user/u0992708/wikidataset -n 30000 -r query -o ~/final_result_all/ -w "brazil america" -c 30

