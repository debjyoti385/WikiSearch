
# Wikipedia Search

### Contributors: 
- Zinnia Mukherjee
- Debjyoti Paul

## Modules or Structure:
###   OFFLINE PART:
####   a) Parsing of Wikipedia Data
- For our initial dataset, we have a list of xml files, each file corresponding to a wikipedia page.
- We have used the [Wikiextractor](https://github.com/bwbaugh/wikipedia-extractor.git) module and [BeautifulSoup](http://www.crummy.com/software/BeautifulSoup/bs4/doc/) library of python to parse the dump of wikipedia xml files and extract links and page content.
- On the extracted page content, we have filtered out the stop words whose set is defined using the [nltk corpus](http://www.nltk.org/api/nltk.corpus.html) for "english" language.
- After removing stop words, we have applied a lemmatization of word token using the [nltk](http://www.nltk.org/api/nltk.stem.html) WordNetLemmatizer library.
- Finally this module produces the following list of output files
	- **pageID.csv** : Each line of this file contains a unique file ID and  the corresponding page title of it.
	- **pageLinks.csv** : Each line of this file contains an entry for a page. The first element denotes the page ID and the successive elements denote the pageIDs of the outgoing links from this page.
	- **pageContent.csv** : Each line of this file contains the pageID and the corresponding page content.
	- **transitionMatrix.txt** : The file is basically the transition matrix required for computing page rank. Each line contains the row number(outgoing page ID), col number (current page ID) and the probability of going from the current page to the outgoing page.
	- **vector.txt** : This file contains the file ID(the row number in the vector) and the probability of being in the file, which is initially 1/number of pages.
	
####   b) Page Rank Computation
- The page rank algorithm has been followed from [this book.](http://infolab.stanford.edu/~ullman/mmds/ch5.pdf)
- For the matrix vector multiplication of the transition matrix and the vector, we have implemented the block matrix vector operation.
- In order to avoid dead ends and spider traps, we have applied taxation at each step. At each step, a random surfer has a probability of beta to follow an out link from that page and a probability of 1-beta to transport to transport to any other random page.
- The matrix vector operation is carried until convergence. Convergence happens when the average of the absolute difference of the values at each position of the two vectors is less than `1*10^-6`. We carry this repetition for `50` iterations at most. 
- The final vector is the page rank vector. The following list shows the top `10` ranked pages.

```
RESULT

Page Rank 	Page ID 	Page Title
    0		56730 		"World War II" 
    1		6437 		Latin 
    2		44850 		Julian calendar
    4		32954 		"United Kingdom"
    5		47722 		"Roman numerals"
    6		19485 		"Soviet Union"
    7		38111 		Europe 
    8		20778 		Germany 
    9		23339 		Italy 
    10		58905 		"Greek language"
```
		
####   c) Term-Document Relevance 
- Since we are dealing with query instead of finding pairwise doc-doc relevance we found term-document relevance with Latent Semantic Analysis will be a realistic approach. 
- We implemented `Term-Document Matrix (TDM)` with `TF-IDF` values in the following manner.
	- We have used same formula given in [wikipedia tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
	- Calculated term frequency for each word in each document
	- We have considered tens of thousands of most frequent words (used 30,000 words for demo purpose)
	- Calculated `tf` based on the formula given
	- Calculated `idf` for each word in the corpus
	- Calculated `tf * idf`
	- Now `tf-idf` TDM is ready
- Having TDM in hand our next approach is to perform `Latent Semantic Analysis` ([LSA](https://en.wikipedia.org/wiki/Latent_semantic_analysis)) 
- For LSA we have to perform `Singular Value Decomposition` ([SVD](https://en.wikipedia.org/wiki/SVD)) to yield three basic components for analysis
- Since our TDM Matrix is not so dense it is good to use `Sparse SVD` technique where `SVD` is performed on sparse matrix. We have used [sparsesvd](https://pypi.python.org/pypi/sparsesvd/)  to find the following components :
	-    **`U`** : `Document-Feature` Matrix 
	-    **`S`** : `Feature-Feature` Diagonal Matrix
	-    **`V`** : `Feature-Word` Matrix
- This module stores `U, S` and `V` in pickle format in the output directory which is used by Online Query Module. 


###   ONLINE PART:
####   Query Module: 
###### Given a query it finds specified number of documents relevant to that query.
- Reads `U, S` and `V` from output directory
- Read the page rank computed by the Page Rank module 
- Finds all rows from `V` corresponding to each word from given query and creates `V'`
- Calculates `U*S*V'` and sort on the sum of each row which is basically the relevance score of each document for the given query.
- Consider first few documents (default 25) and sort based on page rank.
- Output the result

## Usage help
### Page Rank Computation
#### ([Suggested reading](infolab.stanford.edu/~ullman/mmds/ch5.pdf)) 
```
usage: 
spark-submit PageRankCompute.py --mat <transitionMatrix> --vec <documenteVector> [ -e <epsilon value> -b <beta value>-s <group_size> -m <executor_memory in GB> -p <parallelism>]

[-h] --mat MAT --vec VEC [-s SIZE] [-e EPSILON] [-b BETA] [-m MEMORY] [-p PROC] 

Page Rank Calculator

optional arguments:
  -h, --help            show this help message and exit
  --mat MAT    			Input Transition Matrix file.
  --vec VEC 		    Input Vector file.
  -s SIZE, --size SIZE  Each Group size.. default 200
  -e EPSILON, --epsilon EPSILON
                        epsilon value .. default: 0.000001
  -b BETA, --beta BETA  beta value .. default: 0.85
  -m MEMORY, --memory MEMORY
                        Spark Executor Memory in GB .. default: 4GB
  -p PROC, --proc PROC  number of CPU cores .. default: 4
```
Given a transition Matrix and Document vector it computes page rank of each documents



### Term-document Similarity with [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) for [LSA](https://en.wikipedia.org/wiki/Latent_semantic_analysis)

``` 
usage: 
spark-submit term_document.py  -i <inputDirectory> -o <outputDirectory> -n <vocab_size> -r < all | lsa | query > -w <word> [  -m <executor_memory in GB> -p <parallelism>] 

 [-h] -i DIR -o OUTFILE [-n SIZE] [-r RUN] -w WORD [-c COUNT] [-m MEMORY] [-p PROC] 

optional arguments:
  -h, --help            show this help message and exit
  -i DIR, --dir DIR     Input Directory where pageContent.csv pageRank.csv and pageID.csv  is present
  -o OUTFILE, --outfile OUTFILE
                        Output Directory where it stores all the meta information and later used by query
  -n SIZE, --size SIZE  Specify your Dictoionary/Vocab size
  -r RUN, --run RUN     Specify which section of program to run <tdm | t | lsa | l | query | q > 
                        Step 1: TDM: Constructs Term-Document Matrix 
                        Step 2: LSA: Construct SVD Components for LSA 
                        Step 3: Query
  -w WORD, --word WORD  Enter your search string
  -c COUNT, --count COUNT
                        Specify number of documents to show as output.. default: 25
  -m MEMORY, --memory MEMORY
                        Spark Executor Memory in GB .. default: 4GB
  -p PROC, --proc PROC  number of CPU cores .. default: 4
```

Example: 
```
spark-submit --master yarn --num-executors 10 --driver-memory 6g --executor-memory 8g --executor-cores 1 --queue general term_document_cluster.py -i hdfs://elephant/user/u0992708/wikidateset -n 40000 -r query -o ~/output -w "sample query" -c 30
```





### Few Queries:

- Command:
```
$ spark-submit --master yarn --num-executors 10 --driver-memory 6g --executor-memory 8g --executor-cores 1 --queue general term_document_cluster.py -i hdfs://elephant/user/u0992708/wikidataset -n 30000 -r q -o ~/final_result_all -w "utah" -c 10
```
- Result:
```
###################################################################################################################
 Query :  UTAH
###################################################################################################################
|	PAGE RANK  	|	      TITLE                                       	|	 RELEVANCE INDEX	
###################################################################################################################
|	     13559 	|	"History of The Church of Jesus Christ of Latter-day Saints"	|	    1.266137	
|	     22528 	|	"First Transcontinental Railroad"                 	|	    2.521217	
|	     25823 	|	"Missouri River"                                  	|	    1.434554	
|	     28109 	|	"Salt Lake City"                                  	|	    1.296667	
|	     28149 	|	Montana                                           	|	    1.555194	
|	     28181 	|	Utah                                              	|	    2.049825	
|	     37233 	|	Colorado                                          	|	    1.680753	
|	     37271 	|	"Oregon Trail"                                    	|	    3.115999	
|	     53561 	|	"Dallas Mavericks"                                	|	    1.361094	
|	     55546 	|	"Utah Jazz"                                       	|	    1.318745	
###################################################################################################################
```
- Command:
```
spark-submit --master yarn --num-executors 10 --driver-memory 6g --executor-memory 8g --executor-cores 1 --queue general term_document_cluster.py -i hdfs://elephant/user/u0992708/wikidataset -n 30000 -r q -o ~/final_result_all -w "solar system" -c 10
```
- Result:
```
###################################################################################################################
 Query :  SOLAR SYSTEM
###################################################################################################################
|	PAGE RANK  	|	      TITLE                                       	|	 RELEVANCE INDEX	
###################################################################################################################
|	      3384 	|	Astronomy                                         	|	    5.595503	
|	     10719 	|	Jupiter                                           	|	    6.218736	
|	     11883 	|	"Operating system"                                	|	    5.558648	
|	     17118 	|	"Interplanetary spaceflight"                      	|	    5.750967	
|	     26127 	|	Exoplanet                                         	|	    6.554704	
|	     29156 	|	"Galileo (spacecraft)"                            	|	    6.049845	
|	     35281 	|	Planet                                            	|	    6.417857	
|	     45236 	|	"Solar System"                                    	|	    6.610066	
|	     45325 	|	"Space colonization"                              	|	    5.645838	
|	     55147 	|	Sun                                               	|	    5.610186	
###################################################################################################################
```




