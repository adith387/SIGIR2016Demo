###Classes that define different metrics for semi-synthetic experiments
import numpy


class Metric:
    #ranking_size: (int) Maximum size of slate across contexts, l
    def __init__(self, ranking_size):
        self.rankingSize = ranking_size
        self.name = None
        ###All sub-classes of Metric should supply a computeMetric method
        ###Requires: ranking_list of size ranking_size, query_id
        ###Returns: float, indicating value.


class ConstantMetric(Metric):
    def __init__(self, dataset, ranking_size, constant = 1.0):
        Metric.__init__(self, ranking_size)
        self.constant = constant
        self.name = 'Constant'
        print("ConstantMetric:init [INFO] RankingSize: %d Constant: %f" % 
                (ranking_size, constant), flush = True)
    
    #ranking_list ([int],length=ranking_size): Document ids in
    #each slot of the slate.
    #query_id: (int) Index of the query (unused)
    def computeMetric(self, ranking_list, query_id):
        return self.constant


class Revenue(Metric):
    def __init__(self, dataset, ranking_size):
        Metric.__init__(self, ranking_size)
        self.name = 'Revenue'
        self.dataset = dataset
        self.anchorFeatures = [1,6,11,16,21,26,31,36]
        self.titleFeatures = [2,7,12,17,22,27,32,37]
        self.urlFeatures = [3,8,13,18,23,28,33,38,43,44]
        self.bodyFeatures = [0,5,10,15,20,25,30,35,40,41,42,45]
        self.docFeatures = [4,9,14,19,24,29,34,39]
        
        print("Revenue:init [INFO] RankingSize:", ranking_size, flush = True)
        
    def computeMetric(self, ranking_list, query_id):
        numAllowedDocs = self.dataset.docsPerQuery[query_id]
        validDocs = min(numAllowedDocs, self.rankingSize)
        rankedDocFeatures = self.dataset.features[query_id, ranking_list[0:validDocs], :]
        scaledFeature = numpy.amax(rankedDocFeatures, axis = 0)
        
        relevanceList = self.dataset.relevances[query_id, ranking_list[0:validDocs]] + 1
        relevanceList[0] = relevanceList[0]*5
        relevanceList[1] = relevanceList[1]*2
        
        positionScore = relevanceList.sum(dtype = numpy.float32)
        
        anchor = numpy.median(scaledFeature[self.anchorFeatures])
        title = numpy.median(scaledFeature[self.titleFeatures])
        url = numpy.median(scaledFeature[self.urlFeatures])
        body = numpy.median(scaledFeature[self.bodyFeatures])
        doc = numpy.median(scaledFeature[self.docFeatures])
        
        setScore = 0.0
        if relevanceList[0] > 10:
            setScore = title + 2 * url * anchor + body + 3 * doc * doc
        elif relevanceList[1] == 2 * relevanceList[2]:
            setScore = title + 2 * url + anchor
        else:
            setScore = 2 * anchor + 2 * title + url
        
        score = positionScore * (setScore + positionScore)
        return score
 
if __name__ == "__main__":
    import Datasets
    
    mq2008Data = Datasets.Datasets()
    mq2008Data.loadNpz('../../Data/MQ2008.npz', 'MQ2008')
    
    rankingSize = 5
    numQueries = numpy.shape(mq2008Data.features)[0]
    queryID = numpy.random.randint(0, numQueries)
    print("Selected queryID:", queryID)
    
    numAllowedDocs = mq2008Data.docsPerQuery[queryID]
    validDocs = min(numAllowedDocs, rankingSize)
    
    ranking = numpy.random.choice(numAllowedDocs, size = validDocs, replace = False)
    
    constant = ConstantMetric(mq2008Data, rankingSize)
    print("Constant", constant.computeMetric(ranking, queryID))
    
    revenue = Revenue(mq2008Data, rankingSize)
    print("Revenue", revenue.computeMetric(ranking, queryID))
    