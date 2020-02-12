###Class that generates logs from an exploration policy
import numpy


class Logger:
    #policy: (StochasticPolicy) Must have called StochasticPolicy.setupExploration before passing to Logger
    #ranking_size: (int) Size of slate
    def __init__(self, policy, ranking_size):
        self.dataset = policy.dataset
        self.policy = policy
        self.rankingSize = ranking_size
        print("Logger:init [INFO] Policy: %s RankingSize: %d" % 
                    (self.policy.name(), ranking_size), flush = True)
    
    #metric: Metrics::Metric that will be logged
    def createOneSample(self, metric):
        numQueries, numDocs, numFeatures = numpy.shape(self.dataset.features)    
        query = numpy.random.randint(0, numQueries)
        allowedDocs = self.dataset.docsPerQuery[query]
        validDocs = min(self.rankingSize, allowedDocs)
        
        ranking = self.policy.predict(query, self.rankingSize)
            
        value = metric.computeMetric(ranking, query)
            
        return query, ranking, value
        
    #log_size: (int) Required number of events from logging policy
    #metric: Metric that will be logged
    def createLog(self, log_size, metric):
        numQueries, numDocs, numFeatures = numpy.shape(self.dataset.features)
        queryStream = numpy.zeros(log_size, dtype = numpy.int32)
        rankingStream = -numpy.ones((self.rankingSize, log_size), dtype = numpy.int32)
        
        metricValues = numpy.zeros(log_size, dtype = numpy.float32)
        
        for i in range(log_size):
            currentQuery, ranking, values = self.createOneSample(metric)
            queryStream[i] = currentQuery
            rankingStream[:, i] = ranking
            metricValues[i] = values
                
            if i%1000 == 0:
                print(".", end = "", flush = True)
        
        print("")
        print("Logger:createLog [INFO] LogSize: %d Metric: %s" %
                (log_size, metric.name), flush = True)
        return queryStream, rankingStream, metricValues
        
        
if __name__ == "__main__":
    import Datasets
    import Metrics
    import Policy        

    mq2008Data = Datasets.Datasets()
    mq2008Data.loadNpz('../../Data/MQ2008.npz', 'MQ2008')
    
    detLogger = Policy.DeterministicPolicy(mq2008Data, 'lasso')
    detLogger.train(mq2008Data, range(47), False)
    
    newData = detLogger.createFilteredDataset(5)
    
    rankingSize = 8
    revenue = Metrics.Revenue(newData, rankingSize)
    
    uniformPolicy = Policy.StochasticPolicy(newData, 0, detLogger)
    uniformPolicy.setupExploration(rankingSize)
    logger = Logger(uniformPolicy, rankingSize)
        
    print("One sample: Query, Ranking, MetricValue")
    print(logger.createOneSample(revenue), flush = True)
    
    queries, rankings, metricValues = logger.createLog(10000, revenue)
    histogram = numpy.bincount(queries)
    print("Histogram of seen queries", histogram, flush = True)
    print("Num unique queries", numpy.sum(histogram > 0), flush = True)
    print (revenue.name, metricValues.mean(dtype = numpy.longdouble), flush = True)
