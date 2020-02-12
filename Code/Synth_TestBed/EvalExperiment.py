###Class that runs a semi-synthetic experiment
import numpy
import Policy
import Metrics
import Logger
import Estimators
import sys
import sklearn.cross_validation
import os


class Experiment:
    def __init__(self, dataset, ranking_size):
        self.dataset = dataset
        self.rankingSize = ranking_size
        self.logger = None
        self.metric = None
        self.estimators = None
        print("Experiment:init [INFO] Dataset: %s RankingSize: %d" % (dataset.name, ranking_size), flush = True)
        #Sequence: init() -> setupLogger() -> setupMetric() -> setupEstimators() -> runExperiment()
        #runExperiment() calls trainEstimators if necessary and sets scores
    
    def setupLogger(self, out_dir, det_policy, temperature):
        loggingPolicy = None
        loggingPolicy = Policy.StochasticPolicy(self.dataset, temperature, det_policy)
        
        fileName = out_dir+loggingPolicy.name()+'_gamma'+str(self.rankingSize)+'.pkl'
        if os.path.exists(fileName):
            loggingPolicy.loadGammas(fileName)
        else:
            loggingPolicy.setupExploration(self.rankingSize)
            loggingPolicy.serializeGammas(out_dir, self.rankingSize)
            
        self.logger = Logger.Logger(loggingPolicy, self.rankingSize)
        print("Experiment:setupLogger [INFO] Temperature: %f" %
                temperature, flush = True)
    
    def setupMetric(self, metric_string, dataset):
        currentMetric = None
        if metric_string == 'Constant':
            currentMetric = Metrics.ConstantMetric(dataset, self.rankingSize, 1.0)
        elif metric_string == 'Revenue':
            currentMetric = Metrics.Revenue(dataset, self.rankingSize)
        else:
            print("Experiment:setupMetric [ERR] Metric ", metric_string, "currently not supported.", flush = True)
            sys.exit(0)
                
        self.metric = currentMetric
        print("Experiment:setupMetric [INFO] ", metric_string, flush = True)

    def setupEstimators(self, estimator_list):
        numEstimators = len(estimator_list)
        numFeatures = numpy.shape(self.dataset.features)[2]
        self.estimators = {}
        for i in range(numEstimators):
            estimatorName = estimator_list[i]
            if estimatorName == 'Truth':
                continue

            currentEstimator = None
            tokens = estimatorName.split('_', 2)
            if tokens[0] == 'Direct':
                currentEstimator = Estimators.EstimatorWrapper(Estimators.DirectEstimator(self.logger.policy, tokens[1],
                                            self.rankingSize, numFeatures))
            elif tokens[0] == 'IPS':
                currentEstimator = Estimators.EstimatorWrapper(Estimators.InversePropensityEstimator(self.rankingSize, 
                                            self.logger.policy))
            elif tokens[0] == 'IPS-SN':
                currentEstimator = Estimators.SelfNormalWrapper(Estimators.InversePropensityEstimator(self.rankingSize, 
                                            self.logger.policy))
            elif tokens[0] == 'Gamma':
                currentEstimator = Estimators.EstimatorWrapper(Estimators.GammaEstimator(self.rankingSize, 
                                            self.logger.policy))
            elif tokens[0] == 'Gamma-SN':
                currentEstimator = Estimators.SelfNormalWrapper(Estimators.GammaEstimator(self.rankingSize, 
                                            self.logger.policy))
            elif tokens[0] == 'DoublyRobust':
                ipsEstimator = Estimators.InversePropensityEstimator(self.rankingSize, self.logger.policy)
                directEstimator = Estimators.DirectEstimator(self.logger.policy, tokens[1],
                                            self.rankingSize, numFeatures)
                                            
                currentEstimator = Estimators.EstimatorWrapper(Estimators.DoublyRobustEstimator(ipsEstimator, directEstimator))
            else:
                print("Experiment:setupEstimators [ERR] %s currently not supported." % estimatorName)
                sys.exit(0)
                    
            self.estimators[estimatorName] = currentEstimator
            
        print("Experiment:setupEstimators [INFO] ", self.estimators.keys(), flush = True)
        
    def trainEstimators(self, explored_queries, new_rankings, explored_ranking_features, explored_values):
        folds = sklearn.cross_validation.LabelKFold(explored_queries, n_folds = 5)
        numInstances = numpy.shape(explored_queries)[0]
        for trainEstimator, estimatorWrap in self.estimators.items():
            estimator = estimatorWrap.estimator
            prefix = estimator.name[0:2]
            if prefix == 'DR':
                estimator.directEstimator.train(folds, explored_ranking_features, explored_values)
                estimator.directEstimator.precompute(new_rankings)
                self.estimators[trainEstimator].estimator = estimator
            elif prefix == 'Di':
                tokens = trainEstimator.split('_',2)
                limit = int(tokens[2])
                if limit == numInstances:
                    estimator.train(folds, explored_ranking_features, explored_values)
                    estimator.precompute(new_rankings)
                    self.estimators[trainEstimator].estimator = estimator
                
 
if __name__ == "__main__":
    import Datasets
    import argparse
    import pickle
   
    parser = argparse.ArgumentParser(description='Synthetic Testbed Experiments.')
    parser.add_argument('--max_docs', '-m', metavar='M', type=int, help='Filter documents',
                        default=7)
    parser.add_argument('--length_ranking', '-l', metavar='L', type=int, help='Ranking Size',
                        default=3)
    parser.add_argument('--temperature', '-t', metavar='T', type=float, help='Temperature for logging policy', 
                        default=1.0)
    parser.add_argument('--filtering_ranker', '-f', metavar='F', type=str, help='Model for filtering ranker', 
                        default="lasso")
    parser.add_argument('--evaluation_ranker', '-e', metavar='E', type=str, help='Model for evaluation ranker', 
                        default="lasso")
    parser.add_argument('--value_metric', '-v', metavar='V', type=str, help='Which metric to evaluate',
                        default="Revenue")
    parser.add_argument('--samples', '-s', metavar='S', type=int, help='How many times expt is repeated',
                        default=10)
    parser.add_argument('--numpy_seed', '-n', metavar='N', type=int, 
                        help='Seed for numpy.random', default=387)
    parser.add_argument('--output_dir', '-o', metavar='O', type=str, 
                        help='Directory to store pkls', default='../../Logs/')
            
    anchorTitleURLFeatures = [1,2,3,6,7,8,11,12,13,16,17,18,21,22,23,26,27,28,31,32,33,36,37,38,43,44,46]
    bodyDocFeatures = [0,4,5,9,10,14,15,19,20,24,25,29,30,34,35,39,40,41,42,45,46]
    
    args = parser.parse_args()
    
    numpy.random.seed(args.numpy_seed)
    
    data = Datasets.Datasets()
    data.loadNpz('../../Data/MQ2008.npz', 'MQ2008')
    
    numQueries, numDocs, numFeatures = numpy.shape(data.features)
    
    if args.max_docs < 0:
        args.max_docs = None
      
    filterer = None 
    if args.max_docs is not None:
        filterer = Policy.DeterministicPolicy(data, args.filtering_ranker)
        filterer.train(data, anchorTitleURLFeatures, False)
        data = filterer.createFilteredDataset(args.max_docs)
        filterer.dataset = data
        
    expt = Experiment(data, args.length_ranking)
    
    newPolicy = Policy.DeterministicPolicy(data, args.evaluation_ranker)
    newPolicy.train(data, bodyDocFeatures, False)
        
    #DIAGNOSTIC
    newRankings = newPolicy.predictAll(args.length_ranking)
    if filterer is not None:
        filterRankings = filterer.predictAll(args.length_ranking)
        perfectMatches = 0.0
        matchesPerPosition = numpy.zeros(args.length_ranking, dtype = numpy.float32)
        for i in range(numQueries):
            validDocs = min(args.length_ranking, data.docsPerQuery[i])
            positionMatches = ((filterRankings[i,0:validDocs] - newRankings[i,0:validDocs]) == 0).astype(numpy.float32)
            matchesPerPosition[0:validDocs] += positionMatches
            if positionMatches.sum(dtype = numpy.float32) >= validDocs:
                perfectMatches += 1.0
    
        perfectMatches = 100.0 * perfectMatches / numQueries
        matchesPerPosition = matchesPerPosition / numQueries
        print("Discrepancy between filterer and target: ExactMatches: %0.3f, PerPositionMatches:" % perfectMatches,
                matchesPerPosition, flush = True)
    
    expt.setupLogger(args.output_dir+'precomp_', filterer, args.temperature)
    expt.setupMetric(args.value_metric, data)
   
    BreakPoints = [100, 1000, 10000, 100000]#, 1000000]
    
    approachList = ['Truth','IPS','IPS-SN','Gamma','Gamma-SN','DoublyRobust_ridge']
    for i in BreakPoints:
        if i != BreakPoints[-1]:
            approachList.append('Direct_ridge_'+str(i))
    
    expt.setupEstimators(approachList)
    
    #Brute-force compute truth
    trueMetric = numpy.empty(numQueries, dtype = numpy.float32)
    for i in range(numQueries):
        allowedDocs = expt.dataset.docsPerQuery[i]
        validDocs = min(allowedDocs, args.length_ranking)
        newRanking = newRankings[i,0:validDocs]
        trueMetric[i] = expt.metric.computeMetric(newRanking, i)
    target = trueMetric.mean(dtype = numpy.longdouble)
    print("*** TARGET: ", target, flush = True)
    
    numApproaches = len(approachList)
    numRuns = args.samples
    
    numInstances = BreakPoints[-1]
    saveInstances = BreakPoints[-2]
    saveValues = numpy.linspace(9, numInstances-1,
                            num = saveInstances, endpoint = True, dtype = numpy.int32)
    saveMSEs = numpy.zeros((numApproaches, numRuns, saveInstances), dtype = numpy.float32)
    saveEstimates = numpy.zeros((numApproaches, saveInstances), dtype = numpy.float32)
    
    outputString = args.output_dir+'ssynth_'+args.value_metric+'_'
    if args.max_docs is None:
        outputString += '-1_'
    else:
        outputString += str(args.max_docs)+'_'

    outputString += str(args.length_ranking) +'_'
    outputString += str(int(args.temperature)) + '_' 
    outputString += 'f' + args.filtering_ranker + '_e' + args.evaluation_ranker + '_' + str(args.numpy_seed)
    numpy.random.seed(args.numpy_seed + 100)    
    
    approachDict = {}
    for ind, approach in enumerate(approachList):
        approachDict[approach] = ind

    truthRunningSum = numpy.zeros(numRuns, dtype = numpy.int32)
    runningMeans = numpy.zeros((numApproaches, numRuns), dtype = numpy.float32)
    def updateRunningAverage(run, value):
        truthRunningSum[run] += 1
        approachIndex = approachDict['Truth']
        
        delta = value - runningMeans[approachIndex, run]
        runningMeans[approachIndex, run] += delta / truthRunningSum[run]

    for run in range(numRuns):
        maxTrainInstances = BreakPoints[-2]
        exploredQueries = numpy.empty(maxTrainInstances, dtype = numpy.int32)
        exploredRankingFeatures = numpy.empty((maxTrainInstances, numFeatures * args.length_ranking),
                                        dtype = numpy.float32)
        exploredValues = numpy.empty(maxTrainInstances, dtype = numpy.float32)
        
        saveIndex = 0
        for i in range(len(BreakPoints)):
            if i == 0:
                startInstance = 0
            else:
                startInstance = BreakPoints[i-1]
            endInstance = BreakPoints[i]
            
            if i != 0:
                expt.trainEstimators(exploredQueries[0:startInstance],
                                    newRankings, exploredRankingFeatures[0:startInstance,:],
                                    exploredValues[0:startInstance])
            
            for estimatorName, estimator in expt.estimators.items():
                prefix = estimatorName[0:2]
                if prefix == 'Di':
                    limit = int(estimatorName.split('_',2)[2])
                    if startInstance <= limit:
                        estimator.resetRunningAverage()
                        expt.estimators[estimatorName] = estimator
                
            for j in range(startInstance, endInstance):
                currentQuery, currentRanking, currentValue = \
                        expt.logger.createOneSample(expt.metric)
                numAllowedDocs = expt.dataset.docsPerQuery[currentQuery]
                validDocs = min(args.length_ranking, numAllowedDocs)
                
                newRanking = newRankings[currentQuery,:]
                
                #Maintain these to train direct/semibandit estimators at the next breakpoint
                if i < (len(BreakPoints) - 1):
                    rankedDocFeatures = numpy.zeros((args.length_ranking, numFeatures), dtype = numpy.float32)
                    rankedDocFeatures[0:validDocs, :] = expt.dataset.features[currentQuery,
                                                                        currentRanking[0:validDocs], :]
                    exploredRankingFeatures[j, :] = numpy.reshape(rankedDocFeatures,
                                                            numFeatures * args.length_ranking)
                    exploredValues[j] = currentValue
                    exploredQueries[j] = currentQuery
                
                truth = expt.metric.computeMetric(newRanking, currentQuery)
                updateRunningAverage(run, truth)
                
                for estimatorName, estimator in expt.estimators.items():
                    estimatedValue = estimator.estimate(currentQuery, currentRanking, currentValue, newRanking)
                    approachIndex = approachDict[estimatorName]
                    runningMeans[approachIndex, run] = estimatedValue

                if j == saveValues[saveIndex]:
                    for ind,approach in enumerate(approachList):
                        prediction = runningMeans[ind, run]
                        saveMSEs[ind,run,saveIndex] = (prediction - target)**2
                        if run == 0:
                            saveEstimates[ind, saveIndex] = prediction
                            
                    saveIndex += 1
                    
                if j%1000 == 0:
                    print(".", end = "", flush = True)
                    
        print("")
        print("Iter:%d Truth Estimate=%0.5f" % (run, target), flush = True)
        for ind, approach in enumerate(approachList):
            prediction = runningMeans[ind, run]
            mse = (prediction - target)**2
            print("%s Estimate=%0.3f MSE=%0.3e" % (approach, prediction, mse))
   
        for estimatorName, estimator in expt.estimators.items():
            estimator.resetRunningAverage()
            expt.estimators[estimatorName] = estimator

        #Over-write scores after each run
        outFile = open(outputString+'.pkl', "wb")
        outputData = (run+1, approachDict, saveValues, saveMSEs, saveEstimates, runningMeans, target)
        pickle.dump(outputData, outFile, -1)
        outFile.close()
