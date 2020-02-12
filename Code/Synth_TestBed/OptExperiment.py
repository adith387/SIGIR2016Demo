###Class that runs a semi-synthetic experiment
if __name__ == "__main__":
    import Datasets
    import argparse
    import pickle
    import numpy
    import Policy
    import Estimators
    import sys
    import sklearn.cross_validation
    import os
    import EvalExperiment
    import itertools

   
    parser = argparse.ArgumentParser(description='Synthetic Testbed Experiments.')
    parser.add_argument('--max_docs', '-m', metavar='M', type=int, help='Filter documents',
                        default=7)
    parser.add_argument('--length_ranking', '-l', metavar='L', type=int, help='Ranking Size',
                        default=3)
    parser.add_argument('--temperature', '-t', metavar='T', type=float, help='Temperature for logging policy', 
                        default=10.0)
    parser.add_argument('--filtering_ranker', '-f', metavar='F', type=str, help='Model for filtering ranker', 
                        default="tree")
    parser.add_argument('--value_metric', '-v', metavar='V', type=str, help='Which metric to evaluate',
                        default="Revenue")
    parser.add_argument('--numpy_seed', '-n', metavar='N', type=int, 
                        help='Seed for numpy.random', default=387)
    parser.add_argument('--dump_vw', '-d', metavar='D', type=bool, 
                        help='Materializes a dataset in vw-compatible format', default=False)
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
    else:
        print("[ERROR] Must specify a filterer for optimization expt with max_docs > 0", flush = True)
        sys.exit(0)
        
    expt = EvalExperiment.Experiment(data, args.length_ranking)
    expt.setupLogger(args.output_dir+'precomp_', filterer, args.temperature)
    expt.setupMetric(args.value_metric, data)
   
    filterRankings = filterer.predictAll(args.length_ranking)
    
    #Brute-force compute truth
    trueMetric = numpy.empty(numQueries, dtype = numpy.float32)
    for i in range(numQueries):
        allowedDocs = expt.dataset.docsPerQuery[i]
        validDocs = min(allowedDocs, args.length_ranking)
        newRanking = filterRankings[i,0:validDocs]
        trueMetric[i] = expt.metric.computeMetric(newRanking, i)
    target = trueMetric.mean(dtype = numpy.longdouble)
    print("*** LOGGER: ", target, flush = True)
    
    numInstances = 100000
    
    outputString = args.output_dir+'optsynth_'+args.value_metric+'_'+ str(args.max_docs)+'_'

    outputString += str(args.length_ranking) +'_'
    outputString += str(int(args.temperature)) + '_' 
    outputString += 'f' + args.filtering_ranker + '_' + str(args.numpy_seed)
    
    numpy.random.seed(args.numpy_seed + 100)    
    
    exploredQueries = numpy.empty(numInstances, dtype = numpy.int32)
    exploredRankingFeatures = numpy.empty((numInstances, numFeatures * args.length_ranking),
                                        dtype = numpy.float32)
    exploredValues = numpy.empty(numInstances, dtype = numpy.float32)
    
    relevances = numpy.zeros((numQueries, args.length_ranking, numDocs), dtype = numpy.float32)
    queryHistogram = numpy.zeros(numQueries, dtype = numpy.int)
    inversePropensities = numpy.zeros(numInstances, dtype = numpy.longdouble)
    
    vwFile = None
    if args.dump_vw:
        vwFile = open(args.output_dir+'vw_train.dat', 'w')
        
    for i in range(numInstances):    
        currentQuery, currentRanking, currentValue = \
                        expt.logger.createOneSample(expt.metric)
        queryHistogram[currentQuery] += 1
        
        numAllowedDocs = expt.dataset.docsPerQuery[currentQuery]
        validDocs = min(args.length_ranking, numAllowedDocs)
                
        rankedDocFeatures = numpy.zeros((args.length_ranking, numFeatures), dtype = numpy.float32)
        rankedDocFeatures[0:validDocs, :] = expt.dataset.features[currentQuery,
                                                                        currentRanking[0:validDocs], :]
        exploredRankingFeatures[i, :] = numpy.reshape(rankedDocFeatures,
                                                            numFeatures * args.length_ranking)
        exploredValues[i] = currentValue
        exploredQueries[i] = currentQuery
        
        samplingProbabilities = (expt.logger.policy.documentProbabilities[currentQuery, 0:numAllowedDocs]).copy()
        invPropensity = 1.0
        
        vectorDimension = args.length_ranking * numAllowedDocs
        posRelVector = numpy.zeros(vectorDimension, dtype = numpy.longdouble)
        for j in range(args.length_ranking):
            currAction = currentRanking[j]
            posRelVector[j*numAllowedDocs + currAction] = currentValue
            invPropensity *= (samplingProbabilities.sum(dtype = numpy.longdouble) / samplingProbabilities[currAction])
            samplingProbabilities[currAction] = 0.0
            
        inversePropensities[i] = invPropensity
        
        estimatedPhi = numpy.dot(expt.logger.policy.gammaInverses[currentQuery], posRelVector)
        estimatedPhi = estimatedPhi.reshape((args.length_ranking, numAllowedDocs))
        relevances[currentQuery,:,0:numAllowedDocs] += estimatedPhi
        
        if args.dump_vw and (numInstances - i) <= 1000:
            currentRankingList = currentRanking[0:validDocs].tolist()
        
            allFeatures = expt.dataset.features[currentQuery, 0:numAllowedDocs, :]    
            currId = -1
            for eachRanking in itertools.permutations(range(numAllowedDocs), validDocs):
                currId += 1
                prefixStr = str(currId)
        
                matched = True
                for k in range(args.length_ranking):
                    if currentRankingList[k] != eachRanking[k]:
                        matched = False
                        break
                    
                if matched:
                    prefixStr += ':'+str(1.0 - (currentValue/750.0))+':'+str(1.0 / invPropensity)
                
                prefixStr += ' | '
        
                featString = ''
                for k in range(args.length_ranking):
                    for m in range(numFeatures):
                        if allFeatures[eachRanking[k], m] != 0.0:
                            featString += str(k*numFeatures + m) + ':'+ str(allFeatures[eachRanking[k], m])+' '
                    
                vwFile.write(prefixStr + featString + '\n')
            
            vwFile.write('\n')
        
        if i%1000 == 0:
            print(".", end = "", flush = True)
            
    if vwFile is not None:
        vwFile.close()
    
    banditDataset = Datasets.Datasets()
    banditDataset.features = data.features
    banditDataset.docsPerQuery = data.docsPerQuery
    banditDataset.relevances = None
    with numpy.errstate(divide='ignore', invalid='ignore'):
        currentRelevances = numpy.divide(relevances, queryHistogram[:,None,None])
    currentRelevances[numpy.isnan(currentRelevances)] = 0
    
    currentPolicy = Policy.CustomPolicy(banditDataset, numFeatures, args.length_ranking, 'tree', True)
    currentPolicy.train(banditDataset, currentRelevances)
            
    print("*** SEEN LOGS: ", numpy.mean(exploredValues, dtype = numpy.longdouble), flush = True)
    
    folds = sklearn.cross_validation.LabelKFold(exploredQueries, n_folds = 5)
    estimator = Estimators.DirectEstimator(expt.logger.policy, 'tree',
                                            expt.rankingSize, numFeatures)
                                            
    estimator.train(folds, exploredRankingFeatures, exploredValues)
    
    supervisedPolicy = Policy.DeterministicPolicy(data, 'ridge')
    allFeatures = list(range(47))
    supervisedPolicy.train(data, allFeatures, False)
    
    directRevenues = numpy.zeros(numQueries, dtype = numpy.float32)
    supervisedRevenues = numpy.zeros(numQueries, dtype = numpy.float32)
    slateRevenues = numpy.zeros(numQueries, dtype = numpy.float32)
    
    vwFile = None
    vwFile2 = None
    if args.dump_vw:
        vwFile = open(args.output_dir+'vw_test.dat', 'w')
        vwFile2 = open(args.output_dir+'vw_test2.dat', 'w')
        
    for i in range(numQueries):
        allowedDocs = expt.dataset.docsPerQuery[i]
        validDocs = min(allowedDocs, args.length_ranking)
    
        allFeatures = expt.dataset.features[i, 0:allowedDocs, :]
            
        bestScore = None
        bestRanking = None
        currId = -1
        for eachRanking in itertools.permutations(range(allowedDocs), validDocs):
            currId += 1
            score = estimator.estimate(i, None, None, eachRanking)
            if bestScore is None or score >= bestScore:
                bestScore = score
                bestRanking = eachRanking
            
            if args.dump_vw:
                metricScore = expt.metric.computeMetric(eachRanking, i)
                vwFile2.write(str(currId) + ' ' + str(1.0 - (metricScore/750.0)) + '\n')
                prefixStr = str(currId) + ' | '
                featString = ''
                for k in range(args.length_ranking):
                    for m in range(numFeatures):
                        if allFeatures[eachRanking[k], m] != 0.0:
                            featString += str(k*numFeatures + m) + ':'+ str(allFeatures[eachRanking[k], m]) +' '
                    
                vwFile.write(prefixStr + featString + '\n')
        
        if args.dump_vw:
            vwFile.write('\n')
            vwFile2.write('\n')
                
        directRevenues[i] = expt.metric.computeMetric(bestRanking, i)
        
        supervisedRanking = supervisedPolicy.predict(i, args.length_ranking)    
        supervisedRevenues[i] = expt.metric.computeMetric(supervisedRanking, i)
        
        slateRanking = currentPolicy.predict(i, args.length_ranking)    
        slateRevenues[i] = expt.metric.computeMetric(slateRanking, i)
    
    if vwFile is not None:
        vwFile.close()
    if vwFile2 is not None:
        vwFile2.close()
    
    print("*** DIRECT: ", numpy.mean(directRevenues, dtype = numpy.longdouble), flush = True)
    print("*** SUPERVISED: ", numpy.mean(supervisedRevenues, dtype = numpy.longdouble), flush = True)
    print("*** SLATE: ", numpy.mean(slateRevenues, dtype = numpy.longdouble), flush = True)