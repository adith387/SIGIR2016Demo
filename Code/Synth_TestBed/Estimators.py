###Classes that define different off policy estimators for semi-synthetic experiments
import numpy
import sklearn.linear_model
import sklearn.tree
import sklearn.grid_search
import scipy.linalg
import sys
import pickle


class Estimator:
    #ranking_size: (int) Size of slate, l
    def __init__(self, ranking_size, policy):
        self.rankingSize = ranking_size
        self.name = None
        self.policy = policy
        ###All sub-classes of Estimator should supply a estimate method
        ###Requires: query, explored_ranking, explored_clicks, explored_value,
        ###             new_ranking, new_ranking_features
        ###Returns: float indicating estimated value

    def stringName(self):
        return self.name+'_'+self.policy.name()
    
    
class TrainedEstimator(Estimator):
    #num_params: (int) Number of parameters maintained by this TrainedEstimator
    def __init__(self, ranking_size, policy, num_params):
        Estimator.__init__(self, ranking_size, policy)
        self.params = None
        self.tree = None
        self.numParams = num_params
        
        self.hyperParams = numpy.logspace(-3,2,num=6,base=5)
        self.treeDepths = {'max_depth': list(range(1,8,2))}
        
        ###All sub-classes of TrainedEstimator should supply a train method
        ###Requires: explored_queries, explored_ranking_features, 
        ###                 explored_click_features, explored_values
        ###Returns: Nothing
        ###Sets: self.params
    
    def computeScore(self, new_features):
        if self.params is None:
            return 0.0
        return numpy.dot(self.params, new_features)
        
        
class DirectEstimator(TrainedEstimator):
    ###Learns a linear predictor using offline <query, doc> features to predict reward
    
    #num_features: (int) Number of <query, doc> similarity features
    def __init__(self, policy, estimator_type, ranking_size, num_features):
        TrainedEstimator.__init__(self, ranking_size, policy, num_features * ranking_size)
        self.name = 'Direct_'+estimator_type
        self.estimatorType = estimator_type
        self.precomputed = None
        
    def saveModel(self, file_path, num_instances):
        outputFileName = file_path+self.stringName()+'_l'+str(self.rankingSize)+'_'+str(num_instances)+'.pkl'
        
        outputData = None
        if self.estimatorType == 'tree':
            outputData = (self.precomputed, self.tree)
        else:
            outputData = (self.precomputed, self.params)
            
        outFile = open(outputFileName, 'wb')
        pickle.dump(outputData, outFile, -1)
        outFile.close()
        print("DirectEstimator:saveModel [INFO] Saved Model after %d samples at" % num_instances,
                outputFileName, flush = True)
    
    def loadModel(self, file_name):
        inFile = open(file_name, 'rb')
        (self.precomputed, model) = pickle.load(inFile)
        inFile.close()
        if self.estimatorType == 'tree':
            self.tree = model
        else:
            self.params = model
    
        print("DirectEstimator:loadModel [INFO] Loaded trained model for direct estimator from ",
                file_name, flush = True)
                
    def train(self, folds, explored_ranking_features, explored_values, inverse_propensities = None):
        ###DirectEstimator does not need explored_clicklogs, explored_dwelllogs
        ###Ensure that no -1's (invalid documents) are passed in explored_ranking_features 
        if self.estimatorType == 'tree':
            treeCV = sklearn.grid_search.GridSearchCV(sklearn.tree.DecisionTreeRegressor(),
                            param_grid=self.treeDepths,
                            scoring=None, fit_params={'sample_weight':inverse_propensities}, n_jobs=1, pre_dispatch=1,
                            iid=False, cv=folds, refit=True, verbose=0, error_score='raise')
            treeCV.fit(explored_ranking_features, explored_values)
            self.tree = treeCV.best_estimator_
            print("DirectEstimator:train [INFO] %s Best depth: %d" %
                (self.estimatorType, treeCV.best_params_['max_depth']), flush = True)
        elif self.estimatorType == 'ridge':
            cv = sklearn.linear_model.RidgeCV(alphas=self.hyperParams, fit_intercept=False,
                            normalize=False, scoring=None, cv=folds, gcv_mode=None, store_cv_values=False)
            cv.fit(explored_ranking_features, explored_values, sample_weight = inverse_propensities)                        
            self.params = cv.coef_
            print("DirectEstimator:train [INFO] %s CVAlpha: %f" % (self.estimatorType, cv.alpha_),
                        flush = True)
        elif self.estimatorType == 'lasso':
            cv = sklearn.linear_model.LassoCV(alphas=self.hyperParams,
                            precompute=False, max_iter=1000, tol=1e-3,
                            cv=folds, verbose=False, n_jobs=1, positive=False,
                            selection='cyclic', fit_intercept=False, normalize=False, copy_X=True)
            cv.fit(explored_ranking_features, explored_values)                        
            self.params = cv.coef_
            print("DirectEstimator:train [INFO] %s CVAlpha: %f" % (self.estimatorType, cv.alpha_),
                        flush = True)
            
        elif self.estimatorType == 'lstsq':
            self.params = scipy.linalg.lstsq(explored_ranking_features, explored_values, cond = 1e-9)[0]
            print("DirectEstimator:train [INFO] %s" % self.estimatorType, flush = True)
        else:
            print("DirectEstimator:train [ERR] %s not supported." % self.estimatorType, flush = True)
            sys.exit(0)
            
    def estimateHelper(self, new_ranking_features):
        if self.estimatorType != 'tree':
            return self.computeScore(new_ranking_features)
        else:
            if self.tree is None:
                return 0.0
            return self.tree.predict(new_ranking_features.reshape(1,-1))

    def precompute(self, predicted_rankings):
        numQueries, numDocs, numFeatures = numpy.shape(self.policy.dataset.features)
        
        estimates = numpy.zeros(numQueries, dtype = numpy.float32)
        for i in range(numQueries):
            validDocs = min(self.rankingSize, self.policy.dataset.docsPerQuery[i])
            newRankingFeatures = numpy.zeros((self.rankingSize, numFeatures), dtype = numpy.float32)
            newRankingFeatures[0:validDocs,:] = self.policy.dataset.features[i,
                                                            predicted_rankings[i,0:validDocs], :]
            
            newRankingFeatures = numpy.reshape(newRankingFeatures, self.numParams)
            estimates[i] = self.estimateHelper(newRankingFeatures)
        print("DirectEstimator:precompute [INFO] Precomputed ", numpy.shape(predicted_rankings),
                            flush = True)    
        self.precomputed = estimates
        
    def estimate(self, query, explored_ranking, explored_value, new_ranking):
        if self.precomputed is not None:
            return self.precomputed[query]
        else:
            validDocs = min(self.rankingSize, self.policy.dataset.docsPerQuery[query])
            numFeatures = numpy.shape(self.policy.dataset.features)[2]
            newRankingFeatures = numpy.zeros((self.rankingSize, numFeatures), dtype = numpy.float32)
            newRankingFeatures[0:validDocs,:] = self.policy.dataset.features[query, new_ranking[0:validDocs], :]
            newRankingFeatures = numpy.reshape(newRankingFeatures, self.numParams)
            return self.estimateHelper(newRankingFeatures)
              
        
class InversePropensityEstimator(Estimator):
    def __init__(self, ranking_size, policy):
        Estimator.__init__(self, ranking_size, policy)
        self.name = 'IPS'
        
    def estimate(self, query, explored_ranking, explored_value, new_ranking):
        exactMatch = numpy.absolute(new_ranking - explored_ranking).sum() == 0
        if exactMatch:
            numAllowedDocs = self.policy.dataset.docsPerQuery[query]
            validDocs = min(numAllowedDocs, self.rankingSize)
            queryDocProbabilities = self.policy.documentProbabilities[query,0:numAllowedDocs]
            #Propensity calculation only supports multinomial without replacement exploration
            #Not arbitrary joint distribution over slates
            invPropensity = 1.0
            currentDenom = queryDocProbabilities.sum(dtype = numpy.longdouble)
            for j in range(validDocs):
                invPropensity *= currentDenom / queryDocProbabilities[explored_ranking[j]]
                currentDenom -= queryDocProbabilities[explored_ranking[j]]
                if currentDenom <= 0:
                    break
            return explored_value * invPropensity

        return 0.0


class GammaEstimator(Estimator):
    def __init__(self, ranking_size, policy):
        Estimator.__init__(self, ranking_size, policy)
        self.name = 'Gamma'
        
    def estimate(self, query, explored_ranking, explored_value, new_ranking):
        numAllowedDocs = self.policy.dataset.docsPerQuery[query]
        validDocs = min(numAllowedDocs, self.rankingSize)
        vectorDimension = validDocs * numAllowedDocs
        tempRange = range(validDocs)
        
        exploredMatrix = numpy.zeros((validDocs, numAllowedDocs), dtype = numpy.longdouble)
        exploredMatrix[tempRange, explored_ranking[0:validDocs]] = explored_value
        
        newMatrix = numpy.zeros((validDocs, numAllowedDocs), dtype = numpy.longdouble)
        newMatrix[tempRange, new_ranking[0:validDocs]] = 1
        
        posRelVector = exploredMatrix.reshape(vectorDimension)
        newSlateVector = newMatrix.reshape(vectorDimension)
        
        estimatedPhi = numpy.dot(self.policy.gammaInverses[query], posRelVector)
        return numpy.dot(estimatedPhi, newSlateVector)


class DoublyRobustEstimator(Estimator):
    def __init__(self, ips_estimator, direct_estimator):
        self.ipsEstimator = ips_estimator
        self.directEstimator = direct_estimator
        self.name = 'DR_' + ips_estimator.name + '_' + direct_estimator.name
        
    def estimate(self, query, explored_ranking, explored_value, new_ranking):
        reward = self.directEstimator.estimate(query, explored_ranking, explored_value, new_ranking)
                    
        ipsReward = self.ipsEstimator.estimate(query, explored_ranking, explored_value - reward, new_ranking)
                    
        return reward + ipsReward
    
    
class EstimatorWrapper:
    def __init__(self, ips_estimator):
        self.estimator = ips_estimator
        self.runningSum = 0
        self.runningMean = 0.0
 
    def updateRunningAverage(self, value):
        self.runningSum += 1

        delta = value - self.runningMean
        self.runningMean += delta / self.runningSum

    def resetRunningAverage(self):
        self.runningSum = 0
        self.runningMean = 0.0
      
    def estimate(self, query, explored_ranking, explored_value, new_ranking):
        value = self.estimator.estimate(query, explored_ranking, explored_value, new_ranking)

        self.updateRunningAverage(value)
        return self.runningMean


class SelfNormalWrapper(EstimatorWrapper):
    def __init__(self, ips_estimator):
        EstimatorWrapper.__init__(self, ips_estimator)
        self.estimator.name+='_SN'
        self.runningDenominatorMean = 0.0

    def updateSelfNormalAverage(self, value, inv_propensity):
        self.updateRunningAverage(value)

        denominatorDelta = inv_propensity - self.runningDenominatorMean
        self.runningDenominatorMean += denominatorDelta / self.runningSum
                
    def resetRunningAverage(self):
        self.runningSum = 0
        self.runningMean = 0.0
        self.runingDenominatorMean = 0.0
 
    def estimate(self, query, explored_ranking, explored_value, new_ranking):
        value = self.estimator.estimate(query, explored_ranking, explored_value, new_ranking)
        invPropensity = self.estimator.estimate(query, explored_ranking, 1.0, new_ranking)

        self.updateSelfNormalAverage(value, invPropensity)
        if self.runningDenominatorMean != 0.0:
            return 1.0 * self.runningMean / self.runningDenominatorMean
        else:
            return 0.0
      
        
if __name__ == "__main__":
    import Datasets
    import Metrics
    import Logger
    import Policy

    mq2008Data = Datasets.Datasets()
    mq2008Data.loadNpz('../../Data/MQ2008.npz', 'MQ2008')
    numQueries, numDocs, numFeatures = numpy.shape(mq2008Data.features)
    
    anchorTitleURLFeatures = [1,2,3,6,7,8,11,12,13,16,17,18,21,22,23,26,27,28,31,32,33,36,37,38,43,44,46]
    bodyDocFeatures = [0,4,5,9,10,14,15,19,20,24,25,29,30,34,35,39,40,41,42,45,46]
    
    filterer = Policy.DeterministicPolicy(mq2008Data, 'tree')
    filterer.train(mq2008Data, anchorTitleURLFeatures, False)
    filteredData = filterer.createFilteredDataset(5)
    
    newPolicy = Policy.DeterministicPolicy(filteredData, 'ridge')
    newPolicy.train(filteredData, bodyDocFeatures, True)
    
    rankingSize = 3
    revenue = Metrics.Revenue(filteredData, rankingSize)
    
    loggingPolicy = Policy.StochasticPolicy(filteredData, 1, filterer)
    loggingPolicy.setupExploration(rankingSize)
    
    direct = DirectEstimator(loggingPolicy, 'tree', rankingSize, numFeatures)
    ips = InversePropensityEstimator(rankingSize, loggingPolicy)
    gammaEstimator = GammaEstimator(rankingSize, loggingPolicy)
    
    nonuniformLogger = Logger.Logger(loggingPolicy, rankingSize)
    
    numTrainInstances = 1000
    trainQueries, trainRankings, trainMetricValues = \
                        nonuniformLogger.createLog(numTrainInstances, revenue)
    exploredRankingFeatures = numpy.empty((numTrainInstances, numFeatures * rankingSize),
                                            dtype = numpy.float32)
    for i in range(numTrainInstances):
        currentQuery = trainQueries[i]
        currentRanking = trainRankings[:,i]
        numAllowedDocs = filteredData.docsPerQuery[currentQuery]
        validDocs = min(numAllowedDocs, rankingSize)
        rankedDocFeatures = numpy.zeros((rankingSize, numFeatures), dtype = numpy.float32)
        rankedDocFeatures[0:validDocs] = filteredData.features[currentQuery, currentRanking[0:validDocs], :]
        exploredRankingFeatures[i, :] = numpy.reshape(rankedDocFeatures, numFeatures * rankingSize)
        
    folds = sklearn.cross_validation.LabelKFold(trainQueries, n_folds = 5)
    direct.train(folds, exploredRankingFeatures, trainMetricValues)
    
    numInstances = 10000
    
    estimators = [direct, ips, gammaEstimator]
    values = {direct.name:[], ips.name:[], gammaEstimator.name:[], 'Truth':0.0}
    for i in range(numInstances):
        currentQuery, exploredRanking, metricValues = \
                    nonuniformLogger.createOneSample(revenue)
        numAllowedDocs = filteredData.docsPerQuery[currentQuery]
        validDocs = min(numAllowedDocs, rankingSize)
        producedRanking = newPolicy.predict(currentQuery, rankingSize)
        truth = revenue.computeMetric(producedRanking, currentQuery)
        
        values['Truth'] += truth
        
        for estimator in estimators:
            currValue = estimator.estimate(currentQuery,
                exploredRanking, metricValues, producedRanking)
            
            values[estimator.name].append(currValue)
        
        if i%1000 == 0:
            print(".", end = "", flush = True)

    values['Truth'] = values['Truth'] * 1.0 / numInstances
    
    denominators = range(1,numInstances+1)
    for estimator in estimators:
        values[estimator.name] = numpy.cumsum(values[estimator.name], dtype = numpy.longdouble) / denominators
    
    print("")
    print("Uniform WITHOUT Replacement", flush = True)
    print("Truth Estimate=%0.5f" % values['Truth'], flush = True)

    for estimator in estimators:
        print("%s Estimate=%f MSE=%0.5e" % (estimator.name, values[estimator.name][-1],
            (values[estimator.name][-1] - values['Truth'])**2), flush = True)
