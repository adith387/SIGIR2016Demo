###Class that models a policy for exploration or evaluation
import numpy
import scipy.linalg
import scipy.misc
import Datasets
import itertools
import sklearn.linear_model
import sklearn.tree
import sklearn.grid_search
import sys
import pickle
from joblib import Parallel, delayed
        

class Policy:
    #dataset: (Datasets) Must have called Datasets.loadNpz before passing to Policy.
    def __init__(self, dataset):
        self.dataset = dataset          #This is the dataset to be used for predicting rankings.
        ###All sub-classes of Policy should supply a predict method,
        ###Requires: query_id, ranking_size.
        ###Returns: ranking.

           
class CustomPolicy(Policy):
    #Must call train to set these members before calling CustomPolicy:predict.
    def __init__(self, dataset, num_features, ranking_size, model_type, cross_features):
        Policy.__init__(self, dataset)
        self.rankingSize = ranking_size
        self.numDocFeatures = num_features
        self.modelType = model_type
        self.crossFeatures = cross_features
        if self.modelType == 'tree':
            self.tree = None
        else:
            self.policyParams = None

        self.hyperParams = numpy.logspace(-3,2,num=6,base=5)
        self.treeDepths = {'max_depth': list(range(1,8,2))}
        self.numFeatures = self.numDocFeatures + self.rankingSize 
        if self.crossFeatures:
            self.numFeatures += self.numDocFeatures * self.rankingSize
        print("CustomPolicy:init [INFO] Dataset:", dataset.name, flush = True)

    def createFeature(self, docFeatures, position):
        currFeature = numpy.zeros(self.numFeatures, dtype = numpy.float32)
        currFeature[0:self.numDocFeatures] = docFeatures
        currFeature[self.numDocFeatures + position] = 1
        if self.crossFeatures:
            currFeature[self.numDocFeatures + self.rankingSize + position*self.numDocFeatures: \
                        self.numDocFeatures + self.rankingSize + (position+1)*self.numDocFeatures] = docFeatures

        return currFeature

    def predict(self, query_id, ranking_size):
        allowedDocs = self.dataset.docsPerQuery[query_id]
        validDocs = min(allowedDocs, self.rankingSize)

        allScores = numpy.zeros((allowedDocs, self.rankingSize), dtype = numpy.float32)
        for doc in range(allowedDocs):
            for pos in range(self.rankingSize):
                currFeature = self.createFeature(self.dataset.features[query_id,doc,:], pos)

                if self.modelType == 'tree':
                    allScores[doc, pos] = self.tree.predict(currFeature.reshape(1,-1))
                else:
                    allScores[doc, pos] = numpy.dot(currFeature, self.policyParams)

        producedRanking = -numpy.ones(ranking_size, dtype = numpy.int32)
        for i in range(ranking_size):
            maxIndex = numpy.argmax(allScores)
            chosenDoc, chosenPos = numpy.unravel_index(maxIndex, allScores.shape)
            producedRanking[chosenPos] = chosenDoc
            allScores[chosenDoc,:] = float('-inf')
            allScores[:,chosenPos] = float('-inf')

        return producedRanking

    def train(self, dataset, targets):
        numQueries, numDocs, numFeatures = numpy.shape(dataset.features)
        queryDocPairs = dataset.docsPerQuery.sum()
        designMatrix = numpy.zeros((queryDocPairs * self.rankingSize, self.numFeatures), dtype = numpy.float32)
        regressionTargets = numpy.zeros((queryDocPairs * self.rankingSize), dtype = numpy.float32)
        currID = -1
        for i in range(numQueries):
            numAllowedDocs = dataset.docsPerQuery[i]
            for doc in range(numAllowedDocs):
                for j in range(self.rankingSize):
                    currID += 1

                    designMatrix[currID,:] = self.createFeature(dataset.features[i,doc,:], j)
                    regressionTargets[currID] = targets[i,j,doc] 

        if self.modelType == 'tree':
            treeCV = sklearn.grid_search.GridSearchCV(sklearn.tree.DecisionTreeRegressor(presort=True),
                            param_grid = self.treeDepths,
                            scoring=None, fit_params=None, n_jobs=1, pre_dispatch=1,
                            iid=True, cv=5, refit=True, verbose=0, error_score='raise')
            treeCV.fit(designMatrix, regressionTargets)
            self.tree = treeCV.best_estimator_
            print("CustomPolicy:train [INFO] %s Best depth: %d " %
                (self.modelType, treeCV.best_params_['max_depth']), flush = True)
            
        elif self.modelType == 'ridge':
            cv = sklearn.linear_model.RidgeCV(alphas=self.hyperParams, fit_intercept=False, 
                    normalize=True, scoring=None, store_cv_values=False, cv=5, gcv_mode=None)
            cv.fit(designMatrix, regressionTargets)
            self.policyParams = cv.coef_
            print("CustomPolicy:train [INFO] %s CVAlpha: %f " % 
                    (self.modelType, cv.alpha_), flush = True)
            
        elif self.modelType == 'lasso':
            cv = sklearn.linear_model.LassoCV(alphas=self.hyperParams, precompute=False,
                            max_iter=1000, tol=1e-3, cv=5, verbose=False, n_jobs=1, positive=False,
                            selection='cyclic', fit_intercept=False, normalize=True, copy_X=True)
            cv.fit(designMatrix, regressionTargets)
            self.policyParams = cv.coef_
            print("CustomPolicy:train [INFO] %s CVAlpha: %f " % 
                    (self.modelType, cv.alpha_), flush = True)
            
        elif self.modelType == 'lstsq':
            self.policyParams = scipy.linalg.lstsq(designMatrix, regressionTargets, cond = 1e-9)[0]
            print("CustomPolicy:train [INFO] %s " % 
                    self.modelType, flush = True)
            
        else:
            print("CustomPolicy:train [ERR] %s not supported." % self.modelType, flush = True)
            sys.exit(0)
            
        print("CustomPolicy:train [INFO] Created %s predictor using dataset %s." %
                (self.modelType, dataset.name), flush = True)


class DeterministicPolicy(Policy):
    #Must call train to set these members before calling DeterministicPolicy:predict.
    def __init__(self, dataset, model_type):
        Policy.__init__(self, dataset)
        self.modelType = model_type
        self.featureList = None
        if self.modelType == 'tree':
            self.tree = None
        else:
            self.policyParams = None
            
        self.hyperParams = numpy.logspace(-3,2,num=6,base=5)
        self.treeDepths = {'max_depth': list(range(1,8,2))}
        
        print("DeterministicPolicy:init [INFO] Dataset:", dataset.name, flush = True)
    
    #dataset: (Datasets) Must have called Datasets.loadTxt/loadNpz. Used to train the scoring function of this policy.
    #feature_list: List(int) List of features that should be used for training.
    #exponential_gains: (bool) Should the regression targets be (relevances) or exponentiated (for DCG).
    def train(self, dataset, feature_list, exponential_gains):
        self.featureList = feature_list
        numQueries, numDocs, numFeatures = numpy.shape(dataset.features)
        myFeatures = numpy.zeros(numpy.shape(dataset.features), dtype = numpy.float32)
        myFeatures[:,:,feature_list] = dataset.features[:,:,feature_list]
        
        flatFeatures = numpy.reshape(myFeatures, (numQueries*numDocs, numFeatures))
        flatRelevances = numpy.reshape(dataset.relevances, numQueries*numDocs)
        dummyIndices = flatRelevances == -1
        selectedIndices = numpy.logical_not(dummyIndices)
        
        if exponential_gains:
            flatRelevances = numpy.exp2(flatRelevances) - 1
        flatRelevances = flatRelevances[selectedIndices]
        flatFeatures = flatFeatures[selectedIndices, :]

        if self.modelType == 'tree':
            treeCV = sklearn.grid_search.GridSearchCV(sklearn.tree.DecisionTreeRegressor(presort=True),
                            param_grid = self.treeDepths,
                            scoring=None, fit_params=None, n_jobs=1, pre_dispatch=1,
                            iid=True, cv=5, refit=True, verbose=0, error_score='raise')
            treeCV.fit(flatFeatures, flatRelevances)
            self.tree = treeCV.best_estimator_
            print("DeterministicPolicy:train [INFO] %s Best depth: %d Exponentiated Gains?" %
                (self.modelType, treeCV.best_params_['max_depth']), exponential_gains, flush = True)
            
        elif self.modelType == 'ridge':
            cv = sklearn.linear_model.RidgeCV(alphas=self.hyperParams, fit_intercept=False, 
                    normalize=False, scoring=None, store_cv_values=False, cv=5, gcv_mode=None)
            cv.fit(flatFeatures, flatRelevances)
            self.policyParams = cv.coef_
            print("DeterministicPolicy:train [INFO] %s CVAlpha: %f Exponentiated Gains?" % 
                    (self.modelType, cv.alpha_), exponential_gains, flush = True)
            
        elif self.modelType == 'lasso':
            cv = sklearn.linear_model.LassoCV(alphas=self.hyperParams, precompute=False,
                            max_iter=1000, tol=1e-3, cv=5, verbose=False, n_jobs=1, positive=False,
                            selection='cyclic', fit_intercept=False, normalize=False, copy_X=True)
            cv.fit(flatFeatures, flatRelevances)
            self.policyParams = cv.coef_
            print("DeterministicPolicy:train [INFO] %s CVAlpha: %f Exponentiated Gains?" % 
                    (self.modelType, cv.alpha_), exponential_gains, flush = True)
            
        elif self.modelType == 'lstsq':
            self.policyParams = scipy.linalg.lstsq(flatFeatures, flatRelevances, cond = 1e-9)[0]
            print("DeterministicPolicy:train [INFO] %s Exponentiated Gains?" % 
                    self.modelType, exponential_gains, flush = True)
            
        else:
            print("DeterministicPolicy:train [ERR] %s not supported." % self.modelType, flush = True)
            sys.exit(0)
            
        print("DeterministicPolicy:train [INFO] Created %s predictor using dataset %s. Features:" %
                (self.modelType, dataset.name), feature_list, flush = True)
            
    def predict(self, query_id, ranking_size):  
        numFeatures = numpy.shape(self.dataset.features)[2]
        allowedDocs = self.dataset.docsPerQuery[query_id]    
        validDocs = min(allowedDocs, ranking_size)
        
        allDocFeatures = numpy.zeros((allowedDocs, numFeatures), dtype = numpy.float32)
        currentFeatures = self.dataset.features[query_id, 0:allowedDocs,:]
        allDocFeatures[:, self.featureList] = currentFeatures[:,self.featureList]
        allDocScores = None
        if self.modelType == 'tree':
            allDocScores = self.tree.predict(allDocFeatures)
        else:
            allDocScores = numpy.dot(allDocFeatures, self.policyParams)
            
        sortedDocScores = numpy.argsort(-allDocScores)
        
        producedRanking = -numpy.ones(ranking_size, dtype = numpy.int32)
        producedRanking[0:validDocs] = sortedDocScores[0:validDocs]
        return producedRanking

    def predictAll(self, ranking_size):
        numQueries = numpy.shape(self.dataset.features)[0]
        predictedRankings = - numpy.ones((numQueries, ranking_size), dtype = numpy.int32)
        for i in range(numQueries):
            predictedRankings[i,:] = self.predict(i, ranking_size)
            
        print("DeterministicPolicy:predictAll [INFO] Generated all predictions for %s using %s." %
                (self.dataset.name, self.name()), flush = True)
        return predictedRankings
    
    def name(self):
        outName = self.dataset.name+'_'+self.modelType+'_'
        if 0 in self.featureList:
            outName += 'body'
        else:
            outName += 'anchor'
        return outName
        
    def savePolicy(self, file_path):
        outputFileName = file_path+self.name()+'.pkl'
        
        outputData = None
        if self.modelType == 'tree':
            outputData = (self.featureList, self.tree)
        else:
            outputData = (self.featureList, self.policyParams)
            
        outFile = open(outputFileName, 'wb')
        pickle.dump(outputData, outFile, -1)
        outFile.close()
        print("DeterministicPolicy:savePolicy [INFO] Saved trained policy at ",
                outputFileName, flush = True)
        
    def loadPolicy(self, file_name):
        inFile = open(file_name, 'rb')
        (self.featureList, model) = pickle.load(inFile)
        inFile.close()
        if self.modelType == 'tree':
            self.tree = model
        else:
            self.policyParams = model
    
        print("DeterministicPolicy:loadPolicy [INFO] Loaded trained policy from ",
                file_name, flush = True)
        
    #num_allowed_docs: (int) Creates a new dataset where the max docs per query is num_allowed_docs.
    #                        Uses policyParams to rank and filter the original document set.
    def createFilteredDataset(self, num_allowed_docs):
        newDataset = Datasets.Datasets()
        oldNumQueries, oldNumDocuments, oldNumFeatures = numpy.shape(self.dataset.features)
        newDataset.relevances = -1*numpy.ones((oldNumQueries, num_allowed_docs), dtype = numpy.int32)
        newDataset.features = numpy.nan*numpy.ones((oldNumQueries, num_allowed_docs, oldNumFeatures),
                                                        dtype = numpy.float32)
        newDataset.docsPerQuery = numpy.clip(self.dataset.docsPerQuery, 0, num_allowed_docs)
        for i in range(oldNumQueries):
            producedRanking = self.predict(i, num_allowed_docs)
            allowedDocs = self.dataset.docsPerQuery[i]
            validDocs = min(num_allowed_docs, allowedDocs)
            newDataset.relevances[i, 0:validDocs] = self.dataset.relevances[i, producedRanking[0:validDocs]]
            newDataset.features[i, 0:validDocs, :] = self.dataset.features[i, producedRanking[0:validDocs], :]

        newDataset.name = self.name()+'_'+str(num_allowed_docs)
        
        print("DeterministicPolicy:createFilteredDataset [INFO] %s MaxNumDocs %d" % 
                (newDataset.name, num_allowed_docs), flush = True)
        return newDataset
    
        
class StochasticPolicy(Policy):
    #All sub-classes of StochasticPolicy should supply a setupExploration method to set these members.
    #det_policy:    DeterministicPolicy or None. If None, a document distribution
    #                                           is sampled from a Dirichlet (uniform hyper-prior).
    #               If det_policy is specified, documents are scored to get unnormalized
    #                       probability distribution.
    #temperature:   (float)   Multiplier for document scores. For the Dirichlet case, this constant is added 
    #                       and the distribution is renormalized.
    def __init__(self, dataset, temperature, det_policy):
        Policy.__init__(self, dataset)
        self.gammaInverses = None               #Required for gamma estimator.
        self.documentProbabilities = None       #Internally used by StochasticPolicy:predict.
        
        self.temperature = temperature
        self.detPolicy = det_policy
        
        print("StochasticPolicy:init [INFO] Dataset: %s " % dataset.name, flush = True)
        if det_policy is not None:
            print("StochasticPolicy:init [INFO] Deterministic policy: %s Temperature:" % det_policy.name(),
                        temperature, flush = True)
        else:
            print("StochasticPolicy:init [INFO] Deterministic policy: None Temperature:",
                        temperature, flush = True)
                        
    def predict(self, query_id, ranking_size):
        allowedDocs = self.dataset.docsPerQuery[query_id]
        samplingProbability = self.documentProbabilities[query_id,0:allowedDocs]
        producedRanking = -numpy.ones(ranking_size, dtype = numpy.int32)
        validDocs = min(ranking_size, allowedDocs)
        producedRanking[0:validDocs] = numpy.random.choice(allowedDocs, size = validDocs,
                                replace = False, p = samplingProbability)
        return producedRanking
    
    #query_id:      (int)        ID of the query for which document distribution should be computed.
    def explorationHelper(self, query_id):
        numAllowedDocs = self.dataset.docsPerQuery[query_id]
        uniformDistribution = numpy.ones(numAllowedDocs, dtype = numpy.float32) / numAllowedDocs
        probDistribution = numpy.zeros(numAllowedDocs, dtype = numpy.float32)
        if self.detPolicy is None:
            probDistribution = numpy.random.dirichlet(uniformDistribution)
            probDistribution += self.temperature
            probDistribution /= probDistribution.sum(dtype = numpy.float32)
        else:
            if self.temperature <= 0:
                probDistribution = uniformDistribution
            else:
                numFeatures = numpy.shape(self.dataset.features)[2]
                allDocFeatures = numpy.zeros((numAllowedDocs, numFeatures), dtype = numpy.float32)
                currentFeatures = self.dataset.features[query_id, 0:numAllowedDocs,:]
                allDocFeatures[:, self.detPolicy.featureList] = currentFeatures[:, self.detPolicy.featureList]
                allDocScores = None
                if self.detPolicy.modelType != 'tree':
                    allDocScores = numpy.dot(allDocFeatures, self.detPolicy.policyParams)
                else:
                    allDocScores = self.detPolicy.tree.predict(allDocFeatures)
                
                allDocScores *= self.temperature
                
                partitionFunction = scipy.misc.logsumexp(allDocScores)
                logProb = allDocScores - partitionFunction
                probDistribution = numpy.exp(logProb).astype(numpy.float32)
        
        return probDistribution

    def gammaHelper(self, i, ranking_size):
        numAllowedDocs = self.dataset.docsPerQuery[i]
        currentDistribution = self.documentProbabilities[i, 0:numAllowedDocs]
        
        validDocs = min(ranking_size, numAllowedDocs)
        #Brute-force
        slates = None
        if self.temperature > 0:
            slates = numpy.zeros(tuple([numAllowedDocs for p in range(validDocs)]),
                                        dtype = numpy.float32)
            for x in itertools.permutations(range(numAllowedDocs), validDocs):
                currentDenom = currentDistribution.sum(dtype = numpy.longdouble)
                slateProb = 1.0
                for p in range(validDocs):
                    slateProb *= (currentDistribution[x[p]] / currentDenom)
                    currentDenom -= currentDistribution[x[p]]
                    if currentDenom <= 0:
                        break
                slates[tuple(x)] = slateProb
                
        gamma = numpy.zeros((numAllowedDocs * validDocs, numAllowedDocs * validDocs),
                                dtype = numpy.longdouble)
        for p in range(validDocs):
            currentStart = p * numAllowedDocs
            currentEnd = p * numAllowedDocs + numAllowedDocs
            currentMarginals = None
            if self.temperature > 0:
                currentMarginals = numpy.sum(slates, axis=tuple([q for q in range(validDocs) if q != p]),
                                                            dtype = numpy.longdouble)
            else:
                currentMarginals = numpy.ones(numAllowedDocs, dtype = numpy.longdouble) / numAllowedDocs
            gamma[currentStart:currentEnd, currentStart:currentEnd] = numpy.diag(currentMarginals)
            
        for p in range(validDocs):
            for q in range(p+1, validDocs):
                currentRowStart = p * numAllowedDocs
                currentRowEnd = p * numAllowedDocs + numAllowedDocs
                currentColumnStart = q * numAllowedDocs
                currentColumnEnd = q * numAllowedDocs + numAllowedDocs
                pairMarginals = None
                if self.temperature > 0:
                    pairMarginals = numpy.sum(slates, 
                                    axis=tuple([r for r in range(validDocs) if r != p and r != q]), 
                                    dtype = numpy.longdouble)
                else:
                    pairMarginals = numpy.ones((numAllowedDocs, numAllowedDocs), dtype = numpy.longdouble) /\
                                            (numAllowedDocs * (numAllowedDocs - 1))
                    numpy.fill_diagonal(pairMarginals, 0)

                gamma[currentRowStart:currentRowEnd, currentColumnStart:currentColumnEnd] = pairMarginals
                gamma[currentColumnStart:currentColumnEnd, currentRowStart:currentRowEnd] = pairMarginals.T
        
        thirdOut = scipy.linalg.pinv(gamma, cond = 1e-15, rcond = 1e-15)
        return thirdOut
        
    #ranking_size: (int) Size of rankings to be predicted (used to define gamma).
    def setupExploration(self, ranking_size):
        numQueries, numDocs, numFeatures = numpy.shape(self.dataset.features)
        
        self.documentProbabilities = -numpy.ones((numQueries, numDocs), dtype = numpy.float32)
        self.gammaInverses = [None]*numQueries
        
        for i in range(numQueries):
            numAllowedDocs = self.dataset.docsPerQuery[i]
            currentDistribution = self.explorationHelper(i)
            self.documentProbabilities[i, 0:numAllowedDocs] = currentDistribution
        
        #self.gammaHelper should be supplied by subclasses    
        responses = Parallel(n_jobs = -2, verbose = 50)(delayed(self.gammaHelper)(i,
                                ranking_size) for i in range(numQueries))
            
        for i in range(numQueries):
            self.gammaInverses[i] = responses[i]
                
        print("")        
        print("StochasticPolicy:setupExploration [INFO] RankingSize: %d" %
                    ranking_size, flush = True)
    
    def name(self):
        outName = self.dataset.name+'_stoc'
        if self.detPolicy is not None:
            outName += str(self.temperature) +'_('+self.detPolicy.name()+')'
        else:
            outName += str(self.temperature) +'_(None)'
        return outName
        
    def serializeGammas(self, file_path, ranking_size):
        outputFileName = file_path+self.name()+'_gamma'+str(ranking_size)+'.pkl'
        outputData = (self.gammaInverses,self.documentProbabilities)
            
        outFile = open(outputFileName, 'wb')
        pickle.dump(outputData, outFile, -1)
        outFile.close()
        print("StochasticPolicy:serializeGammas [INFO] Saved computed gammas",flush = True)
    
    def loadGammas(self, file_name):
        inFile = open(file_name, 'rb')
        (self.gammaInverses,self.documentProbabilities) = pickle.load(inFile)
        inFile.close()
        
        print("StochasticPolicy:loadGammas [INFO] Loaded precomputed gammas from %s" % file_name, flush = True)
                
        
if __name__ == "__main__":
    mq2008Data = Datasets.Datasets()
    mq2008Data.loadNpz('../../Data/MQ2008.npz', 'MQ2008')
    numQueries, numDocs, numFeatures = numpy.shape(mq2008Data.features)
    
    anchorTitleURLFeatures = [1,2,3,6,7,8,11,12,13,16,17,18,21,22,23,26,27,28,31,32,33,36,37,38,43,44,46]
    bodyDocFeatures = [0,4,5,9,10,14,15,19,20,24,25,29,30,34,35,39,40,41,42,45,46]
    
    rankingSize = 4
    
    detLogger = DeterministicPolicy(mq2008Data, 'lasso')
    detLogger.train(mq2008Data, anchorTitleURLFeatures, False)
    newData = detLogger.createFilteredDataset(8)
    print("New dataset NumQueries, NumDocs, NumFeatures, [Min,Max]NumDocs: ", numpy.shape(newData.features),
                    numpy.min(newData.docsPerQuery), numpy.max(newData.docsPerQuery))
    
    custom1 = CustomPolicy(newData, numFeatures, rankingSize, 'tree', False)
    custom2 = CustomPolicy(newData, numFeatures, rankingSize, 'lasso', True)
    
    targets = numpy.tile(newData.relevances[:,None,:], (1, rankingSize, 1))
    
    custom1.train(newData, targets)
    custom2.train(newData, targets)

    numQueries = numpy.shape(mq2008Data.features)[0]
    queryID = numpy.random.randint(0, numQueries)
    print("Selected queryID:", queryID)
    
    print("Custom 1 ranking: ", custom1.predict(queryID, rankingSize))
    print("Custom 2 ranking: ", custom2.predict(queryID, rankingSize))
    
    detLogger2 = DeterministicPolicy(newData, 'tree')
    detLogger2.train(newData, bodyDocFeatures, True)
    
    predictedRankings = detLogger2.predictAll(rankingSize)
    print("Predicted rankings :", numpy.shape(predictedRankings), flush = True)
    
    detLogger2.savePolicy('./')
    
    detLogger3 = DeterministicPolicy(newData, 'tree')
    detLogger3.loadPolicy('./'+detLogger2.name()+'.pkl')
    
    predictedRankings2 = detLogger3.predictAll(rankingSize)
    print("After load, are predicted rankings same?", numpy.allclose(predictedRankings, predictedRankings2),
            flush = True)
            
    uniformNoRepLogger = StochasticPolicy(newData, 0, None)
    uniformNoRepLogger.setupExploration(rankingSize)
    
    nonUniformNoRepLogger = StochasticPolicy(newData, 1, None)
    nonUniformNoRepLogger.setupExploration(rankingSize)
    
    paramNoRepLogger = StochasticPolicy(newData, 1, detLogger2)
    paramNoRepLogger.setupExploration(rankingSize)
    
    print("Uniform Stochastic ranking: ", uniformNoRepLogger.predict(queryID, rankingSize))
    print("NonUniform Stochastic ranking: ", nonUniformNoRepLogger.predict(queryID, rankingSize))
    print("Parametrized Stochastic ranking: ", paramNoRepLogger.predict(queryID, rankingSize))
    
    uniformNoRepLogger.serializeGammas('./', rankingSize)
    copyLogger = StochasticPolicy(newData, 0, None)
    copyLogger.loadGammas('./'+uniformNoRepLogger.name()+'_gamma'+str(rankingSize)+'.pkl')
    
    print("Copy Stochastic ranking: ", copyLogger.predict(queryID, rankingSize))
    