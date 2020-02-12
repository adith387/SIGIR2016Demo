###Classes that pre-process datasets for semi-synthetic experiments
import numpy


class Datasets:
    def __init__(self):
        #Must call loadNpz to set all these members
        #before using Datasets objects elsewhere
        self.relevances = None
        self.features = None
        self.docsPerQuery = None
        self.name = None
    
    #file_name: (str) Path to dataset file (.npz format)    
    def loadNpz(self, file_name, name):
        npFile = numpy.load(file_name)
        self.relevances = npFile['relevances']
        self.features = npFile['features']
        self.docsPerQuery = npFile['docsPerQuery']
    
        self.name = name
        numQueries, numDocs, numFeatures = numpy.shape(self.features)
        print("Datasets:loadNpz [INFO] Loaded", file_name,
            " NumQueries, [Min,Max]NumDocs, totalDocs, MaxNumFeatures: ", numQueries,
            numpy.min(self.docsPerQuery), numpy.max(self.docsPerQuery), numDocs, numFeatures, flush = True)
    
    def saveNpz(self, file_path):
        outputFilename = file_path+self.name+'.npz'
        numpy.savez_compressed(outputFilename, relevances = self.relevances,
                                features = self.features, docsPerQuery = self.docsPerQuery)
        print("Datasets:saveNpz [INFO] Saved", outputFilename, flush = True)
    
            
if __name__ == "__main__":
    mq2008Data = Datasets()
    mq2008Data.loadNpz('../../Data/MQ2008.npz', 'MQ2008')
    