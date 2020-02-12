import argparse

      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Helper script to process vw output.')
    parser.add_argument('--path', '-p', metavar='P', type=str, help='Path to vw prediction file',
                        default="../../Logs/test.predict")
    parser.add_argument('--data', '-d', metavar='D', type=str, help='Path to test dataset file',
                        default="../../Logs/vw_test2.dat")
                        
    args = parser.parse_args()
    
    vwFile = open(args.path, 'r')
    vwFile2 = open(args.data, 'r')
    
    predictions = vwFile.readlines()
    losses = vwFile2.readlines()
    
    vwFile.close()
    vwFile2.close()
                        
    numInstances = int(len(predictions) / 2)        #Each instance is followed by a blank line in this file
    
    j = 0
    totalLoss = 0.0
    for i in range(numInstances):
        currentPrediction = int(predictions[2*i])
        currentLosses = []
        token = losses[j].strip()
        while token != '':
            currentLosses.append(float(token.split()[1]))
            j += 1
            token = losses[j].strip()
        j += 1
        
        totalLoss += currentLosses[currentPrediction]
        
    totalLoss /= numInstances
    
    totalLoss = 750*(1 - totalLoss)
    
    print(totalLoss)