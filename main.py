from svm import SVM
import numpy as np

training_size = 50
test_size = 10
dimensionality = 2
shots = int(input("Enter the number of shots you want the quantum computer to take (1-1024): "))
print("Depending on the number of shots it can take few minutes!")

qsvm = SVM(training_size, test_size, dimensionality, shots)
result = qsvm.runSvm()

for i in range(10):
    print("Accuracy: ", np.round(result['testing_accuracy']*100, 2), "%")
