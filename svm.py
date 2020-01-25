import numpy as np
# import scipy
# from scipy.linalg import expm
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import os
import time
import threading
import logging

# from qiskit import Aer
# from qiskit import BasicAer
# from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.input import ClassificationInput
from qiskit.aqua import run_algorithm, QuantumInstance


# from qiskit.aqua.algorithms import QSVM
# from qiskit.aqua.components.feature_maps import SecondOrderExpansion
#
# import logging
# from qiskit.aqua import set_qiskit_aqua_logging

class SVM:

    def __init__(self, training_size, test_size, dimensionality, shots):
        self.training_size = training_size
        self.test_size = test_size
        self.dimensionality = dimensionality
        self.shots = shots

    def Breast_cancer(self):
        class_labels = [r'A', r'B']
        data, target = datasets.load_breast_cancer(True)
        sample_train, sample_test, label_train, label_test = train_test_split(data, target, test_size=self.test_size, random_state=12)

        # Now we standardized for gaussian around 0 with unit variance
        std_scale = StandardScaler().fit(sample_train)
        sample_train = std_scale.transform(sample_train)
        sample_test = std_scale.transform(sample_test)

        # Now reduce number of features to number of qubits
        pca = PCA(n_components=self.dimensionality).fit(sample_train)
        sample_train = pca.transform(sample_train)
        sample_test = pca.transform(sample_test)

        # Scale to the range (-1,+1)
        samples = np.append(sample_train, sample_test, axis=0)
        minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
        sample_train = minmax_scale.transform(sample_train)
        sample_test = minmax_scale.transform(sample_test)

        # Pick training size number of samples from each distro
        training_input = {key: (sample_train[label_train == k, :])[:self.training_size] for k, key in
                          enumerate(class_labels)}
        test_input = {key: (sample_train[label_train == k, :])[self.training_size:(
                self.training_size + self.test_size)] for k, key in enumerate(class_labels)}

        for k in range(0, 2):
            plt.scatter(sample_train[label_train == k, 0][:self.training_size],
                        sample_train[label_train == k, 1][:self.training_size])

        plt.title("PCA Dimension Reduction. Breast Cancer Dataset")

        if os.path.isfile('Dataset.pdf'):
            pass
        else:
            plt.savefig('./Resources/Dataset.pdf', bbox_inches='tight')

        return sample_train, training_input, test_input, class_labels

    def runSvm(self):
        sample_Total, training_input, test_input, class_labels = self.Breast_cancer()

        temp = [test_input[k] for k in test_input]
        total_array = np.concatenate(temp)

        aqua_dict = {
            'problem': {'name': 'classification', 'random_seed': 10598},
            'algorithm': {
                'name': 'QSVM'
            },
            'backend': {'provider': 'qiskit.BasicAer', 'name': 'qasm_simulator', 'shots': self.shots},
            'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2, 'entanglement': 'linear'}
        }

        for i in range(5):
            time.sleep(1)
            print("Ignore Deprecation Warning!!! It can take few minutes. Please wait...")

        algo_input = ClassificationInput(training_input, test_input, total_array)
        result = run_algorithm(aqua_dict, algo_input)

        return result
