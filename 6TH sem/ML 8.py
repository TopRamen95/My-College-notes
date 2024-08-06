import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

heartDisease = pd.read_csv('heart.csv')
heartDisease = heartDisease.replace('?', np.nan)
model = BayesianModel([
    ('age', 'heartdisease'),
    ('sex', 'heartdisease'),
    ('exang', 'heartdisease'),
    ('cp', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'chol')
])
print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)
print('\nInferencing with Bayesian Network:')
HeartDisease_test_infer = VariableElimination(model)

print('\n1. Probability of HeartDisease given evidence = restecg')
q1 = HeartDisease_test_infer.query(variables=['heartdisease'], evidence={'restecg': 1})
print(q1)

print('\n2. Probability of HeartDisease given evidence = cp')
q2 = HeartDisease_test_infer.query(variables=['heartdisease'], evidence={'cp': 2})
print(q2)
print('\nInferencing with Bayesian Network:')
HeartDisease_test_infer = VariableElimination(model)

print('\n1. Probability of HeartDisease given evidence = restecg')
q1 = HeartDisease_test_infer.query(variables=['heartdisease'], evidence={'restecg': 1})
print(q1)

print('\n2. Probability of HeartDisease given evidence = cp')
q2 = HeartDisease_test_infer.query(variables=['heartdisease'], evidence={'cp': 2})
print(q2)
