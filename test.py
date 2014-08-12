import numpy as np
import dai

# Note:
# The indexing for input and output arrays is Fortan-style by default for notational convenience
# If you prefer C-style indexing and want to reverse the indexes yourself, specify order='C'

# build the graph
factors = []

member = [0]
prob = np.array([0.4, 0.6])
factors.append((member, prob))

member = [1]
prob = np.array([0.6, 0.4])
factors.append((member, prob))

member = [0, 1]
prob = np.array([0.1, 0.7, 0.1, 0.1]).reshape(2, 2)
factors.append((member, prob))

# do inference
props = {'inference': 'SUMPROD', 'updates': 'SEQMAX', 'tol': '1e-6', 'maxiter': '100', 'logdomain': '0'}
logz, q, maxdiff, qv, qf, qmap = dai.dai(factors, 'BP', props, order='F')

# print the output
print('LogZ = %.4f' % logz)
print()

print('All beliefs:')
for member, prob in q:
    print('Member: %s\nProb: %s' % (member, prob))
print()

print('MaxDiff = %.4f' % maxdiff)
print()

print('Variable beliefs:')
for member, prob in qv:
    print('Member: %s\nProb: %s' % (member, prob))
print()

print('Factor beliefs:')
for member, prob in qf:
    print('Member: %s\nProb: %s' % (member, prob))
print()

print('Map state = %s' % qmap)
print()

