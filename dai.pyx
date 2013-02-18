# A Cython wrapper for libDAI
# Author: Sameh Khamis

from __future__ import division
from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np
cimport numpy as np

np.import_array()


cdef extern from 'dai/varset.h' namespace 'dai':
    cdef cppclass VarSet:
        VarSet() except +
        VarSet insert(Var)
        size_t size()
        vector[Var] elements()

cdef extern from 'dai/var.h' namespace 'dai':
    cdef cppclass Var:
        Var() except +
        Var(size_t, size_t) except +
        size_t label()
        size_t states()

cdef extern from 'dai/factor.h' namespace 'dai':
    cdef cppclass Factor:
        Factor(vector[Var], vector[double]) except +
        VarSet vars()
        double operator[](size_t)

cdef extern from 'dai/factorgraph.h' namespace 'dai':
    cdef cppclass FactorGraph:
        FactorGraph(vector[Factor]) except +
        size_t nrVars()
        size_t nrFactors()
        Var var(size_t)
        Factor factor(size_t)

cdef extern from 'dai/daialg.h' namespace 'dai':
    cdef cppclass InfAlg:
        void init() except +
        void run() except +
        double logZ() except -1
        vector[Factor] beliefs()
        double maxDiff() except -1
        Factor belief(Var)
        Factor belief(VarSet)
        vector[size_t] findMaximum() except +RuntimeError

cdef extern from 'dai/alldai.h' namespace 'dai':
    InfAlg* newInfAlgFromString(string nameopts, FactorGraph fg) except +


cdef vector[Factor] factors_py2cpp(list factors, string order='F'):
    # Convert the list of (member, prob) tuples to a vector of factors
    cdef vector[Factor] cfactors
    cdef vector[Var] variables
    cdef vector[double] values
    
    cfactors.reserve(len(factors))
    for member, prob in factors:
        variables.reserve(len(member))
        for i in np.arange(len(member)):
            variables.push_back(Var(member[i], prob.shape[i]))
        
        values.reserve(prob.size)
        for p in prob.flatten(order):
            values.push_back(p)
        
        cfactors.push_back(Factor(variables, values))
        variables.clear()
        values.clear()
    
    return cfactors

cdef list factors_cpp2py(vector[Factor] cfactors, string order='F'):
    # Convert the vector of factors to a list of (member, prob) tuples
    factors = [(None, None)] * cfactors.size()
    cdef VarSet variables
    cdef Var var
    
    cdef np.ndarray[np.int32_t] states
    for i in np.arange(cfactors.size()):
        variables = cfactors[i].vars()
        member = np.empty([variables.size()], dtype=np.int32)
        states = np.empty([variables.size()], dtype=np.int32)
        
        for j in np.arange(variables.size()):
            var = variables.elements()[j]
            member[j] = var.label()
            states[j] = var.states()
        
        prob = np.empty([states.prod()], dtype=np.float)
        for j in np.arange(prob.size):
            prob[j] = cfactors[i][j]
        
        prob = prob.reshape(states, order=order)
        factors[i] = (member, prob)
    
    return factors


def dai(factors, method = 'BP', props = {}, with_extra_beliefs=True, with_map_state=True, order='F'):
    """
    dai(factors, method = 'BP', props = {}, with_extra_beliefs=True, with_map_state=True)
    
    factors: a list of (member, prob) tuples, both numpy arrays
    method: a string containing the name of a supported algorithm
    props: algorithm parameters specified as a dictionary of string to string
    with_extra_beliefs: return separately the variable and factor beliefs
    with_map_state: return the joint map state
    """
    # Prepare input
    cdef vector[Factor] cfactors = factors_py2cpp(factors, order)
    
    # Construct the factor graph and run inference
    cdef string cnameopts = '%s[%s]' % (method, ','.join(['%s=%s' %(key, props[key]) for key in props.keys()]))
    cdef FactorGraph *fg = new FactorGraph(cfactors)
    cdef InfAlg *alg = newInfAlgFromString(cnameopts, fg[0])
    alg.init()
    alg.run()
    
    # Prepare output
    logz = alg.logZ()
    q = factors_cpp2py(alg.beliefs(), order)
    maxdiff = alg.maxDiff()
    res = [logz, q, maxdiff]
    
    cdef vector[Factor] cqv
    cdef vector[Factor] cqf
    if with_extra_beliefs:
        cqv.reserve(fg.nrVars())
        for i in np.arange(fg.nrVars()):
            cqv.push_back(alg.belief(fg.var(i)))
        qv = factors_cpp2py(cqv, order)
        
        cqf.reserve(fg.nrFactors())
        for i in np.arange(fg.nrFactors()):
            cqf.push_back(alg.belief(fg.factor(i).vars()))
        qf = factors_cpp2py(cqf, order)
        
        res.append(qv)
        res.append(qf)
    
    cdef vector[size_t] cqmap
    if with_map_state:
        try:
            cqmap = alg.findMaximum()
            qmap = np.empty([cqmap.size()])
            for i in np.arange(cqmap.size()):
                qmap[i] = cqmap[i]
        except RuntimeError:
            qmap = []
        
        res.append(qmap)
    
    # Clean up and return the output
    del alg
    del fg
    return res

