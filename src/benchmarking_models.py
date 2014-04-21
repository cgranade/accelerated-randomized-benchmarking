#!/usr/bin/python
# -*- coding: utf-8 -*-

## FEATURES ###################################################################

from __future__ import division

## IMPORTS ####################################################################

from itertools import starmap

import numpy as np
from qinfer.abstract_model import Model, DifferentiableModel
from qinfer.rb import p, F

from operator import mul

try:
    import qecc as q
except:
    print "QuaEC not found. Some features may not work."
    q = None
    
try:
    import qutip as qp
except:
    print "QuTiP not found. Some features may not work."

## FUNCTIONS ##################################################################

def av_gate_fidelity(oper):
    """
    Given a Qobj representing the supermatrix form of a map, returns the
    average gate fidelity of that map.
    """
    kraus_form = qp.choi_to_kraus(qp.super_to_choi(oper))
    d = kraus_form[0].shape[0]
    
    return (
        d + sum(
            np.abs(A_k.tr())**2
            for A_k in kraus_form
        )
    ) / (d**2 + d)

def _super_U(U):
    return np.kron(U.conj(), U)

## CONSTANTS ##################################################################

IDEAL_GATESET = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]]),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    'R_pi4': q.phase(1, 0).as_unitary(),
    'H': q.hadamard(1, 0).as_unitary()
}
IDEAL_GATESET = {gate: _super_U(U) for gate, U in IDEAL_GATESET.iteritems()}

IDEAL_RHO_PSI = np.diag(np.array([1, 0], dtype=complex)).flatten('F')
IDEAL_E_PSI   = IDEAL_RHO_PSI.conj().transpose()

## CLASSES ####################################################################

class RBPhysicalSimulator(object):
    r"""
    Any omitted parameters are replaced by the *ideal* single-qubit
    implementations.
    
    :param dict gateset: Dictionary from gate names
        (``"I", "H", "R_pi4", "X", "Y", "Z"``) to NumPy arrays containing
        column-stacked superoperators for those gates.
    :param E_psi: Vector representing the measurement effect :math:`E_\psi`
        in the column-stacking convention.
    :param rho_psi: Vector representing the state preparation :math:`\rho_\psi`
        in the column-stacking convention.
    :param bool paranoid: Enables assertions which are slow to compute, but
        useful for checking numerical correctness.
        
    .. todo::
        Use QuTiP superops instead.
    """
    
    # NOTE: single-qubit for now.
    def __init__(self, gateset=None, E_psi=None, rho_psi=None, paranoid=False):
        self._gateset = gateset if gateset is not None else IDEAL_GATESET
        self._E_psi = E_psi if E_psi is not None else IDEAL_E_PSI
        self._rho_psi = rho_psi if rho_psi is not None else IDEAL_RHO_PSI
        self._nq = 1
        
        self._paranoid = bool(paranoid)
        # If all arguments are unspecified, then this is the ideal benchmarking
        # simulator, and should always return 1. Thus, we can turn on a paranoid
        # assertion that only makes sense for the ideal case.
        self._ideal = (gateset is None) and (E_psi is None) and (rho_psi is None)
        
        # Initialize the Clifford factor group C_n / P_n.
        # We store as a list of Cliffords, and will then store H/R_pi4
        # decompositions.
        factor_group = list(q.clifford_group(self._nq, False))
        self._decompositions = map(self.__decompose, factor_group)
        self._pauli_group = list(q.pauli_group(self._nq))
        self._ideal_pauli_group = [_super_U(P.as_unitary()) for P in self._pauli_group]
        
        # We now store the actual representatives of the factor group that we
        # will be using, since the H/R_pi4 decomposition need not produce the
        # same representatives as q.clifford_group finds.
        self._factor_group = map(self.__recompose, self._decompositions)
        self._ideal_factor_group = map(self.__ideal_U, self._factor_group)
         
    @staticmethod
    def __ideal_U(C):
        C = C.circuit_decomposition(False).as_clifford()
        if C is not q.EmptyClifford:
            return _super_U(C.as_unitary())
        else:
            return np.eye(4, dtype=complex)
        
    @staticmethod
    def __recompose(locs):
        circ = q.Circuit(*[(loc, 0) for loc in locs])
        C = circ.as_clifford()
        return C if C is not q.EmptyClifford else q.eye_c(1)
        
    @staticmethod
    def __decompose(C):
        """
        Decomposes a Clifford ``C`` in a canonical manner, as a tuple of
        strings representing elements of the gateset.
        """
        return tuple(
            loc.kind
            for loc in
            C.circuit_decomposition(False).cancel_selfinv_gates()
        )
        
    def gate(self, idx_C, idx_P):
        P = self._gateset[self._pauli_group[idx_P].op]
        decomp = self._decompositions[idx_C]
        if decomp:
            C = reduce(np.dot, [
                self._gateset[loc]
                for loc in reversed(decomp)
            ])
            return np.dot(C, P)
        else:
            return P
        
    def random_sequence(self, m):
        l_C = len(self._factor_group)
        l_P = len(self._pauli_group)
        
        return self.append_inverse([
            (np.random.randint(0, l_C), np.random.randint(0, l_P))
            for idx_m in xrange(m - 1)
        ])
        
    def random_interleaved_sequence(self, m, target):
        l_C = len(self._factor_group)
        l_P = len(self._pauli_group)
        
        return self.append_inverse([
            (np.random.randint(0, l_C), np.random.randint(0, l_P))
            if idx_m % 2 == 0 else target
            for idx_m in xrange(m - 1)
        ])
        
    def append_inverse(self, sequence):
        """
        Given a sequence specified as a list of Clifford and Pauli indices,
        returns a new list such that the sequence represented multiplies to
        the identity.
        """
        C = q.eye_c(self._nq)
        for idx_C, idx_P in sequence:
            C = self._factor_group[idx_C] * self._pauli_group[idx_P].as_clifford() * C

        # Find the actual inverse of the sequence, then try to implement that
        # inverse with the decompositions available, and note the required
        # Pauli correction.
        C_inv = C.inv()
        idx_C_inv = self._decompositions.index(self.__decompose(C_inv))
        
        C_inv_decomp = self._factor_group[idx_C_inv]
        
        # The correct Pauli to end the sequence with comes from multiplying
        # the ideal version gate we have access to by the actual inverse,
        # then seeing what we have left. This uses that P² = I for all Paulis P
        # such that P¯¹ = P.
        P_inv = q.paulify(
            C_inv.inv() * C_inv_decomp
        )
        idx_P_inv = self._pauli_group.index(P_inv)
        
        new_seq = sequence + [(idx_C_inv, idx_P_inv)]
        
        # This assert is very slow even for most asserts, so we will only use it
        # if we're extra paranoid.
        if self._paranoid:
            assert np.sum(np.abs(_super_U(reduce(np.dot, [
                (self._factor_group[idx_C] * self._pauli_group[idx_P].as_clifford()).as_unitary()
                for idx_C, idx_P in reversed(new_seq)
            ])) - np.eye(4))) < 1e-5
        
        return new_seq
        
    def p_survival(self, sequence):
        if sequence:
            S_i_j = reduce(np.dot, list(starmap(self.gate, reversed(sequence))))
        else:
            S_i_j = np.eye(4, dtype=complex)
            
        p_survival = reduce(
            np.dot, [
                self._E_psi.conj(),
                S_i_j,
                self._rho_psi
            ]
        )
        if self._ideal and self._paranoid:
            assert np.abs(p_survival - 1) < 1e-5, "Ideal RB simulator did not produce identity."
        return p_survival
        
    def sample_sequence(self, sequence):
        return np.random.random() < self.p_survival(sequence)
        
    def sample_random_sequence(self, m):
        return self.sample_sequence(self.random_sequence(m))
        
    def sample_interleaved_sequence(self, m, target):
        return self.sample_sequence(self.random_interleaved_sequence(m, target))

    def interleaved_model_parameters(self, target):
        r"""
        Returns the true values of :math:`A`, :math:`B`, :math:`p_{\text{ref}}`
        and :math:`\tilde{p}` for this gateset and the given target.
        These parameters are calculated according to the zeroth-order model of
        [MGE12]_, assuming time independance of the errors in the gateset, such
        that:
        
        .. math::
        
            A = \mathrm{Tr}[E_\psi \lambda\left(\rho_\psi - \frac{\mathbb{1}}{d}\right)] \\
            B = \mathrm{Tr}[E_\psi \lambda\left(\frac{\mathbb{1}}{d}\right)] \\
            p_\text{ref} = \mathbb{E}_{C\sim\mathcal{C}}[\frac{d F(C) - 1}{d - 1}] \\
            \tilde{p} = \frac{d F(C_{\text{target}}) - 1}{d - 1},
            
        where :math:`\Lambda` is the average error map, taken over all Clifford
        group elements,
        
        .. math::
        
            \Lambda_i = C_i \circ C_{\text{ideal},i}^\dagger \\
            \Lambda = \mathbb{E}_{C\sim\mathcal{C}} [\Lambda_i]
        """
        Lambdas = np.array([[
            reduce(
                np.dot,
                [self.gate(idx_C, idx_P), ideal_P.conj().T, ideal_C.conj().T]
            )
            for idx_P, ideal_P in enumerate(self._ideal_pauli_group)]            
            for idx_C, ideal_C in enumerate(self._ideal_factor_group)
        ])
        Lambda = np.mean(Lambdas, axis=(0,1))
        B = np.real(reduce(np.dot, [
            self._E_psi, Lambda, np.eye(2).flatten(order='F') / 2
        ]))[()]
        A = np.real(reduce(np.dot, [
            self._E_psi, Lambda, self._rho_psi
        ]))[()] - B
        
        return (
            p(av_gate_fidelity(qp.Qobj(Lambdas[target])), 2),
            p(av_gate_fidelity(qp.Qobj(Lambda)), 2),
            A, B,
        )


