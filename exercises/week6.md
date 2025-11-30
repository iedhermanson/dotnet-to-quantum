# Week 6 – From Math to Quantum Programming

## Goal

Transition from mathematical simulation to real quantum code.

## Concepts

- Mapping math concepts to Q# syntax
- Quantum operations in Q#:
  - `H(qubit)`: Hadamard gate
  - `CNOT(control, target)`: Controlled-NOT
  - `M(qubit)`: Measurement
  - `X(qubit)`: Pauli-X (reset)
- Building quantum algorithms:
  - Superposition creation
  - Entanglement generation
  - Bell state preparation
- Quantum simulators vs. real quantum hardware
- Measurement statistics and randomness

## Practice

1. Install Microsoft Quantum Development Kit (QDK)
2. Implement a Bell state in Q#
3. Run on quantum simulator (100+ shots)
4. Compare Q# measurement results with .NET mathematical simulation
5. Observe entanglement: measurements are always correlated

## Key Insights

- **.NET simulation**: Shows exact state vector amplitudes
- **Q# execution**: Shows measurement outcomes (probabilistic)
- **Comparison**: Validates that math matches quantum physics
- **Entanglement signature**: |01⟩ and |10⟩ never occur in Bell state

## Resources

### Video & Tutorials

- [Your First Quantum Program](https://www.youtube.com/watch?v=sjINVV2xOow) (Microsoft Quantum)
- [Q# Tutorial Series](https://www.youtube.com/playlist?list=PLOEAjVBNUcZjGPJM3L_WgZJ_p9jJ6M7Tn) (Microsoft Developer)
- [Coding with Qiskit](https://www.youtube.com/playlist?list=PLOFEBzvs-VvrXTMy5Y2IqmSaUjfnhvBHR) (Qiskit - similar concepts in Python)
- [Quantum Computing Crash Course](https://www.youtube.com/watch?v=X2q1PuI2RFI) (John Preskill / Quantum Magazine)

### Interactive & Practice

- [Microsoft Learn: Q# Fundamentals](https://learn.microsoft.com/en-us/training/paths/quantum-computing-fundamentals/) ⭐
- [Q# Language Documentation](https://learn.microsoft.com/en-us/azure/quantum/user-guide/)
- [Azure Quantum](https://azure.microsoft.com/en-us/products/quantum) (free credits available)
- [Quantum Katas](https://github.com/microsoft/QuantumKatas) (interactive Q# exercises) ⭐
- [Quirk](https://algassert.com/quirk) (test your Q# circuits visually first)

### Communities & Forums

- [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/)
- [r/QuantumComputing](https://www.reddit.com/r/QuantumComputing/)
- [Microsoft Q# Community](https://github.com/microsoft/Quantum/discussions)
