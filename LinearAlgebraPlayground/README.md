# .NET to Quantum – Linear Algebra & Q# Journey (no deadlines)

This repository tracks a flexible plan to learn just enough linear algebra to confidently step into quantum programming. You’ll use C#/.NET (Math.NET Numerics) for simulation and then implement basic Q# programs.

## Setup

Prereqs:
- .NET SDK (7 or later recommended): https://dotnet.microsoft.com/
- VS Code + extensions:
  - C#
  - GitHub Copilot
  - Q# (for Week 6)
- Git

Copilot:
- Ensure your Copilot subscription is active: https://github.com/github-copilot/pro/signup
- In VS Code, sign in to GitHub and enable Copilot inline suggestions (Settings → “inline suggestions” and Copilot enabled).

## Repo Structure

- `DotNetToQuantum.sln` – solution file
- `LinearAlgebraPlayground/` – C# console app with exercises organized by week
- `exercises/` – Optional scripts and notes by week
- `qsharp/` – Q# project for Week 6 (Bell state)
- `.vscode/` – Optional VS Code settings

## How to run

```bash
dotnet build
dotnet run --project LinearAlgebraPlayground
```

## Learning Plan (no deadlines)

Week 1 – Foundations & Mindset
- Goal: Build intuition for vectors, matrices, and complex numbers.
- Concepts: Scalars, vectors, matrices, complex numbers, basic operations.
- Practice in .NET:
  - Install Math.NET Numerics.
  - Create and manipulate vectors/matrices.
  - Implement vector addition, scalar multiplication, and dot products.
- Resources:
  - Essence of Linear Algebra (3Blue1Brown, videos 1–3)
  - Math.NET Numerics docs (Vector/Matrix basics)

Week 2 – Quantum State Representation
- Goal: Understand how qubits are represented mathematically.
- Concepts:
  - Column vectors for states (|0⟩, |1⟩)
  - Complex amplitudes and normalization
  - Inner products and probability interpretation
- Practice in .NET:
  - Represent |0⟩ and |1⟩ as vectors.
  - Write a function to check if a vector is normalized.
- Resources:
  - Jim Hefferon’s Linear Algebra (Ch. 1–2)
  - Microsoft Quantum Docs: Qubit basics

Week 3 – Quantum Gates as Matrices
- Goal: Learn how quantum operations are matrix transformations.
- Concepts:
  - Matrix multiplication
  - Unitary matrices (definition & properties)
  - Pauli-X, Y, Z, Hadamard
- Practice in .NET:
  - Implement gates as matrices and apply them to qubit vectors.
  - Verify unitarity: U†U = I.
- Resources:
  - Essence of Linear Algebra (matrix multiplication)
  - Microsoft Quantum Katas: Basic Gates

Week 4 – Multi-Qubit Systems
- Goal: Work with tensor (Kronecker) products to represent multi-qubit states.
- Concepts:
  - Tensor product
  - Building multi-qubit states from single qubits
  - Multi-qubit gates (CNOT)
- Practice in .NET:
  - Implement a tensor product function.
  - Simulate a 2-qubit system and apply CNOT.
- Resources:
  - Math.NET Numerics (Kronecker product)
  - Quantum Katas: Multi-Qubit Gates

Week 5 – Measurement & Eigen Concepts
- Goal: Understand measurement mathematically.
- Concepts:
  - Eigenvalues/eigenvectors in quantum measurement
  - Projection operators
  - Probability of outcomes
- Practice in .NET:
  - Simulate measurement probabilities.
  - Find eigenvalues/eigenvectors of Pauli-Z.
- Resources:
  - Essence of Linear Algebra (Eigenvectors/Eigenvalues)
  - Quantum Katas: Measurements

Week 6 – From Math to Quantum Programming
- Goal: Transition from simulation to real quantum code.
- Concepts:
  - Mapping math concepts to Q# syntax
  - Simple quantum algorithms (superposition, entanglement)
- Practice:
  - Install QDK (Quantum Development Kit) for .NET.
  - Implement a Bell state in Q#.
  - Compare results with your .NET simulation.
- Resources:
  - Microsoft Learn: Q# Fundamentals
  - Azure Quantum (optional)

## Using Copilot Effectively

- Inline completion:
  - Write a comment describing the function (e.g., “Compute dot product of two vectors”) then start typing the method signature; Copilot will suggest implementation.
- Chat:
  - Ask in Copilot Chat: “Explain why Hadamard is unitary.” or “Generate tests for Normalize function.”
- Prompts:
  - Be explicit about types (Complex vectors, double matrices).
  - Include examples. Copilot learns from context in the file.

## License

MIT