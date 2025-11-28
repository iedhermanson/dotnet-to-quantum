namespace BellState {
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Measurement;

    operation PrepareBellState() : Result[] {
        use qs = Qubit[2];
        H(qs[0]);
        CNOT(qs[0], qs[1]);

        // Measure in Z basis
        let results = [M(qs[0]), M(qs[1])];

        // Reset before release
        if (results[0] == One) { X(qs[0]); }
        if (results[1] == One) { X(qs[1]); }

        return results;
    }
}