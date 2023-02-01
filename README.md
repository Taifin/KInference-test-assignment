# KInference-test-assignment

This is a test assignment for JetBrains KInference project to perform inference process for a trained neural network.
This program uses Strassen's algorithm for large computational graphs.

### Usage

Run `./gradlew installDist` and find executable in `build/install/KInference-test-assigment/bin` directory.

Example:

```bash
./gradlew installDist
cd build/install/KInference-test-assigment/bin
./KInference-test-assigment testInput.txt -s 100 -s 10
```

Above example runs calculations using inputs from `testInput.txt` file with dense layers of size 100 and 10.

### Program arguments

* `source`: source file with input data
* `-o`, `--out`: redirect input into given file.
* `-g`, `--gen`: ignore input file and generate random input vector of given size (please, pass any name to source
  anyway)
* `-s`, `--size`: specify size of a dense layer. Call this option once to specify size of hidden layer, and twice to
  specify both.
* `--help`: print help message and exit

### Dependencies

* `xenomachina-argparser:2.0.7`

### Known problems

Current implementation of Strassen's algorithm is very slow, especially on small
matrices. `simpleMultiplicationThreshold` property of `Matrix` class describes minimal size of square matrix, below
which naive multiplication will be used.
