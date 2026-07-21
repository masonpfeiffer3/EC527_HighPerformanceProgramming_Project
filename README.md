# High-Performance Machine Learning - *High-Performance Programming Course Project*
![HPP main image](https://github.com/masonpfeiffer3/EC527_HighPerformanceProgramming_Project/blob/main/pics/titleimage.png)
<br></br>
**Project Dates:** Apr - May 2026

## Description
- A highly-optimized Multilayer Perceptron (MLP) neural network built to classify grayscale handwritten digits
- Optimizations achieved a **156X** training + testing speedup over the serial baseline while maintaining a classification accuracy of **96.57%**
- The MLP was written in C, contained 4 layers + 80,000 parameters, and was optimized for the Intel i9-14900
- Optimizations included loop unrolling, AVX vectorization, OpenMP multithreading, and matrix operation fusion
- **Full report in repo root**

![HPP MLP image](https://github.com/masonpfeiffer3/EC527_HighPerformanceProgramming_Project/blob/main/pics/network.png)

## Tools
- Resources: 3Blue1Brown + Michael Nielsen's *Neural Networks and Deep Learning*
- Coding & Compiling: VSCode + GCC
- Optimization libraries: Intel AVX Vector extension + OpenMP Multithreading
- Claude Code


## Methodology
- Synthesized video and textbook resources to construct serial baseline backpropagation and feedforward algorithms within **3 days** from scratch
- Targeted optimization hotspots according to Amdahl’s Law by prioritizing the largest activation/weight/bias matrices and their functions
- Directed brainstorming sessions with partner, accelerating integration of serial/parallel optimizations
- Achieved **78X** speedup using techniques from class, then utilized Claude Code to identify 2 additional optimization hotspots, improving to **156X** speedup
- Recorded optimizations in 22-page final report and streamlined 9-min presentation to the full class

## Results
![final stats](https://github.com/masonpfeiffer3/EC527_HighPerformanceProgramming_Project/blob/main/pics/FinalStats.png)

## Build Instructions
This codebase consists of three separate code architectures:

- [optimized](https://github.com/masonpfeiffer3/EC527_HighPerformanceProgramming_Project/tree/main/optimized) contains our most efficient code. Navigate to the directory and compile/run on Linux systems with "make run".
- [serial_baseline](https://github.com/masonpfeiffer3/EC527_HighPerformanceProgramming_Project/tree/main/serial_baseline) contains our code derived from first principles. Navigate to the directory and compile/run on Linux systems with "make run".
- [python_reference_code](https://github.com/masonpfeiffer3/EC527_HighPerformanceProgramming_Project/tree/main/python_reference_code) contains the neural network from Michael Nielsen's "Deep Learning and Neural Networks". Navigate to the directory/src and compile/run on Linux systems with "python3 script.py".
