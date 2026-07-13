# High-Performance Machine Learning - *High-Performance Programming Course Project*
![HPP main image](https://github.com/masonpfeiffer3/EC527_HighPerformanceProgramming_Project/blob/main/pics/titleimage.png)
<br></br>
**Project Dates:** Apr - May 2026

## Description
- A highly-optimized Multilayer Perceptron (MLP) neural network built to classify grayscale handwritten digits
- Optimizations achieved a **156X** training + testing speedup over the serial baseline while maintaining a classification accuracy of **96.57%**
- The MLP was written in C, contained 4 layers + 80,000 parameters, and was optimized for the Intel i9-14900
- Optimizations included loop unrolling, AVX vectorization, OpenMP multithreading, and matrix operation fusion

## Tools
- Resources: 3Blue1Brown + Michael Nielsen's *Neural Networks and Deep Learning*
- Coding & Compiling: VSCode + GCC
- Optimization libraries: Intel AVX Vector extension + OpenMP Multithreading
- Claude Code


## Methodology
- Researched extensively Java Swing classes and GridBagLayout to program the UI
- Developed the grid rendering and zoom/pan functions from scratch through much trial and error
- Prevented errors by dummy-proofing all user inputs
- Managed ArrayLists for Parent and Child species with code to decide what species to place in a cell and to allow for editing/deleting species
- Designed my own buttons (DevButtons) built on JPanels that fade between colors on hover
- Accounted for fractional values in graphics with a double-to-int converter that takes a loop index input

## Results
![class heirarchy](https://github.com/ibyteibit/Game-of-Life/blob/main/pics/ClassHeirarchy.png)

## Build Instructions
This codebase consists of three separate code architectures:

- The "optimized" folder contains our most efficient code. Navigate to the directory and compile/run on Linux systems with "make run".
- The "serial_baseline" folder contains our code derived from first principles. Navigate to the directory and compile/run on Linux systems with "make run".
- The "python_reference_code" folder contains code taken from Michael Nielsen's "Deep Learning and Neural Networks". Navigate to the directory/src and compile/run on Linux systems with "python3 script.py".

Our report is included as a pdf file in the root.
