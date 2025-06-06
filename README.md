# Nonlinear Response Spectrum Analysis (NRSA)

## 1. Introduction

**NRSA** is a Python program for **Nonlinear Response Spectrum Analysis (NRSA)** of single-degree-of-freedom (SDOF) systems. It supports three types of analyses: **Time History Analysis (THA)**, **Constant Ductility Analysis (CDA)** and **Constant Strength Analysis (CSA)**.

Users can define custom inelastic materials based on the OpenSees material library. Nonlinear response spectra can be obtained using either CDA or CSA, including the constant ductility spectrum (*R*-*μ*-*T* relationship), ductility demand spectrum, etc.

Due to Python being an interpreted language, in order to speed up the computation, the Newmark-β solver was developed using Cython. Multi-processing technology were also used to achieve parallel computing of multiple earthquake motions.

## 2. Usage

Run the example scripts `example_THA.py`, `example_CDA.py` and `example_CSA.py` to perform THA, CDA and CSA analyses, respectively.
