# piml-stability


Physics-Informed Probabilistic Stability Lobe Diagram (SLD)

This project implements a Physics-Informed Machine Learning (PIML) framework for generating probabilistic Stability Lobe Diagrams (SLDs) in milling processes.

It combines:

Physics-based modeling (FDM / analytical approximation)
Machine learning (MLP surrogate models)
Bayesian inference
Monte Carlo simulation
Closed-loop experimental optimization
 Features
 Load stability boundary from Excel
 Generate labelled experimental data (stable vs chatter)
 Train surrogate neural networks (PyTorch-based)
 Perform Bayesian posterior estimation (MAP + Laplace)
 Monte Carlo-based probabilistic SLD generation
 Closed-loop adaptive testing using Expected Improvement (MRR)
 Fusion of physics + Bayesian maps
 High-quality visualization of stability regions
