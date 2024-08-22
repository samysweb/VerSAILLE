# VerSAILLE & Mosaic

[![container build](https://github.com/samysweb/VerSAILLE/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/samysweb/VerSAILLE/actions/workflows/main.yml)

Welcome to the teaching materials repository for VerSAILLE and Mosaic.
This repository is meant to provide resources for learning about [VerSAILLE & Mosaic](https://arxiv.org/abs/2402.10998) --- a combination of techniques that admits using the rigorous mathematical foundations of [differential dynamic logic](https://link.springer.com/book/10.1007/978-3-319-63588-0) ($d\mathcal{L}$) to reason about neural network control systems.

Let us start with some terminology:
- **VerSAILLE** is the theoretical framework that helps us to come up with correct-by-construction verification properties for Neural Networks deployed in control systems.
- **Mosaic** is an abstract framework for lifting off-the-shelf Open-Loop Neural Network Verification Tools to the task of verifying polynomial arithmetic specifications of arbitrary propositional structure. The framework preserves a verifier's completeness.
- $\mathbf{N^3V}$ is a concrete implementation of the Mosaic framework for $\mathrm{ReLU}$ neural networks and internally uses [nnenum](https://github.com/stanleybak/nnenum/).

## Getting Started
1. Please follow the [installation instructions](INSTALLATION.md)

## Literature
To learn more about the topics covered in this repository, there exist various pieces of literature:
- To learn about **differential dynamic logic** and **KeYmaera X**:
  - [Textbook by André Platzer](https://link.springer.com/book/10.1007/978-3-319-63588-0)
  - [Short-Form Overview](https://arxiv.org/pdf/1910.11232)
  - [KeYmaera X Tutorial](https://keymaerax.org/Xtutorial.html)
- To learn about **ModelPlex** (the monitoring technology that VerSAILLE is built upon):
  - [Paper](https://link.springer.com/article/10.1007/s10703-016-0241-z)
  - [Textbook Chapter](https://link.springer.com/chapter/10.1007/978-3-319-63588-0_19)
- To learn about VerSAILLE & Mosaic:
  - [Paper](https://arxiv.org/abs/2402.10998)
  - [Slides from SAIV'24](documents/Teuber-SAIV-2024.pdf)



## Support
If you struggle with any of the materials offered in this repository, please reach out to [Samuel Teuber](https://teuber.dev/).