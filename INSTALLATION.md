# Installation

To use the resources presented in this repository, we require some technical preparations:

- Installing KeYmaera X:  
  See [installation instructions here](https://keymaerax.org/download.html#quick)
- Installing $N^3V$ (the nonlinear neural network verifier):  
  To simplify the setup, we offer a Docker Container which comes with all required dependencies. To use the Docker Container:
  - Install Docker:
    - [Linux](https://docs.docker.com/desktop/install/linux-install/)
    - [Apple](https://docs.docker.com/desktop/install/mac-install/)
    - [Windows](https://docs.docker.com/desktop/install/windows-install/)
  - Run `docker run -p 8888:8888 samweb/versaille`
  - After starting the container, open the following link in your browser:  
   [http://127.0.0.1:8888/lab?token=versaille](http://127.0.0.1:8888/lab?token=versaille)
  - This should open a Jupyter Notebook Instance where [Intro.ipynb](http://127.0.0.1:8888/lab/tree/Training.ipynb?token=versaille) tells you about the next steps.