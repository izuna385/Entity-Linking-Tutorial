FROM continuumio/anaconda3:2019.03

RUN apt-get -y update && apt-get -y install vim && apt -y upgrade
RUN apt install -y build-essential
RUN pip install --upgrade pip && pip install autopep8
RUN apt install unzip
ARG project_dir=/projects
WORKDIR $project_dir
RUN conda update -n base -c defaults conda
RUN conda create -n allennlp python=3.7
RUN conda init bash
RUN . /root/.bashrc && conda activate allennlp
ADD requirements.txt .
RUN pip install -r requirements.txt
RUN conda install faiss-cpu -c pytorch
COPY . $project_dir
CMD ["/bin/bash"]