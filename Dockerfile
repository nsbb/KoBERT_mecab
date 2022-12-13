FROM nvidia/cuda:11.4.0-devel-ubuntu18.04
ENV TZ=Asia/Seoul
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install apt-utils -y
RUN apt-get update
RUN apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev -y
RUN apt-get install wget -y
RUN mkdir /toy
WORKDIR /opt
RUN wget https://www.python.org/ftp/python/3.9.6/Python-3.9.6.tgz
RUN tar xzf Python-3.9.6.tgz
RUN cd Python-3.9.6 \
    && ./configure --enable-optimizations \
    && make altinstall \
    && update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.9 1
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py
RUN apt-get update
RUN apt-get install vim git -y 
RUN git clone https://github.com/nsbb/change_color.git
RUN cd change_color \
    && chmod +x change_color_docker \
    && source change_color_docker
WORKDIR /toy
RUN pip3 install konlpy JPype1
RUN wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
RUN tar zxfv mecab-0.996-ko-0.9.2.tar.gz
RUN cd mecab-0.996-ko-0.9.2 \
    && ./configure \
    && make \
    && make check \
    && make install \
    && ldconfig
RUN apt-get install automake libtool -y
RUN wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
RUN tar zxfv mecab-ko-dic-2.1.1-20180720.tar.gz
RUN cd mecab-ko-dic-2.1.1-20180720 \
    && ./autogen.sh \
    && ./configure \
    && make \
    && make install
ADD user-custom_v6.csv /toy/mecab-ko-dic-2.1.1-20180720/user-dic/user-custom_v6.csv
RUN cd mecab-ko-dic-2.1.1-20180720 \
    && ./tools/add-userdic.sh \
    && make clean \
    && make install
RUN pip3 install mecab-python pandas openpyxl tqdm jupyter 
RUN git clone https://github.com/SKTBrain/KoBERT.git
RUN cd KoBERT \
    && pip3 install -r requirements.txt \
    && pip3 install .
RUN pip3 install -U scikit-learn scipy matplotlib
RUN pip3 install torch==1.13.0
RUN pip3 install termcolor
ADD LG_data /toy/LG_data
ADD gg_train_v2.py /toy/gg_train_v4.py
ADD gg_test_v3.py /toy/gg_test_v3.py
ADD LG_model /toy/LG_model
ADD .bashrc /root/.bashrc
RUN source /root/.bashrc
ADD transforms.py /usr/local/lib/python3.9/site-packages/gluonnlp/data/transforms.py
ADD logo.sh /toy/logo.sh
CMD python3 gg_test_v3.py
