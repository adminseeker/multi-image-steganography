FROM ubuntu

RUN apt update -y\ 
&& apt install python2 -y\
&& apt install wget -y\
&& wget https://bootstrap.pypa.io/pip/2.7/get-pip.py && python2 get-pip.py \
&& pip install tensorflow==1.15 keras==2.3.1 flask==1.1.4 flask-cors==3.0.10 numpy==1.16.6 pillow==6.2.2 uuid

WORKDIR /usr/share/app/backend

COPY . .

WORKDIR /usr/share/app/backend/src

EXPOSE 5000

CMD ["python2","-m","flask","run","--host=0.0.0.0"]

