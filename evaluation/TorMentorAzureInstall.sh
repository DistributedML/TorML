#!/bin/bash

#install golang
#go to home directory
cd

#download go binary
wget https://storage.googleapis.com/golang/go1.7.4.linux-amd64.tar.gz

#unzip and remove
sudo tar -C /usr/local -xzf go1.7.4.linux-amd64.tar.gz
rm go1.7.4.linux-amd64.tar.gz

#export path
export PATH=$PATH:/usr/local/go/bin
echo "" >> .profile
echo "#export go path" >> .profile
echo "export PATH=$PATH:/usr/local/go/bin" >> .profile

#make root directory and set GOPATH
mkdir go
export GOPATH=$HOME/go
echo "" >> .profile
echo "#set GOPATH" >> .profile
echo "export GOPATH=$HOME/go" >> .profile

#export workspace bin
export PATH=$PATH:$GOPATH/bin
echo "" >> .profile
echo "#set local bin" >> .profile
echo "export PATH=$PATH:$GOPATH/bin" >> .profile
#Go should be installed now

#clone the TorMentor Repository
echo "Installing TorML"
go get github.com/DistributedML/TorML
#Install dependencies
echo "Installing Dependencies"
echo "mat64..."
go get github.com/gonum/matrix/mat64
echo "go python"
go get github.com/sbinet/go-python

##TODO Probably install python
echo "pkg-config"
apt-get install -y pkg-config
echo "pip"
apt install python-pip
echo "pandas"
pip install pandas
echo "emcee"
pip install emcee
echo "utils"
pip install utils


#install tor
echo "installing tor"
echo "deb http://deb.torproject.org/torproject.org stretch main" >> /etc/apt/sources.list
echo "deb-src http://deb.torproject.org/torproject.org stretch main" >> /etc/apt/sources.list

gpg --keyserver keys.gnupg.net --recv A3C4F0F979CAA22CDBA8F512EE8CBC9E886DDD89
gpg --export A3C4F0F979CAA22CDBA8F512EE8CBC9E886DDD89 | sudo apt-key add -
apt update
apt install tor deb.torproject.org-keyring