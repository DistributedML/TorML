#!/bin/bash

#install golang
#go to home directory
cd
chmod -R ug+rw go/

sudo apt-get update
#install git
sudo apt-get install git
#install mercurial
sudo apt-get install mercurial
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

echo "" >> .bashrc
echo "#export go path" >> .bashrc
echo "export PATH=$PATH:/usr/local/go/bin" >> .bashrc

#make root directory and set GOPATH
mkdir go
export GOPATH=$HOME/go
echo "" >> .profile
echo "#set GOPATH" >> .profile
echo "export GOPATH=$HOME/go" >> .profile

echo "" >> .profile
echo "#set GOPATH" >> .bashrc
echo "export GOPATH=$HOME/go" >> .bashrc

#export workspace bin
export PATH=$PATH:$GOPATH/bin
echo "" >> .profile
echo "#set local bin" >> .profile
echo "export PATH=$PATH:$GOPATH/bin" >> .profile

echo "" >> .bashrc
echo "#set local bin" >> .bashrc
echo "export PATH=$PATH:$GOPATH/bin" >> .bashrc
#Go should be installed now

#clone the TorMentor Repository
echo "Installing TorML"
#go get github.com/DistributedML/TorML
#TEMP
go get github.com/wantonsolutions/TorML
#Install dependencies
echo "Installing Dependencies"
echo "DistributedClocks..."
go get github.com/DistributedClocks/GoVector
echo "mat64..."
go get github.com/gonum/matrix/mat64
echo "pkg-config"
sudo apt-get install pkg-config
echo "go python"
go get github.com/sbinet/go-python




echo "installing pip"
sudo apt install python-pip
##TODO Probably install python
echo "pandas"
pip install pandas
echo "emcee"
pip install emcee
echo "utils"
pip install utils
echo "pdb"
pip install pdb
echo "matplot-lib"
pip install matplotlib


#install tor
echo "installing tor"
#sudo echo "deb http://deb.torproject.org/torproject.org stretch main" >> /etc/apt/sources.list
#sudo echo "deb-src http://deb.torproject.org/torproject.org stretch main" >> /etc/apt/sources.list

#gpg --keyserver keys.gnupg.net --recv A3C4F0F979CAA22CDBA8F512EE8CBC9E886DDD89
#gpg --export A3C4F0F979CAA22CDBA8F512EE8CBC9E886DDD89 | sudo apt-key add -
#apt update
#apt install tor deb.torproject.org-keyring
#browser TODO remove
#https://www.torproject.org/dist/torbrowser/7.0.6/tor-browser-linux64-7.0.6_en-US.tar.xz

sudo echo "deb http://deb.torproject.org/torproject.org trusty main" >> /etc/apt/sources.list
sudo echo "deb-src http://deb.torproject.org/torproject.org trusty main" >> /etc/apt/sources.list

gpg --keyserver keys.gnupg.net --recv 886DDD89
gpg --export A3C4F0F979CAA22CDBA8F512EE8CBC9E886DDD89 | sudo apt-key add -

apt-get update
apt-get install tor deb.torproject.org-keyrin

sudo killall tor

#give proper permissions
cd
chmod -R ug+rw go

