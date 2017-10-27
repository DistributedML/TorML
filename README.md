# TorMentor
A collaborative machine learning framework that operates through Tor.

# Instructions
Everything is stored in two directories: DistSys (in Golang, all the distributed systems logic) and ML (in Python, all the ML logic)  
We use [go-python](https://github.com/sbinet/go-python) to link the two.

Before running the system, all machines need to have tor installed.   
https://www.torproject.org/projects/torbrowser.html.en

Once Tor is installed, the service machine needs to open up the hidden service. Do this by modifying the torrc (config file).  
I found this file in: <tor_dir>/Browser/TorBrowser/Data/Tor

Add the following config parameters:
```
HiddenServiceDir <your_directory>  
HiddenServicePort 80 127.0.0.1:6767       # Port for the webinterface
HiddenServicePort 5005 127.0.0.1:5005     # Control port, used for joining and curating
HiddenServicePort 6000 127.0.0.1:6000     # Worker ports: By default we use from 6000 - 6010
HiddenServicePort 6001 127.0.0.1:6001  
HiddenServicePort etc... 127.0.0.1:etc...    
Log notice file /home/cfung/workspace/tor-browser_en-US/Browser/TorBrowser/Log/tor.log  
UseBridges 1
```
Add one HiddenServicePort for every connection you want the tor server to handle.

# Example:
On server machine:  
`go run torserver.go`

On client machines:  
`go run torcurator.go c1 studyName`  
`go run torclient.go h1 studyName dataFile`  
`go run torclient.go h2 studyName dataFile`  

A good test set is the credit dataset.
