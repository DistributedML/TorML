# TorMentor
A collaborative machine learning framework that operates through Tor.

# Publications

* TorMentor is described as part of a paper on *brokered learning*: 
  
  [*Brokered Agreements in Multi-Party Machine Learning*](https://www.cs.ubc.ca/~bestchai/papers/apsys19-brokering.pdf). Clement Fung, Ivan Beschastnikh. APSys 2019
    ```
    @inproceedings{Fung2019,
      author = {Fung, Clement and Beschastnikh, Ivan},
      title = {{Brokered Agreements in Multi-Party Machine Learning}},
      year = {2019},
      doi = {10.1145/3343737.3343744},
      booktitle = {Proceedings of the 10th ACM SIGOPS Asia-Pacific Workshop on Systems (APSys)},
      pages = {69–75},
    }
    ```

* TorMentor is further detailed in an Arxiv paper:
  
  [Dancing in the Dark: Private Multi-Party Machine Learning in an Untrusted Setting](https://arxiv.org/abs/1811.09712v2). Clement Fung, Jamie Koerner, Stewart Grant, Ivan Beschastnikh. 
    ```
    @article{DBLP:journals/corr/abs-1811-09712,
      author    = {Clement Fung and Jamie Koerner and Stewart Grant and Ivan Beschastnikh},
      title     = {{Dancing in the Dark: Private Multi-Party Machine Learning in an Untrusted Setting}},
      journal   = {CoRR},
      volume    = {abs/1811.09712},
      year      = {2018},
    }
    ```


# Talks and Presentations 
May 11th - UBC Cybersecurity Summit: [Talk](presentations/CyberSecuritySummit-05-11-2018.pdf) [Poster](presentations/TorMentor-Poster-Print.pdf)

# Instructions
Everything is stored in two directories: DistSys (in Golang, all the distributed systems logic) and ML (in Python, all the ML logic)  
We use [go-python](https://github.com/sbinet/go-python) to link the two.

Before running the system, all machines need to have tor installed.   
https://www.torproject.org/projects/torbrowser.html.en 

If your ISP blocks Tor, a mirror is available here:
https://thetorproject.github.io/gettor/ 

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

On each client machines:  
`go run torcurator.go curatorName minClients isLocal`  
`go run torclient.go node1 datasetName epsilon isLocal`  
`go run torclient.go node2 datasetName epsilon isLocal`  
`go run torclient.go node3 datasetName epsilon isLocal`  

A good test set is the credit dataset: 
and use: credit1, credit2, creditbad.... etc.

An example deployment, 3 clients, with epsilon=1, running through Tor:
`go run torserver.go`  
`go run torcurator.go c1 3`  
`go run torclient.go h1 credit1 1`  
`go run torclient.go h2 credit2 1`  
`go run torclient.go h3 credit3 1`  


