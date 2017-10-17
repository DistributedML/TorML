# TorMentor
A collaborative machine learning framework that operates through Tor.

# Instructions
Everything is stored in two directories: DistSys (in Golang, all the distributed systems logic) and ML (in Python, all the ML logic)
We use go-python to link the two.

# Example run:

On server machine:
`go run torserver.go`

On client machines:
`go run torcurator.go c1 study`
`go run torclient.go h1 study credit1`
`go run torclient.go h2 study credit2`
...

