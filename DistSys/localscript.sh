#!/bin/sh

export GOPATH=/home/cfung/gospace

go run torcurator.go c1 8 l &
wait
go run torclient.go h1 credit/credit1 1 l &
go run torclient.go h2 credit/credit2 1 l &
go run torclient.go h3 credit/credit3 1 l &
go run torclient.go h4 credit/credit4 1 l &
go run torclient.go h5 credit/creditbad 1 l &
go run torclient.go h6 credit/creditbad 1 l &
go run torclient.go h7 credit/creditbad 1 l &
go run torclient.go h8 credit/creditbad 1 l &

