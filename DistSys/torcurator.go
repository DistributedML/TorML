package main

import (
"flag"
"fmt"
"net"
"os"
"golang.org/x/net/proxy"
"github.com/DistributedClocks/GoVector/govec"
)

var LOCAL_HOST string = "127.0.0.1:5005"
var ONION_HOST string = "4255ru2izmph3efw.onion:5005"
var TOR_PROXY string = "127.0.0.1:9150"

var (
  name string
  modelName string
  isLocal bool
)

type MessageData struct {
  Type          string
  SourceNode    string
  ModelId       string
  Key           string
  NumFeatures   int
  MinClients    int
}

func main() {

	parseArgs()
	logger := govec.InitGoVector(name, name)
	torDialer := getTorDialer()

  sendCurateMessage(logger, torDialer)

}

func parseArgs() {
  flag.Parse()
  inputargs := flag.Args()
  if len(inputargs) < 2 {
    fmt.Println("USAGE: go run torcurator.go curatorName modelName")
    os.Exit(1)
  }

  name = inputargs[0]
  modelName = inputargs[1]

  if len(inputargs) > 2 {
    fmt.Println("Running locally.")
    isLocal = true
  }

  fmt.Println("Done parsing args.")

}

func getTorDialer() proxy.Dialer {

  if isLocal {
    return nil
  }

  // Create proxy dialer using Tor SOCKS proxy
  torDialer, err := proxy.SOCKS5("tcp", TOR_PROXY, nil, proxy.Direct)
  checkError(err)
  return torDialer

}

func sendCurateMessage(logger *govec.GoLog, torDialer proxy.Dialer) int {

  conn, err := getServerConnection(torDialer)
  checkError(err)

  fmt.Println("TOR Dial Success!")

  var msg MessageData
  msg.Type = "curator"
  msg.SourceNode = name
  msg.ModelId = modelName
  msg.Key = ""
  msg.NumFeatures = 7840
  msg.MinClients = 7

  fmt.Println(msg)
  outBuf := logger.PrepareSend("Sending packet to torserver", msg)
      
  _, errWrite := conn.Write(outBuf)
  checkError(errWrite)
  
  inBuf := make([]byte, 2048)
  n, errRead := conn.Read(inBuf)
  checkError(errRead)

  var incomingMsg int
  logger.UnpackReceive("Received Message from server", inBuf[0:n], &incomingMsg)

  conn.Close()

  return incomingMsg

}

func getServerConnection(torDialer proxy.Dialer) (net.Conn, error) {

  var conn net.Conn
  var err error

  if torDialer != nil {
    conn, err = torDialer.Dial("tcp", ONION_HOST)
  } else {
    conn, err = net.Dial("tcp", LOCAL_HOST)
  }

  return conn, err

}

// Error checking function
func checkError(err error) {
  if err != nil {
    fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
    os.Exit(1)
  }
}
