package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"net"
	"os"
	"time"
	"golang.org/x/net/proxy"
	"github.com/DistributedClocks/GoVector/govec"
	"github.com/sbinet/go-python"
)

/*
	An attempt at Tor via TCP. 
*/
var LOCAL_HOST string = "127.0.0.1:5005"
var ONION_HOST string = "4255ru2izmph3efw.onion:5005"
var TOR_PROXY string = "127.0.0.1:9150"

type MessageData struct {
	Type 		  string
	SourceNode    string
	Study         StudyInfo
	Deltas 		  []float64
}

type StudyInfo struct {
	StudyId	 	string
	NumFeatures int
}

var (
	name 			string
	studyName  		string
	datasetName 	string
	isLocal 		bool
	study 			StudyInfo
	logModule      	*python.PyObject
	logInitFunc     *python.PyObject
	logPrivFunc     *python.PyObject
	numFeatures     *python.PyObject
	pulledGradient  []float64
)

func init() {
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

func pyInit(datasetName string) {

	sysPath := python.PySys_GetObject("path")
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("./"))
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("../ML/code"))

	logModule = python.PyImport_ImportModule("logistic_model")
	logInitFunc = logModule.GetAttrString("init")
	logPrivFunc = logModule.GetAttrString("privateFun")
	numFeatures = logInitFunc.CallFunction(python.PyString_FromString(datasetName))

  	study.StudyId = studyName
  	study.NumFeatures = python.PyInt_AsLong(numFeatures)
  	pulledGradient = make([]float64, study.NumFeatures)

}

func main() {

	parseArgs()
	logger := govec.InitGoVector(name, name)
	torDialer := getTorDialer()

	// Initialize the python side
	pyInit(datasetName)

	fmt.Printf("Joining study %s \n", studyName)
  	joined := sendJoinMessage(logger, torDialer)

  	if joined == 0 {
  		fmt.Println("Could not join.")
  		return
  	}
  	
  	sendGradMessage(logger, torDialer, pulledGradient, true)

  	for i := 0; i < 20000; i++ { 
    	sendGradMessage(logger, torDialer, pulledGradient, false)
  	}

  	heartbeat(logger, torDialer)
  	fmt.Println("The end")

}

func heartbeat(logger *govec.GoLog, torDialer proxy.Dialer) {

	for {

		time.Sleep(30 * time.Second)		

		conn, err := getServerConnection(torDialer)
		if err != nil {
			fmt.Println("Got a Dial failure, retrying...")
			continue
		}

		var msg MessageData
		msg.Type = "beat"
		msg.SourceNode = name
		msg.Deltas = nil

		outBuf := logger.PrepareSend("Sending packet to torserver", msg)
		
		_, err = conn.Write(outBuf)
		if err != nil {
			fmt.Println("Got a Conn Write failure, retrying...")
			conn.Close()
			continue
		}
		
		inBuf := make([]byte, 512)
		n, errRead := conn.Read(inBuf)
		if errRead != nil {
			fmt.Println("Got a Conn Read failure, retrying...")
			conn.Close()
			continue
		}

		var reply int
		logger.UnpackReceive("Received Message from server", inBuf[0:n], &reply)

		fmt.Println("Send heartbeat success")
		conn.Close()

	}	

}

func parseArgs() {
	flag.Parse()
	inputargs := flag.Args()
	if len(inputargs) < 3 {
		fmt.Println("USAGE: go run torclient.go nodeName studyName datasetName isLocal")
		os.Exit(1)
	}
	name = inputargs[0]
	studyName = inputargs[1]
	datasetName = inputargs[2]

	if len(inputargs) > 3 {
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

func sendGradMessage(logger *govec.GoLog, 
	torDialer proxy.Dialer, 
	globalW []float64, 
	bootstrapping bool) int {
	
	completed := false

	// prevents the screen from overflowing and freezing
	time.Sleep(10 * time.Millisecond)		

	for !completed {

		conn, err := getServerConnection(torDialer)
		if err != nil {
			fmt.Println("Got a Dial failure, retrying...")
			continue
		}

		var msg MessageData
		if !bootstrapping {
			msg.Type = "grad"
			msg.SourceNode = name
			msg.Study = study
			msg.Deltas, err = oneGradientStep(globalW)

			if err != nil {
				fmt.Println("Got a GoPython failure, retrying...")
				conn.Close()
				continue
			}
		} else {
			msg.Type = "grad"
			msg.SourceNode = name
			msg.Study = study
			msg.Deltas = make([]float64, study.NumFeatures)
			bootstrapping = false
		}

		outBuf := logger.PrepareSend("Sending packet to torserver", msg)
		
		_, err = conn.Write(outBuf)
		if err != nil {
			fmt.Println("Got a conn write failure, retrying...")
			conn.Close()
			continue
		}
		
		inBuf := make([]byte, 2048)
		n, errRead := conn.Read(inBuf)
		if errRead != nil {
			fmt.Println("Got a reply read failure, retrying...")
			conn.Close()
			continue
		}

		var incomingMsg []float64
		logger.UnpackReceive("Received Message from server", inBuf[0:n], &incomingMsg)

		conn.Close()

		pulledGradient = incomingMsg
		completed = true

	}

	return 1

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

func sendJoinMessage(logger *govec.GoLog, torDialer proxy.Dialer) int {

	conn, err := getServerConnection(torDialer)
	checkError(err)

	fmt.Println("TOR Dial Success!")

	var msg MessageData
    msg.Type = "join"
    msg.SourceNode = name
    msg.Study = study
    msg.Deltas = make([]float64, study.NumFeatures)

    outBuf := logger.PrepareSend("Sending packet to torserver", msg)
    	
	_, errWrite := conn.Write(outBuf)
	checkError(errWrite)
	
	inBuf := make([]byte, 512)
	n, errRead := conn.Read(inBuf)
	checkError(errRead)

	var incomingMsg int
	logger.UnpackReceive("Received Message from server", inBuf[0:n], &incomingMsg)

	conn.Close()

	return incomingMsg

}

func oneGradientStep(globalW []float64) ([]float64, error) {
	
	argArray := python.PyList_New(len(globalW))

	for i := 0; i < len(globalW); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(globalW[i]))
	}

	result := logPrivFunc.CallFunction(python.PyInt_FromLong(1), argArray, 
		python.PyInt_FromLong(10))
	
	// Convert the resulting array to a go byte array
	pyByteArray := python.PyByteArray_FromObject(result)
	goByteArray := python.PyByteArray_AsBytes(pyByteArray)

	var goFloatArray []float64
	size := len(goByteArray) / 8

	for i := 0; i < size; i++ {
		currIndex := i * 8
		bits := binary.LittleEndian.Uint64(goByteArray[currIndex : currIndex+8])
		aFloat := math.Float64frombits(bits)
		goFloatArray = append(goFloatArray, aFloat)
	}
	
	return goFloatArray, nil

}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}
