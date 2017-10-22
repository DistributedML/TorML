package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"net"
	"os"
	"time"
	"strconv"
	"strings"
	"golang.org/x/net/proxy"
	"github.com/DistributedClocks/GoVector/govec"
	"github.com/sbinet/go-python"
)

/*
	An attempt at Tor via TCP. 
*/
var LOCAL_HOST 		string = "127.0.0.1:"
var ONION_HOST 		string = "4255ru2izmph3efw.onion:"
var CONTROL_PORT 	int    = 5005
var TOR_PROXY 		string = "127.0.0.1:9150"

type MessageData struct {
	Type   		  string
	SourceNode    string
	ModelId       string
	Key           string
	NumFeatures   int
	MinClients    int
}

// Schema for data used in gradient updates
type GradientData struct {
	ModelId 	  string
	Key			  string
	Deltas 		  []float64
}

var (
	
	name 			string
	modelName  		string
	datasetName 	string
	numFeatures     int
	minClients      int
	isLocal			bool
	puzzleKey		string
	pulledGradient  []float64
	torHost			string
	torAddress		string
	epsilon			float64

	victimModel		[]float64
	myLastModel		[]float64
	myLastUpdate	[]float64

	pyLogModule       *python.PyObject
	pyLogInitFunc     *python.PyObject
	pyLogPrivFunc     *python.PyObject
	pyNumFeatures     *python.PyObject
	

)

func writeModel(weights []float64) {

	file, err := os.Create("victimModel.csv")
	checkError(err)
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	st = strings.Fields(strings.Trim(fmt.Sprint(weights), "[]"))
	writer.Write(st)       

	fmt.Println("Victim written.")

}

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

	pyLogModule = python.PyImport_ImportModule("logistic_model")
	pyLogInitFunc = pyLogModule.GetAttrString("init")
	pyLogPrivFunc = pyLogModule.GetAttrString("privateFun")
	pyNumFeatures = pyLogInitFunc.CallFunction(python.PyString_FromString(datasetName), python.PyFloat_FromDouble(epsilon))

  	numFeatures = python.PyInt_AsLong(pyNumFeatures)
  	minClients = 5
  	pulledGradient = make([]float64, numFeatures)
  	victimModel = make([]float64, numFeatures)
  	myLastModel = make([]float64, numFeatures)
  	myLastUpdate = make([]float64, numFeatures)

  	fmt.Printf("Sucessfully pulled dataset. Features: %d\n", numFeatures)
}

func main() {

	parseArgs()
	logger := govec.InitGoVector(name, name)
	torDialer := getTorDialer()

	// Initialize the python side
	pyInit(datasetName)

	fmt.Printf("Joining model %s \n", modelName)
  	joined := sendJoinMessage(logger, torDialer)

  	if joined == 0 {
  		fmt.Println("Could not join.")
  		return
  	}
  	
  	sendGradMessage(logger, torDialer, pulledGradient, true)

  	for i := 0; i < 200000; i++ { 
    	sendGradMessage(logger, torDialer, pulledGradient, false)
    	if i % 100 == 0 {
    		writeModel(victimModel)
    	}
  	}

  	heartbeat(logger, torDialer)
  	fmt.Println("The end")

}

func heartbeat(logger *govec.GoLog, torDialer proxy.Dialer) {

	for {

		time.Sleep(30 * time.Second)		

		conn, err := getServerConnection(torDialer, false)
		if err != nil {
			fmt.Println("Got a Dial failure, retrying...")
			continue
		}

		var msg MessageData
		msg.Type = "beat"
		msg.SourceNode = name
		msg.ModelId = modelName
		msg.Key = puzzleKey
		msg.NumFeatures = numFeatures
		msg.MinClients = minClients

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
	if len(inputargs) < 4 {
		fmt.Println("USAGE: go run torclient.go nodeName studyName datasetName epsilon isLocal")
		os.Exit(1)
	}
	
	name = inputargs[0]
	modelName = inputargs[1]
	datasetName = inputargs[2]
	
	var err error
	epsilon, err = strconv.ParseFloat(inputargs[3], 64)

	if err != nil {
		fmt.Println("Must pass a float for epsilon.")
		os.Exit(1)
	}

	torHost = ONION_HOST

	fmt.Printf("Name: %s\n", name)
	fmt.Printf("Study: %s\n", modelName)
	fmt.Printf("Dataset: %s\n", datasetName)

	if len(inputargs) > 4 {
		fmt.Println("Running locally.")
		isLocal = true
		torHost = LOCAL_HOST
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
	time.Sleep(100 * time.Millisecond)		

	for !completed {

		conn, err := getServerConnection(torDialer, true)
		if err != nil {
			fmt.Println("Got a Dial failure, retrying...")
			time.Sleep(100 * time.Millisecond)		
			continue
		}

		var msg GradientData
		if !bootstrapping {
			msg.Key = puzzleKey
			msg.ModelId = modelName
			msg.Deltas, err = oneGradientStep(globalW)

			if err != nil {
				fmt.Println("Got a GoPython failure, retrying...")
				conn.Close()
				continue
			}

		} else {
			msg.Key = puzzleKey
			msg.ModelId = modelName
			msg.Deltas = make([]float64, numFeatures)
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
		if (len(incomingMsg) > 0) {
			completed = true
		} else {
			time.Sleep(1 * time.Second)
		}

	}

	return 1
}

func getServerConnection(torDialer proxy.Dialer, isGradient bool) (net.Conn, error) {

	var conn net.Conn
	var err error

	if isGradient && torDialer != nil {
		conn, err = torDialer.Dial("tcp", torAddress)
	} else if isGradient {
		conn, err = net.Dial("tcp", torAddress)	
	} else if torDialer != nil {
		conn, err = torDialer.Dial("tcp", constructAddress(ONION_HOST, CONTROL_PORT))
	} else {
		conn, err = net.Dial("tcp", constructAddress(LOCAL_HOST, CONTROL_PORT))	
	}

	return conn, err

}

func sendJoinMessage(logger *govec.GoLog, torDialer proxy.Dialer) int {

	conn, err := getServerConnection(torDialer, false)
	checkError(err)

	fmt.Println("TOR Dial Success!")

	var msg MessageData
    msg.Type = "join"
    msg.SourceNode = name
    msg.ModelId = modelName
    msg.Key = ""
    msg.NumFeatures = numFeatures
    msg.MinClients = minClients

    outBuf := logger.PrepareSend("Sending packet to torserver", msg)
    	
	_, errWrite := conn.Write(outBuf)
	checkError(errWrite)
	
	inBuf := make([]byte, 512)
	n, errRead := conn.Read(inBuf)
	checkError(errRead)

	var puzzle string 
	var solution string
	var solved bool
	logger.UnpackReceive("Received puzzle from server", inBuf[0:n], &puzzle)
	conn.Close()

	for !solved {

		h := sha256.New()
		h.Write([]byte(puzzle))

		// Attempt a candidate
		timeHash := sha256.New()
		timeHash.Write([]byte(time.Now().String()))

		solution = hex.EncodeToString(timeHash.Sum(nil))
	    h.Write([]byte(solution))

	    hashed := hex.EncodeToString(h.Sum(nil))
	    fmt.Println(hashed)

		if strings.HasSuffix(hashed, "0000") {
		    fmt.Println("BINGO!")
		    solved = true
		}

	}

	conn, err = getServerConnection(torDialer, false)
	checkError(err)

	msg.Type = "solve"
	msg.Key = solution  

	fmt.Printf("Sending solution: %s\n", solution)

	outBuf = logger.PrepareSend("Sending puzzle solution", msg)
	_, errWrite = conn.Write(outBuf)
	checkError(errWrite)
	
	inBuf = make([]byte, 512)
	n, errRead = conn.Read(inBuf)
	checkError(errRead)

	var reply int
	logger.UnpackReceive("Received Message from server", inBuf[0:n], &reply)

	// The server replies with the port
	if reply != 0 {
		fmt.Println("Got ACK for puzzle")
		puzzleKey = solution

		if isLocal {
			torAddress = constructAddress(LOCAL_HOST, reply)
		} else {
			torAddress = constructAddress(ONION_HOST, reply)
		}

		fmt.Printf("Set up connection address %s\n", torAddress)

	} else {
		fmt.Println("My puzzle solution failed.")
	}

	conn.Close()

	return reply

}

func oneGradientStep(globalW []float64) ([]float64, error) {
	
	argArray := python.PyList_New(len(globalW))

	modelDiff := make([]float64, numFeatures)

	for i := 0; i < len(globalW); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(globalW[i]))
	
		// Perform the inversion
		modelDiff[i] = (globalW[i] - myLastModel[i])

		if myLastModel[i] != 0 {
			victimModel[i] += (globalW[i] - myLastModel[i] - myLastUpdate[i])
		}
		
		myLastModel[i] = globalW[i]
	}

	// Either use full GD or SGD here
	result := pyLogPrivFunc.CallFunction(python.PyInt_FromLong(1), argArray,
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

	for i := 0; i < len(goFloatArray); i++ {
		myLastUpdate[i] = goFloatArray[i]
	}
	
	return goFloatArray, nil
}

func constructAddress(host string, port int) string {

	var buffer bytes.Buffer
	buffer.WriteString(host)
	buffer.WriteString(strconv.Itoa(port))
	return buffer.String()
}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}
