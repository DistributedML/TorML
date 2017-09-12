package main

import (
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"os"
	"github.com/DistributedClocks/GoVector/govec"
	"github.com/sbinet/go-python"
)

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

// An active study: list of participants and weights
type Study struct {
	Participants map[string]string // use a map just because its easier to search through
	Weights 	 []float64
}

var (
	maxnode     		int = 0
	numFeatures			int = 0
	registeredNodes		map[string]string
	myStudies 			map[string]Study

	// Test Module for python
	testModule  *python.PyObject
	testFunc    *python.PyObject
	trainFunc   *python.PyObject
)

/*
	Executes when the .onion domain is accessed via the TorBrowser
*/
func handler(w http.ResponseWriter, r *http.Request) {

	req := r.URL.Path[1:]
    
    fmt.Fprintf(w, "Welcome %s!\n\n", r.URL.Path[1:])

    study, exists := myStudies[req]
    if exists {
    	train_error, test_error := testModel(study)
    	fmt.Fprintf(w, "Train Loss: %f\n", train_error)	
    	fmt.Fprintf(w, "Test Loss: %f\n", test_error)	
    }

}

func httpHandler() {
	http.HandleFunc("/", handler)
	http.ListenAndServe(":8080", nil)

	fmt.Printf("HTTP initialized.\n")
}


/*
	Automatically used by GoPython to establish function bindings
*/
func init() {
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

/*
	Initializes python function objects  
*/
func pyInit() {

	sysPath := python.PySys_GetObject("path")
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("./"))
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("../ML/code"))

	testModule = python.PyImport_ImportModule("logistic_model_test")
	trainFunc = testModule.GetAttrString("train_error")
	testFunc = testModule.GetAttrString("test_error")
}

func main() {

	fmt.Println("Launching server...")

	pyInit()
	
	// setup the handler for web requests through the browser
	go httpHandler()

	// setup the handler for the hidden service endpoint
	go runRouter("127.0.0.1:5005")

    // Keeps server running
	select {}
}

// Sets up the TCP connection, and attaches GoVector.
func runRouter(address string) {

	Logger := govec.InitGoVector("torserver", "torserverfile")

	// Convert address string to net.Address
	myaddr, err := net.ResolveTCPAddr("tcp", address)
	checkError(err)

	ln, err := net.ListenTCP("tcp", myaddr)
	checkError(err)

	buf := make([]byte, 2048)
	outBuf := make([]byte, 2048)
	registeredNodes = make(map[string]string)
	myStudies = make(map[string]Study)

	// TODO REMOVE
	// fmt.Printf("Start sample study: %t\n", startStudy("study1", 101))

	for {

		fmt.Printf("Listening for TCP....\n")

		conn, err := ln.Accept()
		checkError(err)

		fmt.Println("Got message")

		// Get the message from client
		conn.Read(buf)

		var incomingData MessageData
		Logger.UnpackReceive("Received Message From Client", buf[0:], &incomingData)
	
		var ok bool

		switch incomingData.Type {

			// Client is sending a gradient update. Apply it and return myWeights
			case "grad":
				ok = gradientUpdate(incomingData.SourceNode, 
					incomingData.Study.StudyId, 
					incomingData.Deltas)

				outBuf = Logger.PrepareSend("Replying", 
					myStudies[incomingData.Study.StudyId].Weights)
			
			// Add new nodes
			case "join":
				ok = processJoin(incomingData.SourceNode, incomingData.Study)
				if ok {
			  		outBuf = Logger.PrepareSend("Replying", 1)
				} else {
					outBuf = Logger.PrepareSend("Replying", 0)
				}

			// curate a new study
			case "curator":
				ok = startStudy(incomingData.Study.StudyId, 
					incomingData.Study.NumFeatures)
				if ok {
			  		outBuf = Logger.PrepareSend("Replying", 1)
				} else {
					outBuf = Logger.PrepareSend("Replying", 0)
				}				

			case "beat":
				fmt.Printf("Heartbeat from %s\n", incomingData.SourceNode)
				outBuf = Logger.PrepareSend("Replying to heartbeat", 1)

			default:
				fmt.Println("Got a message type I dont recognize.")
				ok = false
				outBuf = nil
				
		}

	  	conn.Write(outBuf)
		fmt.Printf("Done processing data from %s\n", incomingData.SourceNode)

	}

}

func testModel(study Study) (float64, float64) {

	myWeights := study.Weights 
	argArray := python.PyList_New(len(myWeights))

	for i := 0; i < len(myWeights); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(myWeights[i]))
	}

	test_result := testFunc.CallFunction(argArray)
	test_err := python.PyFloat_AsDouble(test_result)

	train_result := trainFunc.CallFunction(argArray)
	train_err := python.PyFloat_AsDouble(train_result)
	
	return train_err, test_err
}

func startStudy(studyId string, numFeatures int) bool {

	_, exists := myStudies[studyId]
	if exists {
		fmt.Printf("Study %s already exists: \n", studyId)
		return false
	}

	// Add study to map and set random weights
	var newStudy Study

	newStudy.Participants = make(map[string]string)
	newStudy.Weights = make([]float64, numFeatures)
	for i := 0; i < numFeatures; i++ {
		newStudy.Weights[i] = rand.Float64()
	}

	myStudies[studyId] = newStudy
	fmt.Printf("Added a new study: %s\n", studyId)

	return true
}

func processJoin(node string, study StudyInfo) bool {

	// check if study exists
	theStudy, exists := myStudies[study.StudyId]
	if !exists {
		fmt.Printf("Rejected a fake join for study: %s\n", study.StudyId)
		return false
	}

	_, exists = theStudy.Participants[node]
	if exists {
		fmt.Printf("Node %s is already joined in study: %s \n", node, study.StudyId)
		return false
	}

	// Add node
	theStudy.Participants[node] = "in" 
	fmt.Printf("Joined %s in study %s \n", node, study.StudyId)

	return true
}

func gradientUpdate(nodeId string, studyId string, deltas []float64) bool {

	_, exists := myStudies[studyId]
	if exists {

		_, exists = myStudies[studyId].Participants[nodeId]
		
		// Add in the deltas
		if exists { 

			theStudy := myStudies[studyId]

			for j := 0; j < len(deltas); j++ {
				theStudy.Weights[j] += deltas[j]
			}

			myStudies[studyId] = theStudy
		}
	}

	fmt.Printf("Grad update from %s on %s %t \n", nodeId, studyId, exists)
	return exists

}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}