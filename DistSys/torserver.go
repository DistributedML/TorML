package main

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"os"
	"strings"
	"github.com/DistributedClocks/GoVector/govec"
	"github.com/sbinet/go-python"
)

type MessageData struct {
	Type 		  string
	SourceNode    string
	Model         ModelInfo
	Deltas 		  []float64
}

type ModelInfo struct {
	ModelId	 	string
	NumFeatures int
}

type ClientState struct {
	NumIterations		int
	IsComputingLocal 	bool
	Weights 			[]float64
}

// An active model: list of participants and weights
type Model struct {
	// use a map to store each local server's local updates
	NumFeatures 	int
	Clients 	 	map[string]ClientState
	GlobalWeights 	[]float64
}

var (
	maxnode     		int = 0
	numFeatures			int = 0
	registeredNodes		map[string]string
	myModels 			map[string]Model

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

    model, exists := myModels[req]
    
    if exists {

    	train_error, test_error := testModel(model, "global")
    	fmt.Fprintf(w, "Train Loss: %f\n", train_error)	
    	fmt.Fprintf(w, "Test Loss: %f\n", test_error)	

    	for node, clientState := range model.Clients {

    		fmt.Fprintf(w, "\n");	
    		train_error, test_error = testModel(model, node)
    		fmt.Fprintf(w, "%s iterations: %d\n", node, clientState.NumIterations)	
    		fmt.Fprintf(w, "%s train Loss: %f\n", node, train_error)	
    		fmt.Fprintf(w, "%s test Loss: %f\n", node, test_error)	

    	}
    }

    if req == "flush" {
    
    	file, err := os.Create("models.csv")
    	checkError(err)
    	defer file.Close()

    	writer := csv.NewWriter(file)
    	defer writer.Flush()

    	for _, model := range myModels {

    		st := strings.Fields(strings.Trim(fmt.Sprint(model.GlobalWeights), "[]"))
    		writer.Write(st)

    		for _, modelState := range model.Clients {
    			st := strings.Fields(strings.Trim(fmt.Sprint(modelState.Weights), "[]"))
    			writer.Write(st)
    		}
    	}

    	fmt.Fprintf(w, "Model flushed.")

    }
}

func httpHandler() {
	http.HandleFunc("/", handler)
	http.ListenAndServe(":6767", nil)

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
	myModels = make(map[string]Model)

	fmt.Printf("Listening for TCP....\n")

	for {

		conn, err := ln.Accept()
		checkError(err)

		// Get the message from client
		conn.Read(buf)

		var inData MessageData
		Logger.UnpackReceive("Received Message From Client", buf[0:], &inData)
	
		var ok bool

		switch inData.Type {

			// Client is sending a gradient update. Apply it and return myWeights
			case "grad":

				ok = gradientUpdate(inData.SourceNode,
					inData.Model.ModelId,
					inData.Deltas)

				clientState := myModels[inData.Model.ModelId].Clients[inData.SourceNode]

				// generates a float from 0 to 1
				if rand.Float64() > 0.8 {
					outBuf = Logger.PrepareSend("Replying", clientState.Weights)
					clientState.IsComputingLocal = true
				} else {
					outBuf = Logger.PrepareSend("Replying", myModels[inData.Model.ModelId].GlobalWeights)
					clientState.IsComputingLocal = false
				}

				myModels[inData.Model.ModelId].Clients[inData.SourceNode] = clientState
			
			// Add new nodes
			case "join":
				ok = processJoin(inData.SourceNode, inData.Model)
				if ok {
			  		outBuf = Logger.PrepareSend("Replying", 1)
				} else {
					outBuf = Logger.PrepareSend("Replying", 0)
				}

			// curate a new model
			case "curator":
				ok = startModel(inData.Model.ModelId, 
					inData.Model.NumFeatures)
				if ok {
			  		outBuf = Logger.PrepareSend("Replying", 1)
				} else {
					outBuf = Logger.PrepareSend("Replying", 0)
				}				

			case "beat":
				fmt.Printf("Heartbeat from %s\n", inData.SourceNode)
				outBuf = Logger.PrepareSend("Replying to heartbeat", 1)

			default:
				fmt.Println("Got a message type I dont recognize.")
				ok = false
				outBuf = nil
				
		}
	  	conn.Write(outBuf)
	}
}

func testModel(model Model, node string) (float64, float64) {

	var weights []float64
	if node == "global" {
		weights = model.GlobalWeights 
	} else {
		weights = model.Clients[node].Weights
	}

	argArray := python.PyList_New(len(weights))

	for i := 0; i < len(weights); i++ {
		python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(weights[i]))
	}

	test_result := testFunc.CallFunction(argArray)
	test_err := python.PyFloat_AsDouble(test_result)

	train_result := trainFunc.CallFunction(argArray)
	train_err := python.PyFloat_AsDouble(train_result)
	
	return train_err, test_err
}

func startModel(ModelId string, numFeatures int) bool {

	_, exists := myModels[ModelId]
	if exists {
		fmt.Printf("Model %s already exists: \n", ModelId)
		return false
	}

	// Add model to map and set random weights
	var newModel Model

	newModel.Clients = make(map[string]ClientState)
	newModel.NumFeatures = numFeatures
	newModel.GlobalWeights = newRandomModel(numFeatures)

	myModels[ModelId] = newModel
	fmt.Printf("Added a new model: %s\n", ModelId)

	return true
}

func processJoin(node string, model ModelInfo) bool {

	// check if model exists
	theModel, exists := myModels[model.ModelId]
	if !exists {
		fmt.Printf("Rejected a fake join for model: %s\n", model.ModelId)
		return false
	}

	if model.NumFeatures != theModel.NumFeatures {
		fmt.Printf("Rejected an incorrect numFeatures for model: %s\n", model.ModelId)
		return false
	}

	_, exists = theModel.Clients[node]
	if exists {
		fmt.Printf("Node %s is already joined in model: %s \n", node, model.ModelId)
		return false
	}

	// Add node
	theModel.Clients[node] = ClientState{0, false, newRandomModel(model.NumFeatures)}
	fmt.Printf("Joined %s in model %s \n", node, model.ModelId)

	return true
}

func gradientUpdate(nodeId string, ModelId string, deltas []float64) bool {

	_, exists := myModels[ModelId]
	if exists {

		_, exists = myModels[ModelId].Clients[nodeId]
		
		// Add in the deltas
		if exists { 

			theModel := myModels[ModelId]
			isLocal := theModel.Clients[nodeId].IsComputingLocal

			if isLocal {
				
				// Just update the local copy of the model
				clientState := myModels[ModelId].Clients[nodeId]
				for j := 0; j < len(deltas); j++ {
					clientState.Weights[j] += deltas[j]
				}

				clientState.NumIterations = clientState.NumIterations + 1
				theModel.Clients[nodeId] = clientState
				myModels[ModelId] = theModel
				fmt.Printf("Local grad update from %s on %s \n", nodeId, ModelId)

			} else {

				// Update the global model
				for j := 0; j < len(deltas); j++ {
					theModel.GlobalWeights[j] += deltas[j]
				}
				
				myModels[ModelId] = theModel
				fmt.Printf("Grad update from %s on %s \n", nodeId, ModelId)
							
			}
		}
	}

	return exists
}

// Helper function to generate a random array of length numFeatures
func newRandomModel(numFeatures int) []float64 {

	model := make([]float64, numFeatures)
	for i := 0; i < numFeatures; i++ {
		model[i] = rand.Float64()
	}

	return model
}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}