package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
	"github.com/DistributedClocks/GoVector/govec"
	"github.com/sbinet/go-python"
)

type MessageData struct {
	Type 		  string
	SourceNode    string
	ModelId       string
	Key 		  string
	NumFeatures   int
	MinClients    int
}

// Schema for data used in gradient updates
type GradientData struct {
	ModelId 	  string
	Key			  string
	Deltas 		  []float64
}

type ClientState struct {
	NumIterations		int
	NumValidations      int
	IsComputingLocal 	bool
	Weights 			[]float64
	OutlierScore		float64
}

// An active model: list of participants and weights
type Model struct {
	// Model based parameters: dimensions, w*
	NumFeatures 		int
	GlobalWeights 		[]float64

	// Bookkeeping
	Puzzles				map[string]string
	Clients 	 		map[string]ClientState
	MinClients 			int
}

type Validator struct {
	IsValidating		bool
	NumResponses		int
	ClientResponses 	map[string][]float64
	ValidationModel		[]float64
}

var (
	
	// For eval
	samplingRate		int  // How often will we sample? in ms
	convergThreshold	float64
	lossProgress		[]float64

	mutex 				*sync.Mutex

	myPorts				map[int]bool
	myModels 			map[string]Model
	myValidators	 	map[string]Validator

	MULTICAST_RATE		float64 = 1.1
	
	// Kick a client out after 2% of RONI
	THRESHOLD			float64 = -0.1

	// Test Module for python
	pyTestModule  *python.PyObject
	pyTestFunc    *python.PyObject
	pyTrainFunc   *python.PyObject
	pyRoniFunc    *python.PyObject
	pyPlotFunc    *python.PyObject

)

/*
	Executes when the .onion domain is accessed via the TorBrowser
*/
func handler(w http.ResponseWriter, r *http.Request) {

	req := r.URL.Path[1:]
    
    fmt.Fprintf(w, "Welcome %s!\n\n", r.URL.Path[1:])

    _, exists := myModels[req]
    
    if exists {

    	mutex.Lock()
    	model := myModels[req]
    	trainError, testError := testModel(model, "global")
    	fmt.Fprintf(w, "Train Loss: %f\n", trainError)	
    	fmt.Fprintf(w, "Test Loss: %f\n", testError)
    	fmt.Fprintf(w, "Num Clients: %d\n", len(model.Clients))

    	for node, clientState := range model.Clients {
    	
    		fmt.Fprintf(w, "\n");	
    		trainError, testError = testModel(model, node)
    		fmt.Fprintf(w, "%.5s iterations: %d\n", node, clientState.NumIterations)
    		fmt.Fprintf(w, "%.5s validations: %d\n", node, clientState.NumValidations)
    		fmt.Fprintf(w, "%.5s L2 outlier score: %f\n", node, clientState.OutlierScore)	
    		fmt.Fprintf(w, "%.5s train Loss: %f\n", node, trainError)	
    		fmt.Fprintf(w, "%.5s test Loss: %f\n", node, testError)	
    	
    	}
    	mutex.Unlock()
    }

    if req == "flush" {
    
    	file, err := os.Create("modelflush.csv")
    	defer file.Close()
    	checkError(err)

    	writer := csv.NewWriter(file)
    	defer writer.Flush()

    	mutex.Lock()
    	for _, model := range myModels {

    		st := strings.Fields(strings.Trim(fmt.Sprint(model.GlobalWeights), "[]"))
    		st = append(st, "global")
    		writer.Write(st)

    		for node, modelState := range model.Clients {
    			st := strings.Fields(strings.Trim(fmt.Sprint(modelState.Weights), "[]"))
    			st = append(st, node)
    			writer.Write(st)
    		}       
    	}

    	mutex.Unlock()
    	fmt.Fprintf(w, "Model flushed.")

	} else if req == "lossflush" {

		lossFlush()
		fmt.Fprintf(w, "Progress flushed.")

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

	pyTestModule = python.PyImport_ImportModule("logistic_model_test")
	pyTrainFunc = pyTestModule.GetAttrString("train_error")
	pyTestFunc = pyTestModule.GetAttrString("test_error")
	pyRoniFunc = pyTestModule.GetAttrString("roni")
	pyPlotFunc = pyTestModule.GetAttrString("plot")

}

func main() {

	fmt.Println("Launching server...")

	pyInit()
	
	mutex = &sync.Mutex{}

	myPorts = make(map[int]bool)
	myModels = make(map[string]Model)
	myValidators = make(map[string]Validator)

	// Measured in ms.
	samplingRate = 5000

	// What loss until you claim convergence?
	convergThreshold = 0.05

	// Make ports 6000 to 6010 available
	for i := 6000; i <= 6010; i++ {
		myPorts[i] = false
	}

	// setup the handler for web requests through the browser
	go httpHandler()

	// setup the handler for the hidden service endpoint
	go runRouter("127.0.0.1:5005")

	go runSampler()

    // Keeps server running
	select {}
}

func runSampler() {

	converged := false

	for !converged {

		time.Sleep(time.Duration(samplingRate) * time.Millisecond)

		_, exists := myModels["study"]

		if exists {

			// hardcoded
			mutex.Lock()
			model := myModels["study"]
			trainError, _ := testModel(model, "global")

			// Add the new error value at the back
			lossProgress = append(lossProgress, trainError)
			mutex.Unlock()

			if trainError < convergThreshold {
				converged = true
			}

		}

	}

	fmt.Println("I HAVE.... CONVERGENCE!")

	// Write out the final value
	lossFlush()

}

func lossFlush() {

	file, err := os.Create("lossflush.csv")
	defer file.Close()
	checkError(err)

	writer := csv.NewWriter(file)
	defer writer.Flush()

	mutex.Lock()
	st := strings.Fields(strings.Trim(fmt.Sprint(lossProgress), "[]"))
	writer.Write(st)

	/*

	UNUSED GO PYTHON CODE. IT DOESNT WORK 
	plotArray := python.PyList_New(len(lossProgress))

	for i := 0; i < len(lossProgress); i++ {
		python.PyList_SetItem(plotArray, i, python.PyFloat_FromDouble(lossProgress[i]))
	}

	isPlotted := pyPlotFunc.CallFunction(plotArray)
	fmt.Println(python.PyFloat_AsDouble(isPlotted))*/

	mutex.Unlock()


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

	fmt.Printf("Listening for TCP....\n")

	for {

		conn, err := ln.Accept()
		checkError(err)

		// Get the message from client
		conn.Read(buf)

		var inData MessageData
		Logger.UnpackReceive("Received Message From Client", buf[0:], &inData)

		outBuf = processControlMsg(inData, Logger)
	  	conn.Write(outBuf)

	}
}

func processControlMsg(inData MessageData, Logger *govec.GoLog) []byte {

	outBuf := make([]byte, 2048)
	var ok bool

	switch inData.Type {

		// Add new nodes
		case "join":
			puzzle := makePuzzle(inData.ModelId)
			outBuf = Logger.PrepareSend("Sending Puzzle", puzzle)

		// curate a new model
		case "curator":
			ok = startModel(inData.ModelId, inData.NumFeatures, inData.MinClients)
			if ok {
		  		outBuf = Logger.PrepareSend("Replying", 1)
			} else {
				outBuf = Logger.PrepareSend("Replying", 0)
			}				

		case "solve":
			ok = processJoin(inData.ModelId, inData.Key, inData.NumFeatures)
			if ok {
				address, port := getFreeAddress()
				go gradientWorker(inData.SourceNode, address, Logger)
		  		outBuf = Logger.PrepareSend("Replying", port)
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

	return outBuf

}

func gradientWorker(nodeId string, 
					address string, 
					Logger *govec.GoLog) {

	// Convert address string to net.Address
	myaddr, err := net.ResolveTCPAddr("tcp", address)
	checkError(err)

	ln, err := net.ListenTCP("tcp", myaddr)
	checkError(err)

	buf := make([]byte, 2048)
	outBuf := make([]byte, 2048)
	fmt.Printf("Listening for TCP....\n")

	for {

		conn, err := ln.Accept()
		checkError(err)

		// Get the message from client
		conn.Read(buf)

		var inData GradientData
		Logger.UnpackReceive("Received Message From Client", buf[0:], &inData)

		puzzleKey := inData.Key
		modelId := inData.ModelId

		_, modelExists := myModels[modelId]
		bufferReply := make([]float64, 0)

		if modelExists {

			mutex.Lock()

			_, clientExists := myModels[modelId].Clients[puzzleKey]
			enoughClients := len(myModels[modelId].Clients) >= myModels[modelId].MinClients

			if clientExists && enoughClients {
				gradientUpdate(puzzleKey, modelId, inData.Deltas)
			}

			// Hacky fix, but double check if thid client was kicked out at the last validation
			_, clientExists = myModels[modelId].Clients[puzzleKey]
			if clientExists && enoughClients {

				clientState := myModels[modelId].Clients[puzzleKey]

				if rand.Float64() > MULTICAST_RATE {
					startValidation(modelId)
				}

				if isClientValidating(modelId, puzzleKey) {
					bufferReply = clientState.Weights
					clientState.IsComputingLocal = true
				} else {
					bufferReply =  myModels[modelId].GlobalWeights
					clientState.IsComputingLocal = false
				}

				myModels[modelId].Clients[puzzleKey] = clientState
			} 

			mutex.Unlock()
		} 

		outBuf = Logger.PrepareSend("Replying...", bufferReply)
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

	pyTestResult := pyTestFunc.CallFunction(argArray)
	testErr := python.PyFloat_AsDouble(pyTestResult)

	pyTrainResult := pyTrainFunc.CallFunction(argArray)
	trainErr := python.PyFloat_AsDouble(pyTrainResult)
	
	return trainErr, testErr
}

func startModel(modelId string, numFeatures int, minClients int) bool {

	_, exists := myModels[modelId]
	if exists {
		fmt.Printf("Model %s already exists: \n", modelId)
		return false
	}

	// Add model to map and set random weights
	var newModel Model

	newModel.Clients = make(map[string]ClientState)
	newModel.Puzzles = make(map[string]string)
	newModel.NumFeatures = numFeatures
	newModel.GlobalWeights = newRandomModel(numFeatures)
	newModel.MinClients = minClients
	
	mutex.Lock()
	myModels[modelId] = newModel
	mutex.Unlock()
	fmt.Printf("Added a new model: %s\n", modelId)

	return true
}

/*
	Generates a new cryptographic puzzle for a client to solve
*/
func makePuzzle(modelId string) string {

	_, exists := myModels[modelId]
	if exists {

		// Generate a problem based on the current time
		timeHash := sha256.New()
		timeHash.Write([]byte(time.Now().String()))

		puzzle := hex.EncodeToString(timeHash.Sum(nil))

		mutex.Lock()
		theModel := myModels[modelId]
		theModel.Puzzles[puzzle] = "unsolved"
		myModels[modelId] = theModel
		mutex.Unlock()

		fmt.Printf("Made a new puzzle for model %s\n", modelId)
		return puzzle

	} 
	
	// Return an empty string for bad join
	fmt.Printf("Bad Puzzle request for model %s\n", modelId)
	return ""

}

// numFeature is just bootleg schema validation 
func processJoin(modelId string, givenKey string, numFeatures int) bool {

	fmt.Println("Got attempted puzzle solution")

	// check if model exists
	_, exists := myModels[modelId]
	if !exists {
		fmt.Printf("Rejected a fake join for model: %s\n", modelId)
		return false
	}

	mutex.Lock()
	theModel := myModels[modelId]
	if numFeatures != theModel.NumFeatures {
		fmt.Printf("Rejected an incorrect numFeatures for model: %s\n", modelId)
		mutex.Unlock()
		return false
	}

	_, exists = theModel.Clients[givenKey]
	if exists {
		fmt.Printf("Node %.5s is already joined in model: %s \n", givenKey, modelId)
		mutex.Unlock()
		return false
	}

	// Verify the solution
	for lock, key := range theModel.Puzzles {

		hash := sha256.New()

		if key == "unsolved" {

			hash.Write([]byte(lock))
			hash.Write([]byte(givenKey))
			hashString := hex.EncodeToString(hash.Sum(nil))
			
			if strings.HasSuffix(hashString, "0000") {

				// Add node
				theModel.Clients[givenKey] = 
					ClientState{0, 0, false, newRandomModel(numFeatures), 0}
				fmt.Printf("Joined %.5s in model %s \n", givenKey, modelId)
				
				// Write the solution
				theModel.Puzzles[lock] = givenKey
				myModels[modelId] = theModel
				mutex.Unlock()

				return true

			} 
		} 
	}

	mutex.Unlock()
	return false
}


func gradientUpdate(puzzleKey string, modelId string, deltas []float64) bool {

	_, exists := myModels[modelId]
	if exists {

		_, exists = myModels[modelId].Clients[puzzleKey]
		
		// Add in the deltas
		if exists { 

			theModel := myModels[modelId]
			clientState := theModel.Clients[puzzleKey]

			if clientState.IsComputingLocal {

				// Just update the local copy of the model
				validator := myValidators[modelId]
				validatorWeights := validator.ClientResponses[puzzleKey]

				for j := 0; j < len(deltas); j++ {
					validatorWeights[j] = deltas[j]
					clientState.Weights[j] += deltas[j]
				}

				validator.ClientResponses[puzzleKey] = validatorWeights
				validator.NumResponses++
				clientState.NumValidations++
				myValidators[modelId] = validator
				fmt.Printf("Validation update from %.5s on %s \n", puzzleKey, modelId)

				scoreMap := make(map[string]float64)
				isMulticast := false
				if validator.NumResponses >= len(validator.ClientResponses) {
					scoreMap = runValidation(modelId)
					isMulticast = true
					delete(myValidators, modelId) 
				}

				// Revert the client state
				clientState.IsComputingLocal = false
				theModel.Clients[puzzleKey] = clientState

				// Add all the multicast scores
				if isMulticast {
					
					for node, roni := range scoreMap {
						roniClientState := theModel.Clients[node]
						roniClientState.OutlierScore += roni
						theModel.Clients[node] = roniClientState

						if roniClientState.OutlierScore < THRESHOLD {
							fmt.Printf("node %.5s has RONI %5f past THRESHOLD. KICKED! \n", node, roniClientState.OutlierScore)
							delete(theModel.Clients, node)
							delete(theModel.Puzzles, node)
						}
					}   

					//fmt.Printf("Finished updating validation. There are now %d left.\n",
					//	len(theModel.Clients))

				}

				myModels[modelId] = theModel

			} else {

				// Update the global model
				for j := 0; j < len(deltas); j++ {
					theModel.GlobalWeights[j] += deltas[j]
				}
				
				clientState.NumIterations++

				theModel.Clients[puzzleKey] = clientState
				myModels[modelId] = theModel
				fmt.Printf("Grad update from %.5s on %s \n", puzzleKey, modelId)
							
			}

		} else {
			fmt.Printf("Client %.5s is not in this model.\n", puzzleKey)
		}

	}

	return exists
}

func runValidation(modelId string) map[string]float64 {

	fmt.Printf("Running validation....\n")

	truthModel := myValidators[modelId].ValidationModel
	truthArray := python.PyList_New(len(truthModel))

	for i := 0; i < len(truthModel); i++ {
		python.PyList_SetItem(truthArray, i, python.PyFloat_FromDouble(truthModel[i]))
	}

	scores := make(map[string]float64)

	for node, response := range myValidators[modelId].ClientResponses {
	
		argArray := python.PyList_New(len(truthModel))

		for i := 0; i < len(truthModel); i++ {
			python.PyList_SetItem(argArray, i, python.PyFloat_FromDouble(response[i]))
		}

		pyRoni := pyRoniFunc.CallFunction(truthArray, argArray)
		roni_score := python.PyFloat_AsDouble(pyRoni)

		fmt.Printf("RONI for node %.5s is %f \n", node, roni_score)
		scores[node] = roni_score

	}

	return scores

}

func kickoutNode(modelId string, nodeId string) {

	// Long and annoying atomic delete
	theModel := myModels[modelId]
	
	clientStates := theModel.Clients
	delete(clientStates, nodeId)
	theModel.Clients = clientStates

	delete(theModel.Puzzles, nodeId)
	myModels[modelId] = theModel

	fmt.Printf("Removed node %s from model\n", nodeId)

}

// Starts the validation multicast.
func startValidation(modelId string) {

	_, exists := myValidators[modelId]
	if !exists {

		// Create new validator object
		var valid Validator
		valid.IsValidating = true
		valid.NumResponses = 0
		valid.ClientResponses = make(map[string][]float64)
		valid.ValidationModel = myModels[modelId].GlobalWeights

		// Add to global state
		myValidators[modelId] = valid

		for node, state := range myModels[modelId].Clients {
			state.IsComputingLocal = true
			myModels[modelId].Clients[node] = state
			valid.ClientResponses[node] = make([]float64, myModels[modelId].NumFeatures)
		}

		fmt.Printf("Setup Validation multicast for %d clients.\n", len(myModels[modelId].Clients))	
	} 

}

func getFreeAddress() (string, int) {

	var buffer bytes.Buffer

	mutex.Lock()
	for port := range myPorts {
		
		if !myPorts[port] {

			// Mark port as taken
			myPorts[port] = true
			mutex.Unlock()

			// Construct the address string
			buffer.WriteString("127.0.0.1:")
			buffer.WriteString(strconv.Itoa(port))
			return buffer.String(), port
		}
	}
	
	mutex.Unlock()
	return "", 0
}

func isClientValidating(modelId string, node string) bool {
	
	_, exists := myValidators[modelId]
	
	// Short circuit?
	return exists && 
		myValidators[modelId].IsValidating &&
		myModels[modelId].Clients[node].IsComputingLocal

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