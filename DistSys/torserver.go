package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"fmt"
	"math"
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
	"github.com/gonum/matrix/mat64"
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
	Work 				[][]float64
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
	
	mutex 				*sync.Mutex

	myPorts				map[int]bool
	myModels 			map[string]Model
	myValidators	 	map[string]Validator

	MULTICAST_RATE		float64 = 0.8
	THRESHOLD			float64 = 0.005

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
    		fmt.Fprintf(w, "%s validations: %d\n", node, clientState.NumValidations)
    		fmt.Fprintf(w, "%s L2 outlier score: %f\n", node, clientState.OutlierScore)	
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

    		for node, modelState := range model.Clients {
    			st := strings.Fields(strings.Trim(fmt.Sprint(modelState.Weights), "[]"))
    			st = append(st, node)
    			writer.Write(st)
    		}       
    	}

    	fmt.Fprintf(w, "Model flushed.")

    } else if req == "workflush" {

    	for _, model := range myModels {

    		for node, modelState := range model.Clients {
    			
    			file, err := os.Create("work_" + node + ".csv")
    			checkError(err)
    			defer file.Close()

    			writer := csv.NewWriter(file)
    			defer writer.Flush()

    			for _, update := range modelState.Work {

	    			st := strings.Fields(strings.Trim(fmt.Sprint(update), "[]"))
	    			writer.Write(st)

    			}

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
	
	mutex = &sync.Mutex{}

	myPorts = make(map[int]bool)
	myModels = make(map[string]Model)
	myValidators = make(map[string]Validator)

	// Make ports 6000 to 6010 available
	for i := 6000; i <= 6010; i++ {
		myPorts[i] = false
	}

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
			ok = processJoin(inData.SourceNode, inData.ModelId, inData.Key, inData.NumFeatures)
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

		modelId := inData.ModelId

		// Not ready for gradients yet
		if len(myModels[modelId].Clients) < myModels[modelId].MinClients {
			outBuf = Logger.PrepareSend("Replying", make([]float64, 0))
		} else {

			setUpValid := false

			if rand.Float64() > MULTICAST_RATE {
				setUpValid = true
			}

			gradientUpdate(nodeId, modelId, inData.Deltas)

			mutex.Lock()
			clientState := myModels[modelId].Clients[nodeId]

			if setUpValid {
				outBuf = Logger.PrepareSend("Replying", clientState.Weights)
				clientState.IsComputingLocal = true
				fmt.Printf("Setup Validation for %s\n", nodeId)
			} else {
				outBuf = Logger.PrepareSend("Replying", myModels[modelId].GlobalWeights)
				clientState.IsComputingLocal = false
			}

			myModels[modelId].Clients[nodeId] = clientState
			mutex.Unlock()
		
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
func processJoin(node string, modelId string, givenKey string, numFeatures int) bool {

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

	_, exists = theModel.Clients[node]
	if exists {
		fmt.Printf("Node %s is already joined in model: %s \n", node, modelId)
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
				theModel.Clients[node] = 
					ClientState{0, 0, false, 
						newRandomModel(numFeatures), 0, make([][]float64, 0)}
				fmt.Printf("Joined %s in model %s \n", node, modelId)
				
				// Write the solution
				theModel.Puzzles[lock] = givenKey
				myModels[modelId] = theModel
				mutex.Unlock()

				return true

			} else {
				fmt.Printf("Solution is not right: %s\n", givenKey)
				fmt.Printf("Solution answer: %s\n", hashString)
			}
		} 
	}

	mutex.Unlock()
	return false
}

func gradientUpdate(nodeId string, modelId string, deltas []float64) bool {

	_, exists := myModels[modelId]
	if exists {

		_, exists = myModels[modelId].Clients[nodeId]
		
		// Add in the deltas
		if exists { 

			mutex.Lock()
			theModel := myModels[modelId]
			clientState := theModel.Clients[nodeId]

			if clientState.IsComputingLocal {

				// Just update the local copy of the model
				//validator := myValidators[modelId]
				//validatorWeights := validator.ClientResponses[nodeId]

				for j := 0; j < len(deltas); j++ {
					//validatorWeights[j] = deltas[j]
					clientState.Weights[j] += deltas[j]
				}

				//validator.ClientResponses[nodeId] = validatorWeights
				//validator.NumResponses++
				clientState.NumValidations++
				//myValidators[modelId] = validator
				fmt.Printf("Validation update from %s on %s \n", nodeId, modelId)

				/*if validator.NumResponses >= len(validator.ClientResponses) {
					fmt.Println("READY FOR VALIDATION MAGIC")
					runValidation(modelId)
					delete(myValidators, modelId) 
				}*/

				// Revert the client state
				clientState.IsComputingLocal = false
				theModel.Clients[nodeId] = clientState
				myModels[modelId] = theModel

			} else {

				// Update the global model
				for j := 0; j < len(deltas); j++ {
					theModel.GlobalWeights[j] += deltas[j]
				}
				
				clientState.NumIterations++

				theModel.Clients[nodeId] = clientState
				myModels[modelId] = theModel
				fmt.Printf("Grad update from %s on %s \n", nodeId, modelId)
							
			}

			// End of gradient update
			mutex.Unlock()

		} else {
			fmt.Printf("Client %s is not in this model.\n")
		}
	}

	return exists
}

func runValidation(modelId string) {

	// Convert client responses into mat64 vectors
	scores := make(map[string]*mat64.Dense)
	for node, model := range myValidators[modelId].ClientResponses {
		scores[node] = mat64.NewDense(myModels[modelId].NumFeatures, 1, model)
	}

	mutex.Lock()
	theModel := myModels[modelId]
	for node1, vector1 := range scores {
		
		// Sum distance between vector1 and all other vectors
		var rollingSum float64
		for _, vector2 := range scores {
			rollingSum += FindEucDistance(vector1, vector2)
		}

		// Ignore distance to itself, which is always 0.
		rollingSum /= float64(len(scores) - 1) 

		// Normalize for number of features
		rollingSum /= float64(myModels[modelId].NumFeatures)

		// Atomic update
		clientState := theModel.Clients[node1]
		clientState.OutlierScore += rollingSum
		theModel.Clients[node1] = clientState

		// save the current rolling
		rollingAvg := rollingSum / float64(clientState.NumValidations + 1)
		fmt.Printf("Average Distance for %s: %.9f \n", node1, rollingAvg)

		if rollingAvg > THRESHOLD {
			fmt.Printf("node %s has average %7f over THRESHOLD %7f. KICKED! \n", 
				node1, rollingAvg, THRESHOLD)
			kickoutNode(modelId, node1)
		}

	}

	myModels[modelId] = theModel
	mutex.Unlock()

}

func kickoutNode(modelId string, nodeId string) {

	// Long and annoying atomic delete
	mutex.Lock()
	theModel := myModels[modelId]
	clientStates := theModel.Clients
	delete(clientStates, nodeId)
	theModel.Clients = clientStates
	myModels[modelId] = theModel
	mutex.Unlock()

	fmt.Printf("Removed node %s from model\n", nodeId)

}

// Starts the validation multicast.
func startValidation(modelId string) {

	_, exists := myValidators[modelId]
	if !exists {

		fmt.Println("Doing a validation multicast")

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

		fmt.Println("Setup Validation multicast")	

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


func FindEucDistance(vectorX *mat64.Dense, vectorY *mat64.Dense) float64 {
	
	// Finds X-Y
	distanceVec := mat64.NewDense(0, 0, nil)
	distanceVec.Sub(vectorX, vectorY)

	result := mat64.NewDense(0, 0, nil)
	result.MulElem(distanceVec, distanceVec)

	return math.Sqrt(mat64.Sum(result))
}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}