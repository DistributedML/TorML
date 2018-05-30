package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/binary"
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
)

type MessageData struct {
	Type 		  string
	SourceNode    string
    Key 		  string
	NumFeatures   int
	MinClients    int
}

// Schema for data used in gradient updates
type GradientData struct {
    Key           string
    Solution      string
    Deltas        []float64
}

type GradientReturnData struct {
    NextHash        string
    Difficulty      int
    GlobalModel     []float64
}

type ClientState struct {
	NumIterations		int
	NumValidations      int
	IsComputingLocal 	bool
	Weights 			[]float64
	OutlierScore		float64
    NextHash            string
    HashDifficulty      int
}

// An active model: list of participants and weights
type Model struct {
	// Model based parameters: dimensions, w*
	NumFeatures 		int
	GlobalWeights 		[]float64

	TempGradients		map[string][]float64

	// Bookkeeping
	Puzzles				map[string]string
	Clients 	 		map[string]ClientState
	MinClients 			int
    NumIterations       int
}

type Validator struct {
	Exists              bool
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
	theModel 			Model
	theValidator	 	Validator

    // Inverse rate of multicast. Set to > 1 if you want to disable
	MULTICAST_RATE		float64 = 0.9
	
	// Kick a client out after 2% of RONI
	THRESHOLD			float64 = -0.02

    DEFAULT_JOIN_POW    int = 4

    // synchrony model
    SYNCHRONOUS         bool = false

	// Test Module for python
	pyTestModule  *python.PyObject
	pyTestFunc    *python.PyObject
	pyTrainFunc   *python.PyObject
	pyRoniFunc    *python.PyObject
	pyPlotFunc    *python.PyObject
    pyAggInitFunc *python.PyObject
    pyTestInitFunc *python.PyObject

	pyAggModule   *python.PyObject
	pyKrumFunc    *python.PyObject
	pyLshFunc     *python.PyObject
    pyAvgFunc     *python.PyObject
	
)

/*
	Executes when the .onion domain is accessed via the TorBrowser
*/
func handler(w http.ResponseWriter, r *http.Request) {

	req := r.URL.Path[1:]
    
    fmt.Fprintf(w, "Welcome %s!\n\n", r.URL.Path[1:])

    if req == "model" {

    	mutex.Lock()
    	trainError, testError := testModel(theModel, "global")
    	fmt.Fprintf(w, "Train Loss: %f\n", trainError)	
    	fmt.Fprintf(w, "Test Loss: %f\n", testError)
        fmt.Fprintf(w, "Num Iterations: %d\n", theModel.NumIterations)
    	fmt.Fprintf(w, "Num Clients: %d\n", len(theModel.Clients))

    	for node, clientState := range theModel.Clients {
    	
    		fmt.Fprintf(w, "\n");	
    		trainError, testError = testModel(theModel, node)
    		fmt.Fprintf(w, "%.5s iterations: %d\n", node, clientState.NumIterations)
    		fmt.Fprintf(w, "%.5s validations: %d\n", node, clientState.NumValidations)
    		fmt.Fprintf(w, "%.5s L2 outlier score: %f\n", node, clientState.OutlierScore)	
            fmt.Fprintf(w, "%.5s Hash Difficulty: %d\n", node, clientState.HashDifficulty)   
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
    	
		st := strings.Fields(strings.Trim(fmt.Sprint(theModel.GlobalWeights), "[]"))
		st = append(st, "global")
		writer.Write(st)

		for node, modelState := range theModel.Clients {
			st := strings.Fields(strings.Trim(fmt.Sprint(modelState.Weights), "[]"))
			st = append(st, node)
			writer.Write(st)
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

	// pyAggModule = python.PyImport_ImportModule("logistic_aggregator")
	// pyTestModule = python.PyImport_ImportModule("logistic_model_test")
	
    pyAggModule = python.PyImport_ImportModule("logistic_aggregator")
    pyTestModule = python.PyImport_ImportModule("logistic_model_test")

	pyAggInitFunc = pyAggModule.GetAttrString("init")
    pyLshFunc = pyAggModule.GetAttrString("lsh_sieve")
	pyKrumFunc = pyAggModule.GetAttrString("krum")
    pyAvgFunc = pyAggModule.GetAttrString("average")
    
    pyTestInitFunc = pyTestModule.GetAttrString("init")
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

	// Measured in ms.
	samplingRate = 100

	// What loss until you claim convergence?
	convergThreshold = 0.01

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

func runSampler() {

	converged := false

	for !converged {

		time.Sleep(time.Duration(samplingRate) * time.Millisecond)

		// hardcoded
		mutex.Lock()
		trainError, _ := testModel(theModel, "global")

		// Add the new error value at the back
		lossProgress = append(lossProgress, trainError)
		mutex.Unlock()

		if trainError < convergThreshold {
			converged = true
		}	

	}

	fmt.Println("I HAVE.... CONVERGENCE!")

	// Write out the final value
	lossFlush()

    os.Exit(0)

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
			puzzle := makePuzzle()
			outBuf = Logger.PrepareSend("Sending Puzzle", puzzle)

		// curate a new model
		case "curator":
			ok = startModel(inData.NumFeatures, inData.MinClients)
			if ok {
		  		outBuf = Logger.PrepareSend("Replying", 1)
			} else {
				outBuf = Logger.PrepareSend("Replying", 0)
			}				

		case "solve":
			ok = processJoin(inData.Key, inData.NumFeatures)
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

    // 2048 for credit is ok
    // 131072 for MNIST
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

		// random sleeps from 0 - 0.1 ms
		time.Sleep(time.Duration(100 * rand.Float64()) * time.Millisecond)
		
		mutex.Lock()

        bufferReply := make([]float64, 0)

        // Check basic requirements
		_, clientExists := theModel.Clients[puzzleKey]
		enoughClients := len(theModel.Clients) >= theModel.MinClients

        // Check and update the puzzle
        clientState := theModel.Clients[puzzleKey]        
        solved := verifyPuzzle(clientState.NextHash, inData.Solution, clientState.HashDifficulty)

        if clientExists && enoughClients && solved {
            clientState.NextHash = clientState.NextHash + inData.Solution
            gradientUpdate(puzzleKey, inData.Deltas)
		}

        // gradient update will update difficulty if needed
        difficultyReply := clientState.HashDifficulty
        hashReply := clientState.NextHash

		// Hacky fix, but double check if this client was kicked out at the last validation
		_, clientExists = theModel.Clients[puzzleKey]
		if clientExists && enoughClients {

			clientState := theModel.Clients[puzzleKey]

			if rand.Float64() > MULTICAST_RATE {
				startValidation()
			}

			if isClientValidating(puzzleKey) {
				bufferReply = clientState.Weights
				clientState.IsComputingLocal = true
			} else {
				bufferReply =  theModel.GlobalWeights
				clientState.IsComputingLocal = false
			}

			theModel.Clients[puzzleKey] = clientState
		} 

		mutex.Unlock()

        var gradReply GradientReturnData
        gradReply.NextHash = hashReply
        gradReply.Difficulty = difficultyReply
        gradReply.GlobalModel = bufferReply

		outBuf = Logger.PrepareSend("Replying...", gradReply)
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

    pyTrainResult := pyTrainFunc.CallFunction(argArray)
    trainErr := python.PyFloat_AsDouble(pyTrainResult)

	pyTestResult := pyTestFunc.CallFunction(argArray)
	testErr := python.PyFloat_AsDouble(pyTestResult)
	
	return trainErr, testErr
}

func startModel(numFeatures int, minClients int) bool {

	// Add model to map and set random weights
	var newModel Model

	newModel.Clients = make(map[string]ClientState)
	newModel.Puzzles = make(map[string]string)
	newModel.NumFeatures = numFeatures
	newModel.GlobalWeights = newRandomModel(numFeatures)
	newModel.MinClients = minClients
	newModel.TempGradients = make(map[string][]float64)
    newModel.NumIterations = 0

	mutex.Lock()
	theModel = newModel
	mutex.Unlock()
	fmt.Printf("Started new model! \n")

    pyTestInitFunc.CallFunction(python.PyInt_FromLong(10), 
        python.PyInt_FromLong(784))

    pyAggInitFunc.CallFunction(python.PyInt_FromLong(minClients), 
        python.PyInt_FromLong(numFeatures))

    go runSampler()

	return true
}

/*
	Generates a new cryptographic puzzle for a client to solve
*/
func makePuzzle() string {

	// Generate a problem based on the current time
	timeHash := sha256.New()
	timeHash.Write([]byte(time.Now().String()))

	puzzle := hex.EncodeToString(timeHash.Sum(nil))

	mutex.Lock()
	theModel.Puzzles[puzzle] = "unsolved"
	mutex.Unlock()

	fmt.Printf("Made a new puzzle for model \n")
	return puzzle

}

// numFeature is just bootleg schema validation 
func processJoin(givenKey string, numFeatures int) bool {

	fmt.Println("Got attempted puzzle solution")

	mutex.Lock()
	if numFeatures != theModel.NumFeatures {
		fmt.Printf("Rejected an incorrect numFeatures for model.\n")
		mutex.Unlock()
		return false
	}

    var exists bool
	_, exists = theModel.Clients[givenKey]
	if exists {
		fmt.Printf("Node %.5s is already joined in model.\n", givenKey)
		mutex.Unlock()
		return false
	}

	// Verify the solution
	for lock, key := range theModel.Puzzles {

		if key == "unsolved" && verifyPuzzle(lock, givenKey, DEFAULT_JOIN_POW) {

			// Add node
			theModel.Clients[givenKey] = 
				ClientState{0, 0, false, newRandomModel(numFeatures), 0, lock + key, 1}
			fmt.Printf("Joined %.5s in model \n", givenKey)
			
			// Write the solution
			theModel.Puzzles[lock] = givenKey

            // multiple exits, need to unlock
			mutex.Unlock()
			return true

		} 
	}

	mutex.Unlock()
	return false
}

func verifyPuzzle(lock string, givenKey string, difficulty int) bool {

    var b bytes.Buffer

    for i := 0; i < difficulty; i++ {
        b.WriteString("0")
    }

    trailing := b.String()

    hash := sha256.New()
    hash.Write([]byte(lock))
    hash.Write([]byte(givenKey))
    hashString := hex.EncodeToString(hash.Sum(nil))
    
    return strings.HasSuffix(hashString, trailing)

}

func gradientUpdate(puzzleKey string, deltas []float64) {

	clientState := theModel.Clients[puzzleKey]

	if clientState.IsComputingLocal {

		// Just update the local copy of the model
		validatorWeights := theValidator.ClientResponses[puzzleKey]

		for j := 0; j < len(deltas); j++ {
			validatorWeights[j] = deltas[j]
			clientState.Weights[j] += deltas[j]
		}

		theValidator.ClientResponses[puzzleKey] = validatorWeights
		theValidator.NumResponses++
		clientState.NumValidations++
		fmt.Printf("Validation update from %.5s \n", puzzleKey)

		scoreMap := make(map[string]float64)
		isMulticast := false
		if theValidator.NumResponses >= len(theValidator.ClientResponses) {
			scoreMap = runValidation()
			isMulticast = true
			theValidator.Exists = false
		}

		// Revert the client state
		clientState.IsComputingLocal = false
		theModel.Clients[puzzleKey] = clientState

		// Add all the multicast scores
		if isMulticast {
			
			for node, roni := range scoreMap {
				roniClientState := theModel.Clients[node]
				roniClientState.OutlierScore += roni
				

				if roniClientState.OutlierScore < THRESHOLD {
				    
                    fmt.Printf("node %.5s has RONI %5f past THRESHOLD. UP! \n", node, roniClientState.OutlierScore)	
                    roniClientState.HashDifficulty++
                    roniClientState.OutlierScore = 0

                    /*fmt.Printf("node %.5s has RONI %5f past THRESHOLD. KICKED! \n", node, roniClientState.OutlierScore)
					delete(theModel.Clients, node)
					delete(theModel.Puzzles, node)*/
				}

                theModel.Clients[node] = roniClientState

			}   
		}

	} else {

        // Collect the update 
        if SYNCHRONOUS {

            theModel.TempGradients[puzzleKey] = deltas
            fmt.Printf("Grad update from %.5s \n", puzzleKey)
            clientState.NumIterations++
            theModel.Clients[puzzleKey] = clientState

            // store the updates as they come
            if len(theModel.TempGradients) == theModel.MinClients {
                
                dd := len(deltas)

                // Copy the map to a dgx1 vector
                pyAllDeltas := python.PyList_New(dd * theModel.MinClients)
                
                g := 0
                for cHash := range theModel.TempGradients {
                    for i := 0; i < dd; i++ {
                        python.PyList_SetItem(pyAllDeltas, (g * dd) + i, 
                            python.PyFloat_FromDouble(theModel.TempGradients[cHash][i]))
                    }
                    g++
                }

                pyTotalUpdate := pyAvgFunc.CallFunction(pyAllDeltas)

                total_update := pythonToGoFloatArray(pyTotalUpdate)

                // Update the global model
                for j := 0; j < dd; j++ {
                    theModel.GlobalWeights[j] += total_update[j]
                }

                /*
                USE THE KRUM FUNCTION
                krumIdx := python.PyInt_AsLong(pyKrumIdx)

                // Either use full GD or SGD here
                pyKrumIdx := pyKrumFunc.CallFunction(pyAllDeltas, 
                    python.PyInt_FromLong(len(deltas)), 
                    python.PyInt_FromLong(theModel.MinClients))

                krumIdx := python.PyInt_AsLong(pyKrumIdx)

                // Update the global model
                for j := 0; j < dd; j++ {
                    theModel.GlobalWeights[j] += theModel.TempGradients[krumIdx][j]
                }
                
                fmt.Printf("Krum update using idx %d from %.5s on %s \n", 
                    krumIdx, puzzleKey, modelId)

                */

                theModel.NumIterations++
                fmt.Printf("Grad update %d \n", theModel.NumIterations)
                theModel.TempGradients = make(map[string][]float64)

            } 

        } else {

            // Asycnhronous, just apply the update
            dd := len(deltas)

            // Update the global model
            for j := 0; j < dd; j++ {
                theModel.GlobalWeights[j] += deltas[j]
            }

            theModel.NumIterations++
            clientState.NumIterations++
            theModel.Clients[puzzleKey] = clientState

            fmt.Printf("Grad update %d from %.5s \n", theModel.NumIterations, puzzleKey)
            
        }   
					
	}

	return
}

func runValidation() map[string]float64 {

	fmt.Printf("Running validation....\n")

	truthModel := theValidator.ValidationModel
	truthArray := python.PyList_New(len(truthModel))

	for i := 0; i < len(truthModel); i++ {
		python.PyList_SetItem(truthArray, i, python.PyFloat_FromDouble(truthModel[i]))
	}

	scores := make(map[string]float64)

	for node, response := range theValidator.ClientResponses {
	
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

func kickoutNode(nodeId string) {

	// Long and annoying atomic delete
	clientStates := theModel.Clients
	delete(clientStates, nodeId)
	theModel.Clients = clientStates
	delete(theModel.Puzzles, nodeId)

	fmt.Printf("Removed node %s from model\n", nodeId)

}

// Starts the validation multicast.
func startValidation() {

	if !theValidator.Exists {

		// Create new validator object
		var valid Validator
		valid.Exists = true
        valid.IsValidating = true
		valid.NumResponses = 0
		valid.ClientResponses = make(map[string][]float64)
		valid.ValidationModel = theModel.GlobalWeights

		// Add to global state
		theValidator = valid

		for node, state := range theModel.Clients {
			state.IsComputingLocal = true
			theModel.Clients[node] = state
			valid.ClientResponses[node] = make([]float64, theModel.NumFeatures)
		}

		fmt.Printf("Setup Validation multicast for %d clients.\n", len(theModel.Clients))	
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

func isClientValidating(node string) bool {
	
	// Short circuit
	return theValidator.Exists && 
		theValidator.IsValidating &&
		theModel.Clients[node].IsComputingLocal

}

// Helper function to generate a random array of length numFeatures
func newRandomModel(numFeatures int) []float64 {

	model := make([]float64, numFeatures)
	for i := 0; i < numFeatures; i++ {
		model[i] = rand.Float64() / 1000.0
	}

	return model
}

func pythonToGoFloatArray(result *python.PyObject) []float64 {

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

	return goFloatArray

}

// Error checking function
func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}
