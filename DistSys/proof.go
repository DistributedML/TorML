package main

import (
	"fmt"
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"time"
)

func main() {

	s := "genesispuzzle"

	solved := false
	var sha1_hash string
	var solution string

	for !solved {

		h := sha256.New()
		h.Write([]byte(s))

		// Attempt a candidate
		timeHash := sha256.New()
		timeHash.Write([]byte(time.Now().String()))

		solution = hex.EncodeToString(timeHash.Sum(nil))
	    h.Write([]byte(solution))

	    sha1_hash = hex.EncodeToString(h.Sum(nil))

	    fmt.Println(s, sha1_hash)

		if strings.HasSuffix(sha1_hash, "0000") {
		    fmt.Println("BINGO!")
		    solved = true
		}
	
	}

	// Verify the solution
	h2 := sha256.New()
	h2.Write([]byte(s))
	h2.Write([]byte(solution))
	hash := hex.EncodeToString(h2.Sum(nil))

    fmt.Println(s, solution, hash)

}