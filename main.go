package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/neural"
)

func main() {
	xOne, err := strconv.ParseInt(os.Args[1], 10, 64)
	xTwo, err := strconv.ParseInt(os.Args[2], 10, 64)
	fmt.Println("")

	if err != nil {
		panic(err)
	}

	XNORData, _ := base.ParseCSVToInstances("xnor.csv", false)

	net := neural.NewMultiLayerNet([]int{3})
	net.MaxIterations = 20000
	net.Fit(XNORData)

	pred := net.Predict(XNORData)

	var inputVectorIndex int
	if xOne == 0 && xTwo == 0 {
		inputVectorIndex = 0
		fmt.Println(base.GetClass(pred, inputVectorIndex))
	} else if xOne == 0 && xTwo == 1 {
		inputVectorIndex = 1
		fmt.Println(base.GetClass(pred, inputVectorIndex))
	} else if xOne == 1 && xTwo == 0 {
		inputVectorIndex = 2
		fmt.Println(base.GetClass(pred, inputVectorIndex))
	} else if xOne == 1 && xTwo == 1 {
		inputVectorIndex = 3
		fmt.Println(base.GetClass(pred, inputVectorIndex))
	} else {
		fmt.Println("Your input is incorrect. Quitting. ")
	}

	fmt.Println("")

}
