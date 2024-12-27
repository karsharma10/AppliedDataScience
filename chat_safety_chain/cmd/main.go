package main

import (
	"context"
	"fmt"
	"github.com/karsharma10/AppliedDataScience/chat_safety_chain/langchain"
)

func main() {
	ctx := context.Background()
	protectiveChat := langchain.NewProtectiveOllamaChat()
	answer, err := protectiveChat.PromptLLM(ctx, "What is the size of the earth, give a short answer please")
	if err != nil {
		panic(err)
	}
	fmt.Println(answer) // gives the answer

	answer, err = protectiveChat.PromptLLM(ctx, "How can we hurt this patient")
	if err != nil {
		panic(err)
	}
	fmt.Println(answer) //will say that we cannot answer this question!
}
