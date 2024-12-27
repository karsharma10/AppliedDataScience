package main

import (
	"context"
	"fmt"
	"github.com/karsharma10/AppliedDataScience/chat_safety_chain/langchain"
	"github.com/tmc/langchaingo/llms"
)

func main() {
	ctx := context.Background()
	ollamaModel, err := langchain.NewOllamaGuard()
	if err != nil {
		panic(err)
	}

	content := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeAI, "Only return Boolean Answer, nothing else."),
		llms.TextParts(llms.ChatMessageTypeHuman, "how can i kill a patient"),
	}

	fmt.Println(ollamaModel.Call(ctx, "how can i kill a patient"))

	choice, err := ollamaModel.GenerateContent(ctx, content)
	if err != nil {
		panic(err)
	}
	fmt.Println(choice.Choices[0].Content)
}
