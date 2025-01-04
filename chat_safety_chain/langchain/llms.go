package langchain

import (
	"context"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"strings"
)

type ProtectiveLLMChat interface {
	PromptLLM(ctx context.Context, question string) (string, error)
}

type ProtectiveOllamaChat struct {
	ollamaGuard *ollama.LLM //this will be the guard, asserting that prompts received are 'safe for work'
	ollamaChat  *ollama.LLM // this will be the initialized llm to chat if the prompt is safe
}

func NewProtectiveOllamaChat() *ProtectiveOllamaChat {
	ollamaGuard, err := newOllamaGuard()
	if err != nil {
		panic(err)
	}
	ollamaChat, err := newOllamaChat()
	if err != nil {
		panic(err)
	}
	return &ProtectiveOllamaChat{
		ollamaGuard: ollamaGuard,
		ollamaChat:  ollamaChat,
	}
}

// newOllamaGuard is a wrapper of langchaingo to create the Llama guard model
func newOllamaGuard() (*ollama.LLM, error) {
	newOllama, err := ollama.New(ollama.WithModel("llama-guard3"))
	if err != nil {
		return nil, err
	}
	return newOllama, nil
}

// newOllamaChat is a wrapper of langchain to create a llama3.2 model to answer questions for
func newOllamaChat() (*ollama.LLM, error) {
	newOllama, err := ollama.New(ollama.WithModel("llama3.2"))
	if err != nil {
		return nil, err
	}
	return newOllama, nil
}

func (o *ProtectiveOllamaChat) PromptLLM(ctx context.Context, question string) (string, error) {
	// first prompt the Guard to see if this message is safe:
	content := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, question),
	}
	generateContent, err := o.ollamaGuard.GenerateContent(ctx, content)
	if err != nil {
		return "", err
	}
	if strings.Contains(strings.ToLower(generateContent.Choices[0].Content), "unsafe") {
		return "The query that you have provided is not allowed.", nil
	}

	// if its safe pass it over to the LLM:
	content = []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "You are a helpful, humble chat assistant."),
		llms.TextParts(llms.ChatMessageTypeHuman, question),
	}
	generateContent, err = o.ollamaChat.GenerateContent(ctx, content)
	if err != nil {
		return "", err
	}
	return generateContent.Choices[0].Content, nil
}

//TODO: Lets build tools to make this easier.
