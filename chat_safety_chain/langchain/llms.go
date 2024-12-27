package langchain

import (
	"context"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"strings"
)

type ProtectiveLLMChat interface {
	PromptLLM(ctx context.Context, guard *ollama.LLM, llm *ollama.LLM, question string) (string, error)
}

type ProtectiveOllamaChat struct {
	OllamaGuard *ollama.LLM //this will be the guard, asserting that prompts received are 'safe for work'
	OllamaChat  *ollama.LLM // this will be the initialized llm to chat if the prompt is safe
}

// NewOllamaGuard is a wrapper of langchaingo to create the Llama guard model
func (o *ProtectiveOllamaChat) NewOllamaGuard() (*ollama.LLM, error) {
	newOllama, err := ollama.New(ollama.WithModel("llama-guard3"))
	if err != nil {
		return nil, err
	}
	return newOllama, nil
}

// NewOllamaChat is a wrapper of langchain to...
func (o *ProtectiveOllamaChat) NewOllamaChat() (*ollama.LLM, error) {
	newOllama, err := ollama.New(ollama.WithModel("llama-guard3"))
	if err != nil {
		return nil, err
	}
	return newOllama, nil
}

func (o *ProtectiveOllamaChat) PromptLLM(ctx context.Context, guard *ollama.LLM, llm *ollama.LLM, question string) (string, error) {
	// first prompt the Guard to see if this message is safe:
	content := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, question),
	}
	generateContent, err := guard.GenerateContent(ctx, content)
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
	generateContent, err = llm.GenerateContent(ctx, content)
	if err != nil {
		return "", err
	}
	return generateContent.Choices[0].Content, nil
}
