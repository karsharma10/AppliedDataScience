package langchain

import (
	"github.com/tmc/langchaingo/llms/ollama"
)

type ProtectiveLLMChat interface {
	promptLLM(guard *ollama.LLM, llm *ollama.LLM) (string, error)
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
