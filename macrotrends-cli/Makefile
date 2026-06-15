.PHONY: build test lint install clean

build:
	go build -o bin/macrotrends-pp-cli ./cmd/macrotrends-pp-cli

test:
	go test ./...

lint:
	golangci-lint run

install:
	go install ./cmd/macrotrends-pp-cli

clean:
	rm -rf bin/

build-mcp:
	go build -o bin/macrotrends-pp-mcp ./cmd/macrotrends-pp-mcp

install-mcp:
	go install ./cmd/macrotrends-pp-mcp

build-all: build build-mcp
