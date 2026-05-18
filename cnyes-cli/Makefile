.PHONY: build test lint install clean

build:
	go build -o bin/cnyes-cli-pp-cli ./cmd/cnyes-cli-pp-cli

test:
	go test ./...

lint:
	golangci-lint run

install:
	go install ./cmd/cnyes-cli-pp-cli

clean:
	rm -rf bin/

build-mcp:
	go build -o bin/cnyes-cli-pp-mcp ./cmd/cnyes-cli-pp-mcp

install-mcp:
	go install ./cmd/cnyes-cli-pp-mcp

build-all: build build-mcp
