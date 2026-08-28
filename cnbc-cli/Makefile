.PHONY: build test lint install clean

build:
	go build -o bin/cnbc-cli-pp-cli ./cmd/cnbc-cli-pp-cli

test:
	go test ./...

lint:
	golangci-lint run

install:
	go install ./cmd/cnbc-cli-pp-cli

clean:
	rm -rf bin/

build-mcp:
	go build -o bin/cnbc-cli-pp-mcp ./cmd/cnbc-cli-pp-mcp

install-mcp:
	go install ./cmd/cnbc-cli-pp-mcp

build-all: build build-mcp
