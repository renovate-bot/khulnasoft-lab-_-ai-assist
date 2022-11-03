develop-local:
	docker-compose -f docker-compose.dev.yaml up --build --remove-orphans

clean:
	docker-compose -f docker-compose.dev.yaml rm -s -v -f

.PHONY: all test test
