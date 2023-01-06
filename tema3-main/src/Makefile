# Mihnea Dinica 333CA
TARGETS=tema3

build: $(TARGETS)

tema3: tema3.cpp
	mpic++ tema3.cpp -o tema3

run: tema3
	mpirun --oversubscribe -np 12 ./tema3

clean:
	rm -rf tema3
.PHONY: build clean