DIR_INC = ./include
DIR_SRC = ./src
DIR_OBJ = ./obj
DIR_BIN = ./bin
SRC = $(wildcard ${DIR_SRC}/*.cpp)  
OBJ = $(patsubst %.cpp,${DIR_OBJ}/%.o,$(notdir ${SRC})) 
TARGET = cars
BIN_TARGET = ${DIR_BIN}/${TARGET}
CC = g++
CFLAGS = -fopenmp -g -Wall -I${DIR_INC}
${BIN_TARGET}:${OBJ}
	$(CC)  $(CFLAGS)  $(OBJ)  -o $@
${DIR_OBJ}/%.o:${DIR_SRC}/%.cpp
	$(CC)  $(CFLAGS) -c  $< -o $@
.PHONY:clean
clean:
	find ${DIR_OBJ} -name *.o -exec rm -rf {}
