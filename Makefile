OS := $(shell uname)
CFLAGS := -Wall -Wextra -Werror -Wpedantic
ifeq ($(OS), Darwin)
CC := clang
else
CC := gcc
CFLAGS += -march=native
endif

bin := crc32

.PHONY: all clean

all: $(bin)

$(bin): main.o
	$(CC) $^ -o $(bin)

main.o: main.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(obj) $(bin)

