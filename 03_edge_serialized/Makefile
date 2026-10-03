CC = gcc
CFLAGS = -Wall -Wextra -O3 -std=c99 -Iinclude -Isrc
LDFLAGS = -lm

TARGET = edge_simulator
SRCS = main_edge_test.c src/tinyml_infer.c src/sms_alert_encoder.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
