CXX      ?= g++
CXXFLAGS ?= -std=c++11 -O2 -Wall -Wextra -Wno-unused-parameter

SKETCH := ../arduino/softmax_power_bench

test_softmax: test_softmax.cpp $(SKETCH)/softmax_kernels.cpp $(SKETCH)/softmax_kernels.h
	$(CXX) $(CXXFLAGS) -o $@ test_softmax.cpp $(SKETCH)/softmax_kernels.cpp -lm

.PHONY: check clean avr-check avr-bench avr-size all
all: check avr-check

check: test_softmax
	./test_softmax

# ---- AVR verification (needs gcc-avr, avr-libc, simavr) ----------------
# Runs the kernels on an emulated ATmega328P. Catches 16-bit-int and
# PROGMEM bugs that the host build cannot see.

AVR_MCU   ?= atmega328p
AVR_F_CPU ?= 16000000UL
AVR_FLAGS := -mmcu=$(AVR_MCU) -DF_CPU=$(AVR_F_CPU) -Os -std=gnu++11 \
             -Wall -Wextra -ffunction-sections -fdata-sections

avr_selftest.elf: avr_selftest.cpp $(SKETCH)/softmax_kernels.cpp $(SKETCH)/softmax_kernels.h
	avr-g++ $(AVR_FLAGS) -o $@ avr_selftest.cpp $(SKETCH)/softmax_kernels.cpp -Wl,--gc-sections -lm

avr-check: avr_selftest.elf
	simavr -m $(AVR_MCU) -f 16000000 avr_selftest.elf 2>&1 | python3 check_avr_output.py

avr_bench.elf: avr_bench.cpp $(SKETCH)/softmax_kernels.cpp $(SKETCH)/softmax_kernels.h
	avr-g++ $(AVR_FLAGS) -o $@ avr_bench.cpp $(SKETCH)/softmax_kernels.cpp -Wl,--gc-sections -lm

# Cycles per call on an emulated ATmega328P, for the reference table.
avr-bench: avr_bench.elf
	@simavr -m $(AVR_MCU) -f 16000000 avr_bench.elf 2>&1 \
	  | sed -e 's/\x1b\[[0-9;]*m//g' -e 's/\.$$//' \
	  | grep -vE '^(Loaded|$$)'

# Flash/SRAM cost of the kernels alone, for the footprint table.
avr-size: $(SKETCH)/softmax_kernels.cpp
	avr-g++ $(AVR_FLAGS) -c -o /tmp/sm_kernels_avr.o $<
	avr-nm --print-size --size-sort -C /tmp/sm_kernels_avr.o | grep -iE ' [tTrR] '

clean:
	rm -f test_softmax avr_selftest.elf avr_bench.elf
