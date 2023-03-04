#_____________________________________________________________________________
# VARIABLES
#
# PROJECT NAME
BASE_NAME =conv_testing
# 
# EXECUTABLES
EXECUTABLE       =$(BASE_NAME)
EXECUTABLE_GCOV  =$(EXECUTABLE)_gcov
# 
# SOURCE FILENAMES
MAIN_C =test.cpp -ljsoncpp
SRC_C  =convolution.c 
SRC_H  =convolution.h 
#
# cleaning
RM_FILES =test_outs/*
# 
# SYSTEM
SHELL           =/bin/bash
CC              =g++
CFLAGS          =-g -std=c++2a -Wall -Wshadow -Wvla -Werror #  -D DEBUGGING
DBG_FLAGS       =-g std=c++2a -D DEBUGGING
CFLAGS_GCOV     =$(DBG_FLAGS) -fprofile-arcs -ftest-coverage
MEM_FLAGS       =--leak-check=full  --show-leak-kinds=all --track-origins=yes --verbose
OUT				=echo -e -n
#_____________________________________________________________________________
# RULES
#
$(EXECUTABLE): $(SRC_C) $(MAIN_C) $(SRC_H)
	$(CC) $(CFLAGS) -o $(EXECUTABLE) $(MAIN_C) $(SRC_C) 
	$(OUT) "executable: \"$(EXECUTABLE)\" was sucessfully created\n"

# fill the rest of the args later
grind: $(EXECUTABLE)
	valgrind $(MEM_FLAGS) ./$(EXECUTABLE) 

clean:
	$(OUT) "Files removed on clean: "
	rm -vf $(EXECUTABLE) $(EXECUTABLE_GCOV) $(RM_FILES) *.c.gcov *.gcno *.gcda | wc -l

clean_crash:
	$(OUT) "Files removed on clean: "
	rm -vf .*.c.swp .*.swp .*.h.swp | wc -l

coverage: $(SRC_C) $(MAIN_C) $(SRC_H)
	$(CC) $(CFLAGS) -o $(EXECUTABLE_GCOV) $(MAIN_C) $(SRC_C) 
	./$(EXECUTABLE_GCOV)
	gcov -f $(SRC_C)

.PHONY:  clean clean_crash coverage grind grind_test editc
.SILENT: clean clean_crash coverage grind grind_test editc # $(EXECUTABLE) 
# vim set noexpandtab tabstop=4 filetype=make: