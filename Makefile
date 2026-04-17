CXX      = smpicxx
CXXFLAGS = -O2 -Wall
SRCDIR   = src
BINDIR   = bin
RESDIR   = results

SOURCES  = $(wildcard $(SRCDIR)/*.cpp)
TARGETS  = $(patsubst $(SRCDIR)/%.cpp, $(BINDIR)/%, $(SOURCES))

.PHONY: all clean

all: $(BINDIR) $(TARGETS)

$(BINDIR):
	mkdir -p $(BINDIR)

$(BINDIR)/%: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean-all:
	rm -rf $(BINDIR) $(RESDIR)

clean:
	rm -rf $(BINDIR)
