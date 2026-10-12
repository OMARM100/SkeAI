PYTHON ?= python
CXX ?= c++

.PHONY: all build check-sources test clean info

all: test

build:
	$(PYTHON) -m tools.build_cpp
	$(PYTHON) -m compileall -q src tests tools training benchmarks

check-sources:
	$(PYTHON) -m tools.check_sources

test: check-sources build
	$(PYTHON) -m unittest discover -s tests -v

clean:
	$(PYTHON) -m tools.clean

info:
	$(PYTHON) -c "from pathlib import Path; roots=('src','cpp','tests','tools','training','benchmarks'); exts={'.py','.c','.h','.cc','.cpp','.cxx','.hpp','.hh','.asm','.s','.S'}; files=[p for r in roots if Path(r).exists() for p in Path(r).rglob('*') if p.is_file() and p.suffix in exts]; [print('{}: {}'.format(p.suffix,p)) for p in sorted(files)]; print('Total source files:',len(files))"
