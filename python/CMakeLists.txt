# Explicit Python3 search (before pybind11 to control version)
find_package(Python3 3.11 EXACT COMPONENTS Interpreter Development QUIET)
if(NOT Python3_FOUND)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
endif()

# Find Pybind11
find_package(pybind11 CONFIG REQUIRED)
if(NOT pybind11_FOUND)
  execute_process(COMMAND python -m pybind11 --cmake-prefix-path 
    OUTPUT_VARIABLE pybind11_DIR 
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED)
endif()

# Python extension module
pybind11_add_module(pysuperansac MODULE
  pybind_module.cpp
  pybind_functions.cpp
)

# Workaround for pybind11 + Python 3.11 compatibility
target_compile_options(pysuperansac PRIVATE -Wno-error)

# Link the C++ core library
target_link_libraries(pysuperansac PRIVATE SupeRANSAC)

# Ensure loader finds the sibling core .so at runtime
set_target_properties(pysuperansac PROPERTIES
  BUILD_RPATH   "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN NO
)

# LTO can stay enabled globally; disable here if needed
# set_property(TARGET pysuperansac PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE)

# Install module next to the core .so
install(TARGETS pysuperansac
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
  ARCHIVE DESTINATION .
)
