# Find Eigen3
find_package(Eigen3 REQUIRED)

# Find Boost
find_package(Boost REQUIRED COMPONENTS system)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Recursively find all .cc and .cpp files
file(GLOB_RECURSE SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/include/*.cc
  ${PROJECT_SOURCE_DIR}/include/*.cpp
)

# Core shared library
add_library(SupeRANSAC SHARED
  superansac.cpp
  ${SOURCE_FILES}
)

# Give a unique, stable SONAME to avoid stale copies being picked at runtime
set_target_properties(SupeRANSAC PROPERTIES
  OUTPUT_NAME superansac_core
  SOVERSION 1
  VERSION   1.0.0
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN NO
)

# On MSVC, /GL produces LTCG "anonymous" .obj files that dumpbin cannot parse,
# which breaks WINDOWS_EXPORT_ALL_SYMBOLS (cmake -E __create_def yields an empty
# .def). Disable IPO/GL specifically for this shared-library target so that the
# auto-export mechanism works. The pybind module (which links this .lib) can
# still use LTO via the global setting.
if(MSVC)
  set_property(TARGET SupeRANSAC PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
  set_property(TARGET SupeRANSAC PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
  # Strip /GL from this target's compile options (it was added globally)
  target_compile_options(SupeRANSAC PRIVATE $<$<CONFIG:Release>:/GL->)
endif()

# Public includes
target_include_directories(SupeRANSAC PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Link deps
target_link_libraries(SupeRANSAC
  Eigen3::Eigen
  Boost::system
  ${OpenCV_LIBS}
)

# Install next to the Python module
install(TARGETS SupeRANSAC
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
  ARCHIVE DESTINATION .
)
