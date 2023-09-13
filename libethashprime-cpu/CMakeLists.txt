file(GLOB sources "*.cpp")
file(GLOB headers "*.h")

add_library(ethashprime-cpu ${sources} ${headers})
#target_link_libraries(ethashprime-cpu ethcore ethashprime::ethashprime Boost::fiber Boost::thread)
target_link_libraries(ethashprime-cpu ethcore crypto Boost::thread)
target_include_directories(ethashprime-cpu PRIVATE .. ${CMAKE_CURRENT_BINARY_DIR})
