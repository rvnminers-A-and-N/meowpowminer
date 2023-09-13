set(SOURCES
	Farm.cpp Farm.h
	Miner.h Miner.cpp
)

include_directories(BEFORE ..)

add_library(ethcore ${SOURCES})
target_link_libraries(ethcore PUBLIC devcore crypto PRIVATE hwmon)

if(ETHASHPRIMECL)
	target_link_libraries(ethcore PRIVATE ethashprime-cl)
endif()
if(ETHASHPRIMECUDA)
	target_link_libraries(ethcore PUBLIC ethashprime-cuda)
endif()
if(ETHASHPRIMECPU)
	target_link_libraries(ethcore PUBLIC ethashprime-cpu)
endif()
