if(APPLE)
	list(PREPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
	list(PREPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
endif()

find_package(benchmark)

if(benchmark_FOUND)
    add_executable(vbz_hdf_perf_test
        vbz_hdf_perf.cpp
    )
    add_sanitizers(vbz_hdf_perf_test)

    target_link_libraries(vbz_hdf_perf_test
        PRIVATE
            benchmark::benchmark
            ${HDF5_C_LIBRARIES}
            hdf_test_utils
            vbz_hdf_plugin
    )

    set_property(TARGET vbz_hdf_perf_test PROPERTY CXX_STANDARD 11)

    add_test(
        NAME vbz_hdf_perf_test
        COMMAND vbz_hdf_perf_test
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_executable(vbz_fast5_perf
        vbz_fast5_perf.cpp
    )
    add_sanitizers(vbz_fast5_perf)

    target_link_libraries(vbz_fast5_perf
        PRIVATE
            vbz
            benchmark::benchmark
            ${HDF5_C_LIBRARIES}
            hdf_test_utils
    )

    target_include_directories(vbz_fast5_perf
        PRIVATE
            ${HDF5_INCLUDE_DIRS}
    )

    set_property(TARGET vbz_fast5_perf PROPERTY CXX_STANDARD 11)
else()
    message(STATUS "Google Benchmark not found — skipping vbz_hdf_perf_test and vbz_fast5_perf")
endif()

add_executable(vbz_fast5_throughput
    vbz_fast5_throughput.cpp
)
add_sanitizers(vbz_fast5_throughput)

target_link_libraries(vbz_fast5_throughput
    PRIVATE
        vbz
        ${HDF5_C_LIBRARIES}
        hdf_test_utils
)

target_include_directories(vbz_fast5_throughput
    PRIVATE
        ${HDF5_INCLUDE_DIRS}
)

set_property(TARGET vbz_fast5_throughput PROPERTY CXX_STANDARD 11)
