cmake_minimum_required(VERSION 3.16)
project(A-CORE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(OpenMP REQUIRED)

# ---------------------------------------------------------------------------
# 全局包含路径：hnswlib / json.hpp（核心框架依赖，header-only）
# ---------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

# 通用编译选项：所有可执行目标都用 C++17 + OpenMP
function(acore_add_executable TARGET SOURCE)
  add_executable(${TARGET} ${SOURCE})
  target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
  target_compile_features(${TARGET} PRIVATE cxx_std_17)
endfunction()

# ===========================================================================
# A-CORE 核心
# ===========================================================================
acore_add_executable(run_acore                      src/run_acore.cpp)
acore_add_executable(run_acore_online_grouping      src/run_acore_online_grouping.cpp)
acore_add_executable(projected_greedy_cluster       src/projected_greedy_cluster.cpp)
acore_add_executable(run_train_get_all_k_onlytop    src/run_train_get_all_k_onlytop.cpp)
acore_add_executable(augment_features               src/augment_features.cpp)
acore_add_executable(make_clustered_queries5        src/make_clustered_queries5.cpp)

# ===========================================================================
# 基线：自包含（无外部依赖）
# ===========================================================================
acore_add_executable(run_hnsw_baseline              baselines/hnsw/run_hnsw_baseline.cpp)
acore_add_executable(build_nsg                      baselines/nsg/build_nsg.cpp)
acore_add_executable(search_nsg                     baselines/nsg/search_nsg.cpp)
acore_add_executable(search_nsg_cluster             baselines/nsg/search_nsg_cluster.cpp)
acore_add_executable(tming_build                    baselines/taumng/tming_build.cpp)
acore_add_executable(search_mrng                    baselines/taumng/search_mrng.cpp)

# ===========================================================================
# 基线：外部依赖（可选，未提供依赖时跳过）
# ---------------------------------------------------------------------------
# RoarGraph  : -DROARGRAPH_DIR=<roargraph 仓库根目录>
# NGFix      : -DNGFIX_DIR=<NGFix 仓库根目录>
# Faiss      : -DFAISS_INSTALL_DIR=<faiss install 目录>
# ===========================================================================

# --- RoarGraph ---
set(ROARGRAPH_DIR "" CACHE PATH "RoarGraph repository root (contains include/efanna2e, include/index_bipartite.h)")
if(ROARGRAPH_DIR AND EXISTS "${ROARGRAPH_DIR}/include/index_bipartite.h")
  add_executable(search_roargraph_fair_base baselines/roargraph/search_roargraph_fair_base.cpp)
  target_include_directories(search_roargraph_fair_base PRIVATE
    "${ROARGRAPH_DIR}/include"
    "${ROARGRAPH_DIR}/include/efanna2e")
  target_link_libraries(search_roargraph_fair_base PRIVATE OpenMP::OpenMP_CXX)
  target_compile_features(search_roargraph_fair_base PRIVATE cxx_std_17)
  message(STATUS "RoarGraph baseline: enabled (${ROARGRAPH_DIR})")
else()
  message(STATUS "RoarGraph baseline: skipped (set -DROARGRAPH_DIR=<path>)")
endif()

# --- NGFix ---
set(NGFIX_DIR "" CACHE PATH "NGFix repository root (contains ngfixlib/graph/hnsw_ngfix.h)")
if(NGFIX_DIR AND EXISTS "${NGFIX_DIR}/ngfixlib/graph/hnsw_ngfix.h")
  add_executable(search_hnsw_ngfix_fair_base baselines/ngfix/search_hnsw_ngfix_fair_base.cpp)
  target_include_directories(search_hnsw_ngfix_fair_base PRIVATE "${NGFIX_DIR}")
  target_link_libraries(search_hnsw_ngfix_fair_base PRIVATE OpenMP::OpenMP_CXX)
  target_compile_features(search_hnsw_ngfix_fair_base PRIVATE cxx_std_17)
  message(STATUS "NGFix baseline: enabled (${NGFIX_DIR})")
else()
  message(STATUS "NGFix baseline: skipped (set -DNGFIX_DIR=<path>)")
endif()

# --- Faiss ---
set(FAISS_INSTALL_DIR "" CACHE PATH "Faiss install directory")
if(FAISS_INSTALL_DIR AND EXISTS "${FAISS_INSTALL_DIR}/include/faiss/IndexFlat.h")
  add_executable(run_faiss_baseline baselines/faiss/run_faiss_baseline.cpp)
  target_include_directories(run_faiss_baseline PRIVATE "${FAISS_INSTALL_DIR}/include")
  target_link_directories(run_faiss_baseline PRIVATE "${FAISS_INSTALL_DIR}/lib")
  target_link_libraries(run_faiss_baseline PRIVATE faiss faiss_avx2 openblas lapack gomp pthread dl)
  target_compile_features(run_faiss_baseline PRIVATE cxx_std_17)
  set_target_properties(run_faiss_baseline PROPERTIES
    INSTALL_RPATH "${FAISS_INSTALL_DIR}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE)
  message(STATUS "Faiss baseline: enabled (${FAISS_INSTALL_DIR})")
else()
  message(STATUS "Faiss baseline: skipped (set -DFAISS_INSTALL_DIR=<path>)")
endif()

# ---------------------------------------------------------------------------
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenMP: ${OpenMP_CXX_FLAGS}")
