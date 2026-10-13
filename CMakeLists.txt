cmake_minimum_required(VERSION 3.8)
project(arena_assets)

find_package(ament_cmake REQUIRED)

find_package(ament_cmake_venv REQUIRED)
find_package(ament_cmake_venv_uv REQUIRED)

uv_venv(DIRECTORY src TEST_EXTRA test)

venv_script("arena_assets" "-m arena_assets")

if(BUILD_TESTING)
  uv_venv_pytest(${PROJECT_NAME}_pytest tests TIMEOUT 300)
endif()

ament_package()
