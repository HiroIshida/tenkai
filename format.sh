find src -type f \( -name "*.cpp" -o -name "*.hpp" \) | xargs clang-format -i -style=Chromium
