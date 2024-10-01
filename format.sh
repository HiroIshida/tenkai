directories=("src" "include")
for dir in "${directories[@]}"; do
    find ${dir} -type f \( -name "*.cpp" -o -name "*.hpp" \) | xargs clang-format -i -style="{ BasedOnStyle: Chromium, ColumnLimit: 100 }"
done
