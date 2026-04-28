# dnk-cpp-infernence-test

A minimal C++ 20 denkweit inference test project. Tested on x86_64 included libdenkflow.so is for x86_64 linux version 0.7.7-b8. As inputs a sample cookie hole detection was used.

## Build

```bash
cmake -Bbuild -H.
cmake --build build -j8
```

## Run

```bash
./build/dnk-inference-cpp \
  --pat <Personal Access Token> \
  --model <path to .denkflow model> \
  --input <input image path>
```

- `--pat`: Personal Access Token for authentication
- `--model`: Path to the `.denkflow` model file
- `--input`: Path to the input image (loaded via `cv::imread`)