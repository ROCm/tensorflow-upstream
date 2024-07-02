#!/bin/bash

input_file="$1"
output_file="$2"

# Debugging: Print the input and output file paths
echo "Input file: $input_file"
echo "Output file: $output_file"

# Debugging: Check if the input file exists
if [ ! -f "$input_file" ]; then
  echo "Input file not found: $input_file"
  exit 1
fi

# Debugging: Print the contents of the input file directory
echo "Contents of the input file directory:"
ls -l "$(dirname "$input_file")"

awk '{ gsub(/^#cmakedefine DTYPES \"@DTYPES@\"/, "/* #undef DTYPES*/");
       gsub(/^#cmakedefine CK_ENABLE_ALL_DTYPES @CK_ENABLE_ALL_DTYPES@/, "#define CK_ENABLE_ALL_DTYPES ON");
       gsub(/^#cmakedefine CK_ENABLE_INT8 @CK_ENABLE_INT8@/, "/* #undef CK_ENABLE_INT8*/");
       gsub(/^#cmakedefine CK_ENABLE_FP8 @CK_ENABLE_FP8@/, "/* #undef CK_ENABLE_FP8*/");
       gsub(/^#cmakedefine CK_ENABLE_BF8 @CK_ENABLE_BF8@/, "/* #undef CK_ENABLE_BF8*/");
       gsub(/^#cmakedefine CK_ENABLE_FP16 @CK_ENABLE_FP16@/, "/* #undef CK_ENABLE_FP16*/");
       gsub(/^#cmakedefine CK_ENABLE_BF16 @CK_ENABLE_BF16@/, "/* #undef CK_ENABLE_BF16*/");
       gsub(/^#cmakedefine CK_ENABLE_FP32 @CK_ENABLE_FP32@/, "/* #undef CK_ENABLE_FP32*/");
       gsub(/^#cmakedefine CK_ENABLE_FP64 @CK_ENABLE_FP64@/, "/* #undef CK_ENABLE_FP64*/");
       gsub(/^#cmakedefine CK_ENABLE_DL_KERNELS @CK_ENABLE_DL_KERNELS@/, "/* #undef CK_ENABLE_DL_KERNELS*/");
       gsub(/^#cmakedefine CK_ENABLE_INSTANCES_ONLY @CK_ENABLE_INSTANCES_ONLY@/, "/* #undef CK_ENABLE_INSTANCES_ONLY*/");
       gsub(/^#cmakedefine CK_USE_XDL @CK_USE_XDL@/, "#define CK_USE_XDL ON");
       gsub(/^#cmakedefine CK_USE_WMMA @CK_USE_WMMA@/, "/* #undef CK_USE_WMMA*/");
       gsub(/^#cmakedefine/, "//cmakedefine"); print;
     }' "$input_file" > "$output_file"
