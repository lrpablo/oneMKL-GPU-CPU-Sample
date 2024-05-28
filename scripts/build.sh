#/bin/sh

# Check for argument and set LOCALMKL
if [ ! -z "$1" ]; then
  LOCALMKL="$1"
else
  # Check for SYCL_MKL environment variable
  if [ -z "${SYCL_MKL}" ]; then
    echo "Error: Missing sycl_mkl argument and SYCL_MKL environment variable."
    exit 1
  else
    LOCALMKL="${SYCL_MKL}"
    echo "Using SYCL_MKL environment variable: $LOCALMKL"
  fi
fi

rm -rf build && mkdir build && cd build

cmake .. -DSYCL_MKLROOT=${LOCALMKL}

make
