#!/bin/bash



if [ "$#" -lt 4 ]; then
    echo "Usage:"
    echo "./run_tests.sh <predictor_name> <warmup> <simulation> <trace1> [trace2] [trace3] ..."
    exit 1
fi

PREDICTOR=$1
WARMUP=$2
SIMULATION=$3
shift 3
TRACES=("$@")

CONFIG_FILE="champsim_config.json"

echo "========================================="
echo "Predictor: $PREDICTOR"
echo "Warmup: $WARMUP"
echo "Simulation: $SIMULATION"
echo "========================================="

# ------------------------------------------------
# Step 1: Modify JSON config (branch predictor)
# ------------------------------------------------
echo "Updating config file..."

# Replace branch predictor in JSON
sed -i "s/\"branch_predictor\": *\"[^\"]*\"/\"branch_predictor\": \"$PREDICTOR\"/g" $CONFIG_FILE

# ------------------------------------------------
# Step 2: Rebuild ChampSim
# ------------------------------------------------
echo "Cleaning previous build..."
rm -rf .csconfig
make clean

echo "Reconfiguring..."
./config.sh $CONFIG_FILE

echo "Building..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Build failed. Exiting."
    exit 1
fi

# ------------------------------------------------
# Step 3: Run traces
# ------------------------------------------------

echo ""
echo "Running simulations..."
echo "-----------------------------------------"

for TRACE in "${TRACES[@]}"; do

    echo ""
    echo "Trace: $TRACE"
    echo "-----------------------------------------"

    OUTPUT=$(bin/champsim \
        --warmup-instructions $WARMUP \
        --simulation-instructions $SIMULATION \
        $TRACE)

    ACCURACY=$(echo "$OUTPUT" | grep "Branch Prediction Accuracy" | awk '{print $6}')
    MPKI=$(echo "$OUTPUT" | grep "Branch Prediction Accuracy" | awk '{print $8}')

    echo "Accuracy: $ACCURACY"
    echo "MPKI: $MPKI"

done

echo ""
echo "All tests complete."
echo "========================================="
