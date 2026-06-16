#!/bin/bash
# Launch TomatoGuard Streamlit app using the ml conda environment
/Users/ac/miniforge3/envs/ml/bin/streamlit run "$(dirname "$0")/app.py"
