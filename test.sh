#!/bin/bash


# Convert scientific notation to integer using Python
# int_arg=$(python3 -c "print(int($my_number))")
int_arg=$(awk -v x="2.2e+6" 'BEGIN {printf("%d\n",x)}')


# Define your default argument value (None in this case)
# int_arg=-1

# Cast the floating-point number to an integer.
echo "int: $int_arg"

# Execute your Python script with the argument
python3 my_script.py --my_arg "$int_arg"