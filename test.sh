#!/bin/bash

# Define the number in scientific notation
my_number="2.2e+6"

# Convert scientific notation to integer using Python
# int_arg=$(python3 -c "print(int($my_number))")
# int_arg=awk -v x="$my_number" 'BEGIN {printf("%d\n",x)}'
printf -v my_integer "%.0f" "2.2e+6"

echo "Original number: $my_number"
echo "As an integer: $my_integer"


# Define your default argument value (None in this case)
# int_arg=-1
# int_arg=$(( 2.2 * 10**6 )) # 1e+6

# # Cast the floating-point number to an integer.
# echo "int: $int_arg"
# # Execute your Python script with the argument
# python3 my_script.py --my_arg "$int_arg"