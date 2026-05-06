#!/bin/bash

# Get a count of the Level 3 CAM and PAM products per date and per data version number.
# With no arguments, automatically determine the most recent 2 version numbers and query those.
# With 1 or 2 arguments, query files of those versions.
# Output is a CSV file with columns such as: "Path,Total_FITS_Files,v0j,v0k"

# Check if too many arguments are provided (zero, one, or two allowed)
if [ $# -gt 2 ]; then
    echo "Usage: $0 [pattern1 [pattern2]]"
    echo "Example: $0"
    echo "Example: $0 v0j v0k"
    exit 1
fi

# Navigate to the target directory
cd /mnt/archive/soc/data/3 || exit 1

# Get data version(s) to query
if [ $# -ge 1 ]; then
# use the 1 or 2 versions given as arguments
    PATTERN1=$1
    PATTERN2=$2
elif [ $# -eq 0 ]; then
# determine the latest two versions automatically
    output=$(find CAM PAM -name "*.fits" | cut -d _ -f 5 | cut -d . -f 1 | sort | uniq)
    PATTERN1=$(echo "$output" | tail -2 | head -1)  # Second-to-last line (if there are >1 versions)
    lc=$(echo "$output" | wc -l)
    if [ "$lc" -gt 1 ]; then
        PATTERN2=$(echo "$output" | tail -1)        # Last line
    fi
fi

# Output CSV file
OUTPUT_FILE=L3_CAM_PAM_filecounts.csv

# Prepare the output file with the CSV header
if [ -z "$PATTERN2" ]; then
    echo "Path,Total_FITS_Files,$PATTERN1" > "$OUTPUT_FILE"
else
    echo "Path,Total_FITS_Files,$PATTERN1,$PATTERN2" > "$OUTPUT_FILE"
fi

# Collect total number of files
find CAM PAM -name "*.fits" | cut -d / -f '1,2,3,4' | sort | uniq -c | awk -v pattern="$PATTERN1" '{print $2","$1}' > "/tmp/pattern0.csv"

# Collect results for both versions
# For PATTERN1
find CAM PAM -name "*_${PATTERN1}.fits" | cut -d / -f '1,2,3,4' | sort | uniq -c | awk -v pattern="$PATTERN1" '{print $2","$1}' > "/tmp/pattern1.csv"

# Join those two files together
join -t"," -j 1 -a 1 -a 2 -e "0" -o 0,1.2,2.2 "/tmp/pattern0.csv" "/tmp/pattern1.csv" > "/tmp/pattern01.csv"

# For PATTERN2 (if provided)
if [ -n "$PATTERN2" ]; then
    find CAM PAM -name "*_${PATTERN2}.fits" | cut -d / -f '1,2,3,4' | sort | uniq -c | awk -v pattern="$PATTERN2" '{print $2","$1}' > "/tmp/pattern2.csv"

    # Join results from both patterns into a single CSV
    join -t "," -j 1 -a 1 -a 2 -e "0" -o 0,1.2,1.3,2.2 "/tmp/pattern01.csv" "/tmp/pattern2.csv" >> "$OUTPUT_FILE"
else
    # Just output results for PATTERN1 if there's no PATTERN2
    cat "/tmp/pattern01.csv" >> "$OUTPUT_FILE"
fi

# Go back to the original directory
cd $OLDPWD || exit 1

# Cleanup temporary files
rm -f /tmp/pattern0.csv /tmp/pattern1.csv /tmp/pattern01.csv /tmp/pattern2.csv

#echo "Results written to $OUTPUT_FILE"
