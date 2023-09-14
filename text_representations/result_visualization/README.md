
This repository contains Python scripts for analyzing prediction results and creating figures and significance tables for ontology-based function prediction.

# Dependencies
 1.	Python 3.7.3
 2.	pandas 1.1.4
 3.	scipy
 4.	seaborn
 5.	matplotlib
 6.	numpy
 7.	glob
 8.	os
 9.	statsmodels
 10.	math
 11.	ast

## Data

The function prediction data should be provided in CSV format with the following columns:

- Model: The name of the prediction model.
- Measure: The evaluation measure used (e.g., F1-Weighted, Accuracy, Precision).
- Aspect: The functional aspect (BP, CC, or MF).
- Value: The value of the evaluation measure for each model and aspect.

Please make sure the data is formatted correctly before running the scripts.

### Options

The script supports the following command-line options:

- `-f` or `--figures`: Create figures from the results.
- `-s` or `--significance`: Create significance tables from the results.
- `-rfp` or `--resultfilespath`: Path for the result files (required).
- `-a` or `--all`: Create both figures and significance tables.

At least one option should be selected. If no options are provided, an error message will be displayed.

### How to Run

Step by step operation:
  1. Clone repository
  2. Install dependencies(given above)
  3. Download and unzip result files to result_files folder from https://drive.google.com/file/d/1Y6WIfkM9IQakqvDJJHsIKrfY2Td3C7r8/view?usp=drive_link
  4. Run the script

Examples:

1. To create figures from the result files:

```
python visualize_results -f -rfp /path/to/result/files
```

2. To create significance tables from the result files:

```
python visualize_results -s -rfp /path/to/result/files
```

3. To create both figures and significance tables:

```
python visualize_results -a -rfp /path/to/result/files
```

Make sure to replace `/path/to/result/files` with the actual path to your result files.

## Definition of Output

The script will load the results and perform the selected actions based on the provided options. The output will be generated in the following manner:

- If the `-f` or `--figures` option is selected, figures will be created and saved.
- If the `-s` or `--significance` option is selected, significance tables will be created and saved.
- If the `-a` or `--all` option is selected, both figures and significance tables will be created and saved.
- The `figures` directory will contain the generated figures in PNG format.
- The `significance` directory will contain the calculated significance scores in CSV format.

## License

Copyright (C)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
