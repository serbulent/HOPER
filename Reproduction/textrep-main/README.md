# textrep

This repository contains Python scripts for analyzing prediction results and creating figures and significance tables for ontology-based function prediction.

## Definition of Scripts

### create_figures.py

This script generates figures for ontology-based function prediction. It uses the following Python libraries:

- `pandas`: Used for data manipulation and analysis.
- `glob`: Used for searching for files using pattern matching.
- `os`: Used for interacting with the operating system.
- `seaborn`: Used for statistical data visualization.
- `numpy`: Used for mathematical operations on arrays.
- `matplotlib`: Used for creating plots and figures.

#### Functions

- `create_index_from_model_name(index_names)`: Creates an index list from model names.
- `create_pred_table(measure)`: Reads prediction results, orders them alphabetically, and creates a prediction table.
- `get_go_pred_table_for_aspect(aspect, go_pred_table)`: Slices the prediction table by aspect and orders subgroups.
- `prepare_figure_data_for_aspect(aspect)`: Calculates mean measures for different aspects and returns F1 weighted scores.
- `set_colors_and_marks_for_representation_groups(ax)`: Sets colors and marks for representation groups in a plot.
- `create_figures()`: Creates dataframes for figures and generates the figures.

### create_significance.py

This script calculates significance scores for ontology-based function prediction. It uses the following Python libraries:

- `pandas`: Used for data manipulation and analysis.
- `glob`: Used for searching for files using pattern matching.
- `os`: Used for interacting with the operating system.
- `numpy`: Used for mathematical operations on arrays.
- `ast`: Used for evaluating literal values from strings.
- `scipy`: Used for statistical computations.
- `math`: Used for mathematical operations.

#### Functions

- `calculate_q_vals(go_pred_score_table)`: Calculates q-values using the Benjamini/Hochberg method.
- `check_for_normality(go_pred_signinificance_score_df)`: Checks if data is drawn from a normal distribution.
- `nan_to_zero(x)`: Replaces NaN values with zero.
- `create_significance_tables()`: Creates significance tables for different aspects.

### visualize_results.py

This script serves as the main entry point for running the analysis. It uses the following Python libraries:

- `argparse`: Used for parsing command-line arguments.
- `create_figures`: A module that contains functions for creating figures.
- `create_significance`: A module that contains functions for calculating significance scores.
- `pandas`: Used for data manipulation and analysis.
- `tqdm`: Used for creating progress bars.

#### Functions

- `main()`: The main function that parses command-line arguments and runs the analysis.

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
