# GO Prediction Results Heatmap Visualization

-his project generates heatmaps to compare Gene Ontology (GO) prediction results obtained from two methods: NODE2VEC and HOPE. It provides a clear and visual representation of predictions for the following GO categories:

* Biological Process (BP) (optional)
* Molecular Function (MF) (optional)
* Cellular Component (CC) (optional)


## Dependencies

 1.pandas
 2.numpy
 3.seaborn
 4.matplotlib

### Step 1: Prepare Input Files
Place the input files in the `data/` folder. Use the following naming conventions:

- **BP category**:
  - `go_pred_tableF1_BP_filter_node2vec.csv`
  - `go_pred_tableF1_BP_filter_HOPE.csv`

*(Optional)* Add MF and CC files with similar naming:
- `go_pred_tableF1_MF_filter_dropna_node2vec.csv`
- `go_pred_tableF1_MF_filter_dropna_HOPE.csv`
- `go_pred_tableF1_CC_filter_node2vec.csv`
- `go_pred_tableF1_CC_filter_HOPE.csv`

### Step 2: Run the Script
Navigate to the `scripts/` folder and run the Python script:

python heatmap_mf_bp_cc.py

### Step 3: View Output
The generated heatmap will be saved in the `output/` folder as:

BP_filter_heatmap_.png

### Customization
## Color Mapping
To change the NODE2VEC and HOPE colors, modify the following dictionary in the script:

group_color_dict = {
    "NODE2VEC": "orange",
    "HOPE": "blue"
}

## Heatmap Options
You can adjust the Seaborn `clustermap` parameters to:
- Change colormap (`cmap`) to styles like `coolwarm`, `viridis`, etc.
- Enable or disable clustering (`row_cluster` or `col_cluster`).

The output heatmap shows:
- Rows: Index values (e.g., GO terms or sample IDs).
- Columns: Prediction values for NODE2VEC and HOPE.
- Color-coded labels:
  - **Orange**: NODE2VEC
  - **Blue**: HOPE
  
### SUMMARY
This project generates heatmaps to compare Gene Ontology (GO) prediction results obtained from two methods: NODE2VEC and HOPE. It provides a clear and visual representation of predictions for the following GO categories:

Biological Process (BP) (optional)
Molecular Function (MF) (optional)
Cellular Component (CC) (optional)
Key Highlights
Input: CSV files containing prediction data for NODE2VEC and HOPE.
Integration: Combines results side-by-side with clear prefixes for identification.
Visualization: Generates clustered heatmaps:
Orange: NODE2VEC
Blue: HOPE
Customization:
Easily change colors, clustering, and output filenames.
Workflow
Prepare Input Files: Place CSVs in the data/ folder.
Run Script: Execute heatmap_visualization.py.
View Output: Heatmaps saved in the output/ folder as PNG files.