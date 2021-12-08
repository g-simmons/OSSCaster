LINEPLOT_STYLE = {
    "width": "100%",
    "height": "40vh",
}
EXPLANATIONS_PLOT_STYLE = {
    # "width": "100%",
    "height": "40vh",
}
# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "20rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


FIGURE_MARGINS = dict(l=0, r=0, t=0, b=0)

INSTRUCTIONS = """
### Instructions
- Upload your project history by clicking the upload button
- Your project history will be displayed in a line plot and the editable data table
- Click cells in the table to edit them (this allows you to explore predicted success )
- Mouse over months to see detailed project information for that month
- Click within the line plot to lock the currently selected month to the Month Display panel
- View feature importances for the project in the global importances panel to the right
- View feature importances for a specific month by selecting a month and clicking "Explain"

"""

DATATABLE_STYLE = {"fontSize": 12, "font-family": "sans-serif"}

MONTH_FEATURES_STYLE = {"font-size": 8, "margin": 0}

# MODEL_PATH = "./model.h5"

# REQUIRED_FEATURES = [
#     "active_devs",
#     "e_edges",
#     "num_commits",
#     "inactive_e",
#     "e_nodes",
#     "inactive_c",
#     "e_triangles",
#     "c_nodes",
#     "skew_c",
#     "e_bidirected_edges",
#     "skew_e",
#     "c_edges",
#     "num_respondents",
#     "c_mean_degree",
#     "c_triangles",
#     "e_long_tail",
#     "c_long_tail",
#     "num_senders",
#     "num_committers",
#     "num_files",
#     "num_emails",
#     "e_c_coef",
#     "e_mean_degree",
#     "c_c_coef",
#     "e_percentage",
#     "c_percentage",
# ]
