import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import ast


translate_options = {
    "pct_total" : "Percentage of conventionally high talent candidates admitted altogether by players.",
    "game_mode" : "Whether we're interested in the expected value of all candidates, or just the top-k.",
    "win_value_underdog" : "The probability that in the event of an admission from competing institutions, the underdog player gets the candidate.",
    "blind_combo_0" : "If the underdog can't distinguish between high and low unconventional talent candidates.",
    "blind_combo_1" : "If the overdog can't distinguish between high and low unconventional talent candidates.",
    "level_0" : "Maximum number of iterations of best response the underdog can perform.",
    "level_1" : "Maximum number of iterations of best response the overdog can perform.",
    "lognormal" : "How utility is being measured, on a normal or lognormal scale.",
    "pct_high_mean" : "The percentage of candidates that are high talent.",
    "high_low_ratio_mean" : "The ratio of the mean utility of high talent candidates to low talent candidates.",
    "high_low_ratio_variance" : "The ratio of the standard deviation of the utility of high talent candidates to low talent candidates.",
    "mean_variance_ratio" : "The ratio of the baseline mean to the baseline standard deviation.",
    "pct_high_sigma" : "The percentage of candidates that are high talent by unconventional standards.",
}

# 1. Load your main data from data.csv
df = pd.read_csv("combined.csv")

# 2. Load dimension-value options from dimension_values.csv
dim_values_df = pd.read_csv("dimension_values.csv")

# Identify all unique dimensions from the dimension_values.csv
all_dimensions = dim_values_df['dimension'].unique()
dim_values_df['values'] = dim_values_df['values'].apply(ast.literal_eval)

# Build a dictionary: dimension -> list of possible values
dimension_options = {}
for dim in all_dimensions:
    dimension_options[dim] = dim_values_df.loc[dim_values_df['dimension'] == dim, 'values'].to_list()[0]
    #print(type(dimension_options[dim]), dimension_options[dim])

# 3. Create a Dash application
app = dash.Dash(__name__)
app.title = "Dynamic Heatmap with Dimension Filters"

# 4. Layout of the app
#    - We will create:
#        A) Dropdowns to pick X-axis dimension, Y-axis dimension
#        B) For each known dimension, a dropdown to pick filter values
#        C) A graph to show the heatmap
app.layout = html.Div([
    html.H1("Dynamic Heatmap with Dimension Filters"),
    
    html.Div([
        html.Label("Select X-axis Dimension:"),
        dcc.Dropdown(
            id='x-dim-dropdown',
            options=[{'label': dim+": "+translate_options[dim], 'value': dim} for dim in all_dimensions],
            value='win_value_underdog',  # Default selection
            clearable=False
        ),
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    
    html.Div([
        html.Label("Select Y-axis Dimension:"),
        dcc.Dropdown(
            id='y-dim-dropdown',
            options=[{'label': dim+": "+translate_options[dim], 'value': dim} for dim in all_dimensions],
            value='pct_total',  # Default selection
            clearable=False
        ),
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '5%'}),
    
    html.Br(), html.Hr(),

    html.Div([
        html.H3("Filter by Dimension Values")
    ]),
    
    # Create a dropdown for each dimension to filter by valid values
    # We'll allow multi-select so that user can pick multiple values (or none).
    # If the user leaves it blank, it won't filter by that dimension.
    html.Div([
        html.Div([
            html.Label(translate_options[dim]),
            dcc.Dropdown(
                id=f'filter-{dim}',
                options=[{'label': str(val), 'value': val} for val in dimension_options[dim]],
                value=[],           # Start with no filters selected
                placeholder=f"Select {dim} values",
                multi=True
            )
        ], style={'marginBottom': 20})
        for dim in all_dimensions
    ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Br(),
    html.Button("Update Heatmap", id='update-button', n_clicks=0),

    html.Hr(),

    # Heatmap output
    dcc.Graph(id='heatmap-graph')
])

# 4.5 Callback to update filters
@app.callback(
    [Output(f'filter-{dim}', 'options') for dim in all_dimensions],  # Update options for each filter dropdown
    [Input('x-dim-dropdown', 'value'),
     Input('y-dim-dropdown', 'value')]
)

def update_filter_options(selected_x_dim, selected_y_dim):
    updated_options = []
    for dim in all_dimensions:
        # If the dimension is selected as X or Y, remove it from the filter options
        if dim == selected_x_dim or dim == selected_y_dim:
            updated_options.append([])  # No options if the dimension is selected as X or Y
        else:
            # Otherwise, retain the original options
            updated_options.append([{'label': str(val), 'value': val} for val in dimension_options[dim]])
    
    return updated_options


# 5. Callback to update heatmap
@app.callback(
    Output('heatmap-graph', 'figure'),
    Input('update-button', 'n_clicks'),
    State('x-dim-dropdown', 'value'),
    State('y-dim-dropdown', 'value'),
    *[State(f'filter-{dim}', 'value') for dim in all_dimensions]
)
def update_heatmap(n_clicks, x_dim, y_dim, *filter_values_list):
    filter_values_list = list(filter_values_list)  # Convert tuple to list

    # Copy the original dataframe
    dff = df.copy()

    # Apply dimension filters
    for dim, selected_values in zip(all_dimensions, filter_values_list):
        if selected_values:  # if not empty
            dff = dff[dff[dim].isin(selected_values)]

    # Print debug info
    print("Columns in dff after filtering:", dff.columns)
    print("First few rows of dff:", dff.head())

    # Check if x_dim and y_dim exist in DataFrame
    if dff.empty or x_dim not in dff.columns or y_dim not in dff.columns:
        print("DataFrame is empty or missing required dimensions.")
        return px.imshow([[0]], labels=dict(x="No Data", y="No Data", color="Value"))

    target_column = 'underdog_mean'

    # Group by X and Y dimensions and aggregate
    pivoted = dff.groupby([x_dim, y_dim])[target_column].mean().reset_index()

    # Pivot for heatmap
    pivot_table = pivoted.pivot(index=y_dim, columns=x_dim, values=target_column)

    # Create heatmap
    fig = px.imshow(
        pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        color_continuous_scale='Viridis',
        labels=dict(color="Mean of Value"),
        aspect="auto"
    )

    fig.update_layout(
        title=f"Heatmap: {y_dim} (rows) vs {x_dim} (columns)",
        xaxis_title=x_dim,
        yaxis_title=y_dim
    )

    return fig


# 6. Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
