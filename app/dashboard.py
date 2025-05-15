import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import requests
import shap
import base64
import io
import matplotlib.pyplot as plt
import re

MODELS = ["logistic", "randomforest", "xgboost", "catboost"]

dash_app = dash.Dash(
    __name__,
    requests_pathname_prefix="/dashboard/",
    title="ML Report"
)
server = dash_app.server

dash_app.layout = html.Div([
    html.H1("ML InsightHub Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select a Model:"),
        dcc.Dropdown(
            id="model-dropdown",
            options=[{"label": m.title(), "value": m} for m in MODELS],
            value="logistic",
            style={"width": "300px"}
        )
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    html.Div(id="accuracy-text", style={"fontWeight": "bold", "fontSize": "18px"}),

    dcc.Graph(id="metrics-graph"),
    dcc.Graph(id="report-table"),

    html.H3("Feature Importance", style={"textAlign": "center"}),
    dcc.Graph(id="feature-importance"),

    html.H3("SHAP Summary Plot", style={"textAlign": "center"}),
    html.Iframe(id="shap-plot", srcDoc="", style={"width": "100%", "height": "600px", "border": "none"})
])


def parse_classification_report(report_str):
    try:
        lines = report_str.strip().split('\n')
        lines = [line for line in lines if re.search(r'\d', line)]
        class_lines = lines[:-3]
        rows = []
        for line in class_lines:
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) == 5:
                label, prec, rec, f1, support = parts
                rows.append({
                    "Class": label,
                    "Precision": float(prec),
                    "Recall": float(rec),
                    "F1 Score": float(f1),
                    "Support": int(support)
                })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[❌] Failed to parse classification report: {e}")
        return pd.DataFrame(columns=["Class", "Precision", "Recall", "F1 Score", "Support"])

# Plot SHAP summary and return as base64 string
def generate_shap_image(shap_values, feature_names):
    try:
        shap_array = np.array(shap_values)

        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 1]

        if feature_names and len(feature_names) != shap_array.shape[1]:
            feature_names = [f"Feature {i}" for i in range(shap_array.shape[1])]

        plt.figure()
        shap.summary_plot(shap_array, feature_names=feature_names, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        base64_img = base64.b64encode(buf.read()).decode("utf-8")
        return f"<img src='data:image/png;base64,{base64_img}'/>"

    except Exception as e:
        print(f"[SHAP ERROR] {e}")
        return "<p>SHAP plot generation failed.</p>"

@dash_app.callback(
    [Output("accuracy-text", "children"),
     Output("metrics-graph", "figure"),
     Output("report-table", "figure"),
     Output("feature-importance", "figure"),
     Output("shap-plot", "srcDoc")],
    [Input("model-dropdown", "value")]
)
def update_dashboard(model_type):
    try:
        print(f"\n=== Updating dashboard for {model_type} ===")
        
        response = requests.get(f"http://localhost:5050/predict?model_type={model_type}")
        result = response.json()
        
        print(f"Keys in result: {list(result.keys())}")
        print(f"Feature names length: {len(result.get('feature_names', []))}")
        print(f"Feature importance length: {len(result.get('feature_importance', []))}")
        print(f"SHAP values type: {type(result.get('shap_values', None))}")
        
        if isinstance(result.get('shap_values', None), (list, np.ndarray)):
            print(f"SHAP values shape: {np.array(result['shap_values']).shape}")

        accuracy_text = f"Accuracy: {result.get('accuracy', 0):.4f}"
        df_metrics = parse_classification_report(result["report"])

        fig_bar = go.Figure()
        for metric in ["Precision", "Recall", "F1 Score"]:
            fig_bar.add_trace(go.Bar(x=df_metrics["Class"], y=df_metrics[metric], name=metric))
        fig_bar.update_layout(title="Class-wise Metrics", barmode="group")


        fig_table = go.Figure(data=[go.Table(
            header=dict(values=list(df_metrics.columns), fill_color='lightgray', align='center'),
            cells=dict(values=[df_metrics[col] for col in df_metrics.columns],
                     fill_color='lavender', align='center'))
        ])
        fig_table.update_layout(title="Classification Report")

        importance = result.get("feature_importance", [])
        feature_names = result.get("feature_names", [])
        fi_figure = go.Figure()

        if importance and feature_names:
            try:
                fi_df = pd.DataFrame({
                    'features': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                fi_figure.add_trace(go.Bar(
                    x=fi_df['importance'][:20],
                    y=fi_df['features'][:20],
                    orientation='h',
                    marker_color='indianred'
                ))
                fi_figure.update_layout(
                    title="Top 20 Feature Importance",
                    margin=dict(l=150)
                )
            except Exception as e:
                print(f"⚠️ Feature importance error: {e}")
                fi_figure.add_annotation(
                    text="Could not display feature importance",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
        else:
            print(f"⚠️ Missing data - Feature names: {len(feature_names)}, Importance: {len(importance)}")
            fi_figure.add_annotation(
                text="No feature importance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        try:
            shap_html = generate_shap_image(result["shap_values"], result["feature_names"])
        except Exception as e:
            print(f"[SHAP ERROR] Failed to generate SHAP plot: {e}")
            shap_html = "<p>SHAP plot generation failed.</p>"

        return accuracy_text, fig_bar, fig_table, fi_figure, shap_html

    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        empty_fig = go.Figure()
        return f"Error: {e}", empty_fig, empty_fig, empty_fig, ""
