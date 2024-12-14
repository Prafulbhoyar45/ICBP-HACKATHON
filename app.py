import os
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load pre-trained models
cnc_models = {
    "CNC01": pickle.load(open("CNC01_model.pkl", "rb")),
    "CNC02": pickle.load(open("CNC02_model.pkl", "rb")),
    "CNC03": pickle.load(open("CNC03_model.pkl", "rb")),
    "CNC04": pickle.load(open("CNC04_model.pkl", "rb")),
    "CNC05": pickle.load(open("CNC05_model.pkl", "rb")),
    "CNC06": pickle.load(open("CNC06_model.pkl", "rb")),
    "CNC07": pickle.load(open("CNC07_model.pkl", "rb")),
}

# Images for each machining process
process_images = {
    "Transmission Machining": "transmission_image1.jpg",
    "Engine Machining": "engine_image.jpg",
    "Cylinder Head Machining": "cylinder_head_image.jpg",
}

# Threshold values for the graphs
THRESHOLDS = {
    "Control Panel Temperature (\u00b0C)": 50,
    "Spindle Motor Temperature (\u00b0C)": 80,
    "Servo Motor Temperature (\u00b0C)": 65,
}

# Apply custom CSS for button styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #E9F1FA;
        font-family: 'Roboto', sans-serif;
    }}
    .stTitle {{
        font-size: 40px;
        color: #00ABE4;
        font-family: 'Montserrat', sans-serif;
        text-align: center;
    }}
    .stHeader {{
        font-size: 24px;
        color: #00ABE4;
        font-family: 'Roboto', sans-serif;
        text-align: center;
        margin-bottom: 20px;
    }}
    .stButton > button {{
        width: 100%;
        height: 60px;
        font-size: 16px;
        font-family: 'Roboto', sans-serif;
        background-color: #00ABE4;
        color: white;
        border-radius: 10px;
        margin: 10px;
    }}
    .stButton > button:hover {{
        background-color: #008FC4;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    # Sidebar setup
    st.sidebar.image("mentor_cir.png", caption="Usha Mary - Mentor", width=170)
    st.sidebar.image("mario_cir.png", caption="Mario Thokchom", width=170)
    st.sidebar.image("my_image_cir.png", caption="Praful Bhoyar", width=170)

    # Header layout with logos
    header = st.columns([1, 4, 1])
    with header[0]:
        st.image("organization_logo.png", width=100)  # Left logo
    with header[1]:
        st.markdown("<div class='stTitle'>AI Maintenance Tracker</div>", unsafe_allow_html=True)
    with header[2]:
        st.image("Edunet-Foundation-logo.png", use_container_width=True)  # Right logo

    # Title and department image
    st.markdown("<div class='stHeader'><b>Transmission & Machining Department Overview</b></div>", unsafe_allow_html=True)
    st.image("department.jpg", use_container_width=True)

    # Machining buttons
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])  # Equal spacing for columns
    with button_col1:
        if st.button("Transmission Machining"):
            st.session_state["selected_process"] = "Transmission Machining"
    with button_col2:
        if st.button("Engine Machining"):
            st.session_state["selected_process"] = "Engine Machining"
    with button_col3:
        if st.button("Cylinder Head Machining"):
            st.session_state["selected_process"] = "Cylinder Head Machining"

    # Display selected process output below all buttons
    if "selected_process" in st.session_state:
        selected_process = st.session_state["selected_process"]
        st.markdown(f"<div class='stHeader'><b>{selected_process}</b></div>", unsafe_allow_html=True)
        st.image(process_images[selected_process], use_container_width=True)

        # Select a machine
        machining_processes = {
            "Transmission Machining": ["None", "CNC01", "CNC02", "CNC03"],
            "Engine Machining": ["None", "CNC04", "CNC05", "CNC06"],
            "Cylinder Head Machining": ["None", "CNC07"],
        }

        selected_machine = st.selectbox(
            "Select a machine to Monitor", options=machining_processes[selected_process], index=0
        )

        if selected_machine != "None":
            cnc_machine_page(selected_machine)


def cnc_machine_page(machine):
    st.markdown(f"<div class='stHeader'><b>{machine} Monitoring</b></div>", unsafe_allow_html=True)

    # Load the dataset for the selected machine
    dataset_path = f"df_{machine}.xlsx"
    if os.path.exists(dataset_path):
        data = pd.read_excel(dataset_path)

        # Check if required columns exist in the dataset
        required_columns = ["Timestamp", "Control Panel Temperature (\u00b0C)", "Spindle Motor Temperature (\u00b0C)", "Servo Motor Temperature (\u00b0C)"]
        if all(col in data.columns for col in required_columns):
            # Plot the graphs with thresholds
            plot_graph_with_threshold(data, "Control Panel Temperature (\u00b0C)", "Timestamp")
            plot_graph_with_threshold(data, "Spindle Motor Temperature (\u00b0C)", "Timestamp")
            plot_graph_with_threshold(data, "Servo Motor Temperature (\u00b0C)", "Timestamp")
        else:
            st.error("Dataset does not contain required columns for plotting.")
    else:
        st.error(f"Dataset for {machine} not found.")

    # Proceed with the custom parameter testing form
    predict_maintenance_form(machine)

def plot_graph_with_threshold(data, y_col, x_col):
    # Update the graph title to green
    st.markdown(
        f"""
        <div style="color: green; font-size: 24px; font-weight: bold; text-align: center;">
            {y_col} vs {x_col}
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig, ax = plt.subplots()
    ax.plot(data[x_col], data[y_col], label=y_col, color="green")
    ax.axhline(
        y=THRESHOLDS[y_col],
        color="red",
        linestyle="--",
        label=f"Threshold ({THRESHOLDS[y_col]} °C)",
    )
    ax.set_xlabel(x_col, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_col, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)


def predict_maintenance_form(machine):
    st.markdown(f"<div class='stHeader'>Custom Parameter Testing for {machine}</div>", unsafe_allow_html=True)

    # Initialize session state for prediction result
    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = ""

    with st.form("input_form"):
        # Define input fields
        inputs = {}

        # Define the feature names and divide them into rows
        feature_names = [
            "Control Panel Temperature (\u00b0C)",
            "Spindle Motor Temperature (\u00b0C)",
            "Servo Motor Temperature (\u00b0C)",
            "Coolant Temperature (\u00b0C)",
            "Coolant Flow (L/min)",
            "Coolant Level (%)",
            "Tool Wear (%)",
            "Tool Breakage (Yes/No)",
            "Spindle Speed (RPM)",
            "Feed Rate (mm/min)",
            "Vibration (mm/s)",
            "Fan Speed (RPM)",
            "Power Consumption (kW)",
            "Cycle Time (mins)",
            "Idle Time (mins)",
            "Axis Load (X, Y, Z)",
            "Ambient Temperature (\u00b0C)",
            "Hydraulic Pressure (bar)",
            "Status (Running/Stopped)",
        ]

        # Arrange inputs in a grid format
        cols = st.columns(3)  # 3 columns for grid layout
        for i, feature in enumerate(feature_names):
            col = cols[i % 3]  # Distribute inputs across columns
            if feature in ["Tool Breakage (Yes/No)", "Status (Running/Stopped)"]:
                options = ["No", "Yes"] if "Tool Breakage" in feature else ["Stopped", "Running"]
                inputs[feature] = col.selectbox(feature, options, index=0)
            else:
                inputs[feature] = col.text_input(feature, value="")  # Blank numerical inputs

        # Add a submit button
        submit_button = st.form_submit_button("Enter to submit")

        if submit_button:
            # Check if all fields are filled
            if all(value != "" for value in inputs.values()):
                # Prepare input DataFrame
                input_df = pd.DataFrame([{
                    k: (1 if v.lower() == "yes" or v.lower() == "running" else 0) if isinstance(v, str) else float(v)
                    for k, v in inputs.items()
                }])

                # Align input DataFrame with model's expected features
                model = cnc_models[machine]
                expected_features = model.feature_names_in_

                # Add missing columns with default values (0)
                for feature in expected_features:
                    if feature not in input_df.columns:
                        input_df[feature] = 0

                # Reorder columns to match the model
                input_df = input_df[expected_features]

                # Make the prediction
                prediction = model.predict(input_df)[0]

                # Update session state with the prediction result
                st.session_state["prediction_result"] = f"{machine} Requires Maintenance" if prediction == 1 else f"{machine} Requires no Maintenance"
            else:
                st.error("All inputs are mandatory. Please fill in all fields.")

    # Display the prediction result below the form
    if st.session_state["prediction_result"]:
        is_maintenance_required = "Requires Maintenance" in st.session_state["prediction_result"]
        result_color = "#FF3131" if is_maintenance_required else "#50C878"

        st.markdown(
            f"""
            <div style="background-color: {result_color}; 
                        color: white; 
                        border-radius: 10px; 
                        padding: 15px; 
                        text-align: center; 
                        font-size: 18px; 
                        font-family: 'Roboto', sans-serif;
                        font-weight: bold;">
                {st.session_state['prediction_result']}
            </div>
            """,
            unsafe_allow_html=True
        )




if __name__ == "__main__":
    main()

               
