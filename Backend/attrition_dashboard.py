
"""
Employee Attrition Dashboard
===========================
Interactive Streamlit dashboard for Employee Attrition Prediction with Action Simulation

Run with: streamlit run attrition_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from employee_attrition_predictor import EmployeeAttritionPredictor

# Set page configuration
st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 0.5rem 0;
}

.high-risk {
    border-left-color: #ff4444 !important;
}

.medium-risk {
    border-left-color: #ffaa00 !important;
}

.low-risk {
    border-left-color: #44ff44 !important;
}

.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """
    Create sample data for demonstration purposes
    In production, this would load real HR data
    """
    np.random.seed(42)
    n_employees = 500

    # Create sample employee data
    data = {
        'EmployeeID': range(1, n_employees + 1),
        'Age': np.random.normal(35, 8, n_employees).astype(int),
        'MonthlyIncome': np.random.normal(5000, 1500, n_employees).astype(int),
        'YearsAtCompany': np.random.exponential(3, n_employees).astype(int),
        'WorkLifeBalance': np.random.choice([1, 2, 3, 4], n_employees, p=[0.1, 0.2, 0.4, 0.3]),
        'JobSatisfaction': np.random.choice([1, 2, 3, 4], n_employees, p=[0.15, 0.25, 0.35, 0.25]),
        'Department': np.random.choice(['Sales', 'R&D', 'HR', 'IT', 'Marketing'], n_employees),
        'OverTime': np.random.choice(['Yes', 'No'], n_employees, p=[0.3, 0.7]),
        'DistanceFromHome': np.random.uniform(1, 30, n_employees).astype(int),
        'JobLevel': np.random.choice([1, 2, 3, 4, 5], n_employees, p=[0.3, 0.3, 0.2, 0.15, 0.05])
    }

    df = pd.DataFrame(data)

    # Create attrition probability based on various factors (realistic simulation)
    attrition_prob = (
        0.1 +  # Base probability
        (df['WorkLifeBalance'] == 1) * 0.3 +  # Poor work-life balance
        (df['JobSatisfaction'] == 1) * 0.25 +  # Poor job satisfaction
        (df['OverTime'] == 'Yes') * 0.2 +  # Overtime
        (df['YearsAtCompany'] < 2) * 0.15 +  # New employees
        (df['Age'] < 25) * 0.1 +  # Young employees
        np.random.normal(0, 0.1, n_employees)  # Random variation
    )

    # Ensure probabilities are between 0 and 1
    attrition_prob = np.clip(attrition_prob, 0, 1)
    df['AttritionProbability'] = attrition_prob

    # Create risk categories
    df['RiskCategory'] = pd.cut(
        attrition_prob,
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    # Simulate actual attrition based on probability
    df['ActualAttrition'] = np.random.binomial(1, attrition_prob, n_employees)

    return df

def create_overview_metrics(df):
    """Create overview metrics cards"""
    total_employees = len(df)
    high_risk = len(df[df['RiskCategory'] == 'High Risk'])
    medium_risk = len(df[df['RiskCategory'] == 'Medium Risk'])
    avg_risk = df['AttritionProbability'].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Employees",
            value=f"{total_employees:,}",
            delta=f"{((total_employees - 480) / 480 * 100):+.1f}%" if total_employees != 480 else None
        )

    with col2:
        st.metric(
            label="High Risk Employees",
            value=f"{high_risk}",
            delta=f"{(high_risk / total_employees * 100):.1f}% of workforce"
        )

    with col3:
        st.metric(
            label="Medium Risk Employees",
            value=f"{medium_risk}",
            delta=f"{(medium_risk / total_employees * 100):.1f}% of workforce"
        )

    with col4:
        st.metric(
            label="Average Risk Score",
            value=f"{avg_risk:.1%}",
            delta=f"{'High' if avg_risk > 0.5 else 'Moderate' if avg_risk > 0.3 else 'Low'} concern level"
        )

def create_risk_distribution_chart(df):
    """Create risk distribution visualizations"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Risk Category Distribution', 'Risk Score Distribution'),
        specs=[[{'type': 'pie'}, {'type': 'histogram'}]]
    )

    # Pie chart for risk categories
    risk_counts = df['RiskCategory'].value_counts()
    colors = ['#44ff44', '#ffaa00', '#ff4444']

    fig.add_trace(
        go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker_colors=colors,
            hole=0.4,
            textinfo='label+percent'
        ),
        row=1, col=1
    )

    # Histogram for risk scores
    fig.add_trace(
        go.Histogram(
            x=df['AttritionProbability'],
            nbinsx=20,
            marker_color='skyblue',
            opacity=0.7
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Employee Risk Analysis"
    )

    return fig

def simulate_interventions():
    """Simulate different intervention scenarios"""
    st.subheader("🎯 Intervention Simulation")

    # Define intervention scenarios
    interventions = {
        'Salary Increase (10%)': {
            'cost_per_employee': 6000,
            'effectiveness': 0.4,
            'description': 'Increase salary by 10% for at-risk employees'
        },
        'Work-Life Balance Program': {
            'cost_per_employee': 2000,
            'effectiveness': 0.3,
            'description': 'Implement flexible working arrangements and wellness programs'
        },
        'Professional Development': {
            'cost_per_employee': 3000,
            'effectiveness': 0.35,
            'description': 'Provide training and career advancement opportunities'
        },
        'Manager Training Program': {
            'cost_per_employee': 1500,
            'effectiveness': 0.25,
            'description': 'Train managers to better support and engage their teams'
        },
        'Comprehensive Package': {
            'cost_per_employee': 8000,
            'effectiveness': 0.6,
            'description': 'Combined approach with multiple interventions'
        }
    }

    # User input for simulation parameters
    col1, col2 = st.columns(2)

    with col1:
        selected_intervention = st.selectbox(
            "Select Intervention Type",
            list(interventions.keys())
        )

        high_risk_employees = st.number_input(
            "Number of High-Risk Employees",
            min_value=1,
            max_value=200,
            value=50,
            help="Number of employees identified as high-risk for attrition"
        )

    with col2:
        cost_per_replacement = st.number_input(
            "Cost per Employee Replacement ($)",
            min_value=10000,
            max_value=100000,
            value=50000,
            help="Average cost to replace an employee (recruiting, training, lost productivity)"
        )

        intervention_budget = st.number_input(
            "Available Budget ($)",
            min_value=1000,
            max_value=1000000,
            value=300000,
            help="Total budget available for retention interventions"
        )

    # Calculate intervention impact
    if st.button("Run Simulation", type="primary"):
        intervention = interventions[selected_intervention]

        total_intervention_cost = high_risk_employees * intervention['cost_per_employee']
        employees_retained = int(high_risk_employees * intervention['effectiveness'])
        cost_savings = employees_retained * cost_per_replacement
        net_benefit = cost_savings - total_intervention_cost
        roi = (net_benefit / total_intervention_cost) * 100 if total_intervention_cost > 0 else 0

        # Display results
        st.subheader("📊 Simulation Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Employees Retained",
                f"{employees_retained}",
                f"{(employees_retained/high_risk_employees*100):.1f}% success rate"
            )

        with col2:
            st.metric(
                "Total Cost Savings",
                f"${cost_savings:,}",
                f"${cost_savings - total_intervention_cost:,} net benefit"
            )

        with col3:
            st.metric(
                "ROI",
                f"{roi:.1f}%",
                "return on investment"
            )

        # Budget analysis
        if total_intervention_cost > intervention_budget:
            st.error(f"⚠️ Intervention cost (${total_intervention_cost:,}) exceeds available budget (${intervention_budget:,})")
            st.info(f"Consider reducing scope to {int(intervention_budget / intervention['cost_per_employee'])} employees or increasing budget.")
        else:
            st.success(f"✅ Intervention is within budget. Remaining budget: ${intervention_budget - total_intervention_cost:,}")

        # Create ROI comparison chart
        scenarios = list(interventions.keys())
        roi_values = []
        cost_values = []
        retention_values = []

        for scenario in scenarios:
            scenario_data = interventions[scenario]
            scenario_cost = high_risk_employees * scenario_data['cost_per_employee']
            scenario_retained = int(high_risk_employees * scenario_data['effectiveness'])
            scenario_savings = scenario_retained * cost_per_replacement
            scenario_roi = ((scenario_savings - scenario_cost) / scenario_cost) * 100 if scenario_cost > 0 else 0

            roi_values.append(scenario_roi)
            cost_values.append(scenario_cost)
            retention_values.append(scenario_retained)

        # ROI comparison chart
        fig_roi = go.Figure(data=[
            go.Bar(
                x=scenarios,
                y=roi_values,
                marker_color=['#ff4444' if x == selected_intervention else '#1f77b4' for x in scenarios],
                text=[f"{x:.1f}%" for x in roi_values],
                textposition='outside'
            )
        ])

        fig_roi.update_layout(
            title="ROI Comparison Across Interventions",
            xaxis_title="Intervention Type",
            yaxis_title="ROI (%)",
            height=400
        )

        st.plotly_chart(fig_roi, use_container_width=True)

def main():
    """Main dashboard application"""
    st.title("👥 Employee Attrition Analytics Dashboard")
    st.markdown("*Predict attrition risk and simulate retention interventions*")

    # Load sample data
    df = load_sample_data()

    # Sidebar for filters
    st.sidebar.header("🔍 Filters")

    departments = st.sidebar.multiselect(
        "Select Departments",
        options=df['Department'].unique(),
        default=df['Department'].unique()
    )

    risk_categories = st.sidebar.multiselect(
        "Risk Categories",
        options=df['RiskCategory'].unique(),
        default=df['RiskCategory'].unique()
    )

    age_range = st.sidebar.slider(
        "Age Range",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )

    # Filter data
    filtered_df = df[
        (df['Department'].isin(departments)) &
        (df['RiskCategory'].isin(risk_categories)) &
        (df['Age'].between(age_range[0], age_range[1]))
    ]

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "🎯 Risk Analysis", 
        "💡 Interventions", 
        "📈 Department Analysis"
    ])

    with tab1:
        st.header("Employee Attrition Overview")
        create_overview_metrics(filtered_df)

        st.plotly_chart(
            create_risk_distribution_chart(filtered_df),
            use_container_width=True
        )

        # Recent trends (simulated)
        st.subheader("📅 Attrition Risk Trends")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        risk_trend = [0.32, 0.35, 0.38, 0.36, 0.33, 0.31]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=months,
            y=risk_trend,
            mode='lines+markers',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))

        fig_trend.update_layout(
            title="Average Attrition Risk Over Time",
            xaxis_title="Month",
            yaxis_title="Average Risk Score",
            height=300
        )

        st.plotly_chart(fig_trend, use_container_width=True)

    with tab2:
        st.header("Risk Analysis Deep Dive")

        # High-risk employee table
        high_risk_df = filtered_df[filtered_df['RiskCategory'] == 'High Risk'].sort_values(
            'AttritionProbability', ascending=False
        )

        st.subheader(f"🚨 High-Risk Employees ({len(high_risk_df)})")

        if len(high_risk_df) > 0:
            # Display top 10 highest risk employees
            display_cols = ['EmployeeID', 'Age', 'Department', 'YearsAtCompany', 
                          'WorkLifeBalance', 'JobSatisfaction', 'AttritionProbability']
            st.dataframe(
                high_risk_df[display_cols].head(10),
                use_container_width=True,
                column_config={
                    "AttritionProbability": st.column_config.ProgressColumn(
                        "Risk Score",
                        help="Probability of employee leaving",
                        min_value=0,
                        max_value=1,
                        format="%.2f"
                    )
                }
            )

            # Risk factors analysis
            st.subheader("🔍 Risk Factor Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Work-life balance impact
                wlb_risk = filtered_df.groupby('WorkLifeBalance')['AttritionProbability'].mean()
                fig_wlb = px.bar(
                    x=wlb_risk.index,
                    y=wlb_risk.values,
                    title="Risk by Work-Life Balance Score",
                    labels={'x': 'Work-Life Balance (1=Poor, 4=Excellent)', 'y': 'Avg Risk Score'}
                )
                st.plotly_chart(fig_wlb, use_container_width=True)

            with col2:
                # Job satisfaction impact
                js_risk = filtered_df.groupby('JobSatisfaction')['AttritionProbability'].mean()
                fig_js = px.bar(
                    x=js_risk.index,
                    y=js_risk.values,
                    title="Risk by Job Satisfaction Score",
                    labels={'x': 'Job Satisfaction (1=Poor, 4=Excellent)', 'y': 'Avg Risk Score'}
                )
                st.plotly_chart(fig_js, use_container_width=True)
        else:
            st.info("No high-risk employees found with current filters.")

    with tab3:
        simulate_interventions()

    with tab4:
        st.header("Department Analysis")

        # Department-wise risk analysis
        dept_analysis = filtered_df.groupby('Department').agg({
            'AttritionProbability': ['mean', 'count'],
            'MonthlyIncome': 'mean'
        }).round(3)

        dept_analysis.columns = ['Avg Risk Score', 'Employee Count', 'Avg Salary']
        dept_analysis = dept_analysis.reset_index()

        st.dataframe(dept_analysis, use_container_width=True)

        # Department risk visualization
        fig_dept = px.scatter(
            dept_analysis,
            x='Employee Count',
            y='Avg Risk Score',
            size='Avg Salary',
            color='Department',
            title="Department Risk vs Size Analysis",
            hover_data=['Avg Salary']
        )

        st.plotly_chart(fig_dept, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Dashboard Features**: Real-time risk prediction • Intervention simulation • ROI analysis • Department insights"
    )

if __name__ == "__main__":
    main()
