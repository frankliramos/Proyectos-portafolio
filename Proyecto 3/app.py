"""
Customer Churn Prediction Dashboard
Interactive Streamlit application for real-time churn risk assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def classify_risk(probability):
    """Classify churn probability into risk levels"""
    if probability >= 0.6:
        return "🔴 High", "risk-high"
    elif probability >= 0.3:
        return "🟡 Medium", "risk-medium"
    else:
        return "🟢 Low", "risk-low"


def get_retention_strategy(risk_level, probability):
    """Get personalized retention strategy based on risk level"""
    if "High" in risk_level:
        return {
            "priority": "URGENT",
            "actions": [
                "Immediate call from retention team",
                "Personalized retention offer (fee waiver + bonus)",
                "Executive-level outreach",
                "Premium customer service upgrade"
            ],
            "expected_roi": "3.5x",
            "budget": "$200-500 per customer"
        }
    elif "Medium" in risk_level:
        return {
            "priority": "HIGH",
            "actions": [
                "Personalized email campaign",
                "Special offer on additional products",
                "Account manager check-in call",
                "Loyalty rewards activation"
            ],
            "expected_roi": "2.8x",
            "budget": "$100-200 per customer"
        }
    else:
        return {
            "priority": "STANDARD",
            "actions": [
                "Standard customer service",
                "Quarterly satisfaction survey",
                "Loyalty program enrollment",
                "Product usage tips newsletter"
            ],
            "expected_roi": "1.5x",
            "budget": "$20-50 per customer"
        }


def create_sample_data():
    """Create sample customer data for demonstration"""
    np.random.seed(42)
    n_customers = 100
    
    data = {
        'customer_id': range(1, n_customers + 1),
        'credit_score': np.random.randint(300, 850, n_customers),
        'age': np.random.randint(18, 70, n_customers),
        'tenure': np.random.randint(0, 10, n_customers),
        'balance': np.random.uniform(0, 250000, n_customers),
        'num_of_products': np.random.randint(1, 5, n_customers),
        'has_cr_card': np.random.choice([0, 1], n_customers),
        'is_active_member': np.random.choice([0, 1], n_customers, p=[0.3, 0.7]),
        'estimated_salary': np.random.uniform(20000, 200000, n_customers),
        'geography': np.random.choice(['France', 'Spain', 'Germany'], n_customers),
        'gender': np.random.choice(['Male', 'Female'], n_customers)
    }
    
    df = pd.DataFrame(data)
    
    # Generate synthetic churn probabilities based on features
    # Higher probability for: older age, low products, inactive, new tenure
    churn_prob = (
        (df['age'] - 18) / 52 * 0.2 +  # Age factor
        (1 - df['is_active_member']) * 0.3 +  # Activity factor
        (4 - df['num_of_products']) / 3 * 0.25 +  # Products factor
        (10 - df['tenure']) / 10 * 0.15 +  # Tenure factor
        np.random.uniform(0, 0.1, n_customers)  # Random noise
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    df['churn_probability'] = churn_prob
    
    return df


# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">💳 Customer Churn Prediction System</h1>', 
                unsafe_allow_html=True)
    st.markdown("**Banking Retention Intelligence Dashboard**")
    st.markdown("---")
    
    # Load or create sample data
    try:
        # Try to load real data if available
        df = pd.read_csv('data/customer_predictions.csv')
    except:
        # Use sample data for demonstration
        df = create_sample_data()
        st.info("📊 Displaying sample data for demonstration. Load your customer data to see real predictions.")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    
    # Page selection
    page = st.sidebar.radio(
        "Select View",
        ["🏠 Overview", "👤 Individual Prediction", "📊 Customer Segmentation", "📈 Analytics"]
    )
    
    # Risk threshold controls
    st.sidebar.subheader("Risk Thresholds")
    high_threshold = st.sidebar.slider("High Risk Threshold", 0.0, 1.0, 0.6, 0.05)
    medium_threshold = st.sidebar.slider("Medium Risk Threshold", 0.0, 1.0, 0.3, 0.05)
    
    # Filter options
    st.sidebar.subheader("Filters")
    selected_geography = st.sidebar.multiselect(
        "Geography",
        options=df['geography'].unique(),
        default=df['geography'].unique()
    )
    
    age_range = st.sidebar.slider(
        "Age Range",
        int(df['age'].min()),
        int(df['age'].max()),
        (int(df['age'].min()), int(df['age'].max()))
    )
    
    # Apply filters
    df_filtered = df[
        (df['geography'].isin(selected_geography)) &
        (df['age'] >= age_range[0]) &
        (df['age'] <= age_range[1])
    ]
    
    # Classify risk for all customers
    df_filtered['risk_level'] = df_filtered['churn_probability'].apply(
        lambda x: classify_risk(x)[0]
    )
    
    # Page routing
    if page == "🏠 Overview":
        show_overview(df_filtered, high_threshold, medium_threshold)
    elif page == "👤 Individual Prediction":
        show_individual_prediction(df_filtered)
    elif page == "📊 Customer Segmentation":
        show_segmentation(df_filtered)
    else:
        show_analytics(df_filtered)


def show_overview(df, high_threshold, medium_threshold):
    """Display overview dashboard"""
    st.header("📊 Portfolio Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    high_risk = len(df[df['churn_probability'] >= high_threshold])
    medium_risk = len(df[(df['churn_probability'] >= medium_threshold) & 
                         (df['churn_probability'] < high_threshold)])
    low_risk = len(df[df['churn_probability'] < medium_threshold])
    avg_churn_prob = df['churn_probability'].mean()
    
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    with col2:
        st.metric("🔴 High Risk", f"{high_risk:,}", f"{high_risk/total_customers*100:.1f}%")
    with col3:
        st.metric("🟡 Medium Risk", f"{medium_risk:,}", f"{medium_risk/total_customers*100:.1f}%")
    with col4:
        st.metric("🟢 Low Risk", f"{low_risk:,}", f"{low_risk/total_customers*100:.1f}%")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn probability distribution
        fig = px.histogram(
            df,
            x='churn_probability',
            nbins=30,
            title='Churn Probability Distribution',
            labels={'churn_probability': 'Churn Probability'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk level pie chart
        risk_counts = df['risk_level'].value_counts()
        colors = {'🔴 High': '#d32f2f', '🟡 Medium': '#f57c00', '🟢 Low': '#388e3c'}
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Customer Risk Distribution',
            color=risk_counts.index,
            color_discrete_map=colors
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top high-risk customers
    st.subheader("🚨 Top 10 High-Risk Customers")
    high_risk_df = df.nlargest(10, 'churn_probability')[
        ['customer_id', 'age', 'tenure', 'balance', 'num_of_products', 
         'is_active_member', 'churn_probability', 'risk_level']
    ]
    
    # Format the dataframe
    display_df = high_risk_df.copy()
    display_df['churn_probability'] = display_df['churn_probability'].apply(lambda x: f"{x*100:.1f}%")
    display_df['balance'] = display_df['balance'].apply(lambda x: f"${x:,.0f}")
    display_df['is_active_member'] = display_df['is_active_member'].map({0: '❌', 1: '✅'})
    
    st.dataframe(display_df, use_container_width=True)


def show_individual_prediction(df):
    """Display individual customer prediction interface"""
    st.header("👤 Individual Customer Assessment")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Customer Input")
        
        # Input method selection
        input_method = st.radio("Input Method", ["Select Existing Customer", "Enter New Customer"])
        
        if input_method == "Select Existing Customer":
            customer_id = st.selectbox("Customer ID", df['customer_id'].tolist())
            customer_data = df[df['customer_id'] == customer_id].iloc[0]
            churn_prob = customer_data['churn_probability']
        else:
            st.write("Enter customer details:")
            credit_score = st.slider("Credit Score", 300, 850, 650)
            age = st.slider("Age", 18, 70, 35)
            tenure = st.slider("Tenure (years)", 0, 10, 5)
            balance = st.number_input("Balance ($)", 0, 250000, 100000, step=1000)
            num_products = st.slider("Number of Products", 1, 4, 2)
            has_card = st.checkbox("Has Credit Card", value=True)
            is_active = st.checkbox("Active Member", value=True)
            salary = st.number_input("Estimated Salary ($)", 20000, 200000, 75000, step=5000)
            geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            
            # Simple heuristic prediction (replace with actual model in production)
            churn_prob = (
                (age - 18) / 52 * 0.2 +
                (1 - int(is_active)) * 0.3 +
                (4 - num_products) / 3 * 0.25 +
                (10 - tenure) / 10 * 0.15 +
                0.05
            )
            churn_prob = np.clip(churn_prob, 0, 1)
    
    with col2:
        st.subheader("Prediction Results")
        
        # Display prediction
        risk_label, risk_class = classify_risk(churn_prob)
        
        # Large probability display
        st.markdown(f"### Churn Probability: {churn_prob*100:.1f}%")
        st.progress(churn_prob)
        st.markdown(f'<p class="{risk_class}">Risk Level: {risk_label}</p>', 
                   unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Retention strategy
        st.subheader("📋 Recommended Retention Strategy")
        strategy = get_retention_strategy(risk_label, churn_prob)
        
        st.markdown(f"**Priority:** {strategy['priority']}")
        st.markdown(f"**Expected ROI:** {strategy['expected_roi']}")
        st.markdown(f"**Budget:** {strategy['budget']}")
        
        st.markdown("**Recommended Actions:**")
        for action in strategy['actions']:
            st.markdown(f"- {action}")


def show_segmentation(df):
    """Display customer segmentation analysis"""
    st.header("📊 Customer Segmentation")
    
    # Segment by risk and characteristics
    col1, col2 = st.columns(2)
    
    with col1:
        # Age vs Churn Probability
        fig = px.scatter(
            df,
            x='age',
            y='churn_probability',
            color='risk_level',
            size='balance',
            hover_data=['customer_id', 'tenure', 'num_of_products'],
            title='Age vs Churn Probability',
            color_discrete_map={'🔴 High': '#d32f2f', '🟡 Medium': '#f57c00', '🟢 Low': '#388e3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tenure vs Churn Probability
        fig = px.scatter(
            df,
            x='tenure',
            y='churn_probability',
            color='risk_level',
            size='balance',
            hover_data=['customer_id', 'age', 'num_of_products'],
            title='Tenure vs Churn Probability',
            color_discrete_map={'🔴 High': '#d32f2f', '🟡 Medium': '#f57c00', '🟢 Low': '#388e3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Segmentation table
    st.subheader("Customer Segments Summary")
    
    segment_summary = df.groupby('risk_level').agg({
        'customer_id': 'count',
        'age': 'mean',
        'tenure': 'mean',
        'balance': 'mean',
        'num_of_products': 'mean',
        'churn_probability': 'mean'
    }).round(2)
    
    segment_summary.columns = ['Count', 'Avg Age', 'Avg Tenure', 'Avg Balance', 
                                'Avg Products', 'Avg Churn Prob']
    
    st.dataframe(segment_summary, use_container_width=True)


def show_analytics(df):
    """Display detailed analytics"""
    st.header("📈 Advanced Analytics")
    
    # Feature importance (mock data - replace with SHAP in production)
    st.subheader("🎯 Feature Importance")
    
    features = ['Age', 'Active Member', 'Num Products', 'Geography', 'Balance', 
                'Tenure', 'Gender', 'Credit Card']
    importance = [0.22, 0.19, 0.16, 0.14, 0.11, 0.09, 0.05, 0.04]
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title='Feature Importance for Churn Prediction',
        labels={'x': 'Importance Score', 'y': 'Feature'},
        color=importance,
        color_continuous_scale='Blues'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("📊 Feature Correlations")
    
    numeric_cols = ['credit_score', 'age', 'tenure', 'balance', 'num_of_products',
                    'has_cr_card', 'is_active_member', 'estimated_salary', 'churn_probability']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu',
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Geographic analysis
    st.subheader("🌍 Geographic Analysis")
    
    geo_analysis = df.groupby('geography').agg({
        'customer_id': 'count',
        'churn_probability': 'mean'
    }).reset_index()
    geo_analysis.columns = ['Geography', 'Customer Count', 'Avg Churn Probability']
    
    fig = px.bar(
        geo_analysis,
        x='Geography',
        y=['Customer Count', 'Avg Churn Probability'],
        title='Churn Analysis by Geography',
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
