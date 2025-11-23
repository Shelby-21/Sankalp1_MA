import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dynamic Pricing Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
# NOTE: Keeping this CSS block as it was present in your request.
st.markdown("""
<style>
    /* Main container styling */
    .main { background-color: #f8fafc; }
    
    /* Metric Cards */
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 5px solid #3b82f6;
    }
    .metric-label { font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 2rem; color: #1e293b; font-weight: 700; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; }
    .positive { color: #16a34a; }
    
    /* Insight Box */
    .insight-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .insight-header { font-size: 1.1rem; font-weight: bold; color: #1e293b; margin-bottom: 15px; display: flex; align-items: center; }
    .recommendation {
        background: #f8fafc;
        border-left: 4px solid #6366f1;
        padding: 12px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
    }
    
    /* Pricing Cards */
    .pricing-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
    .price-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s;
    }
    .price-card:hover { transform: translateY(-3px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .price-title { font-size: 0.8rem; color: #64748b; font-weight: bold; text-transform: uppercase; height: 40px; display: flex; align-items: center; justify-content: center; }
    .price-tag { font-size: 1.2rem; font-weight: bold; color: #0f172a; margin: 5px 0; }
    .bundle-highlight {
        background: linear-gradient(135deg, #4f46e5, #3b82f6);
        color: white !important;
        border: none;
    }
    .bundle-highlight .price-title, .bundle-highlight .price-tag { color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING (Changed to read fixed file) ---
@st.cache_data
def load_data():
    """Loads the WTP data directly from the specified backend file."""
    # Hardcoded filename based on user's requirement
    FILE_NAME = "Samsung_Sankalp.csv"
    try:
        df = pd.read_csv(FILE_NAME)
        st.success(f"Successfully loaded data from {FILE_NAME}. Optimization running automatically.")
        return df
    except FileNotFoundError:
        st.error(f"FATAL ERROR: The data file '{FILE_NAME}' was not found in the application directory.")
        st.stop() # Stop the script execution
        
# --- 2. OPTIMIZATION ENGINE (No Change) ---

def calculate_baseline(df, products):
    """Calculates revenue if we only use separate pricing (no bundle)."""
    total_rev = 0
    for prod in products:
        wtp = df[prod].values
        candidates = np.unique(wtp)
        best_r = 0
        for p in candidates:
            r = p * np.sum(wtp >= p)
            if r > best_r: best_r = r
        total_rev += best_r
    return total_rev

@st.cache_data(show_spinner=False)
def solve_pricing(df, products):
    """
    Simulates Excel Evolutionary Solver using Differential Evolution.
    Finds optimal [P1, P2, ..., Pn, BundlePrice].
    """
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)

    def objective(prices):
        indiv_prices = np.array(prices[:n_prods])
        bundle_price = prices[n_prods]

        # Logic: Customer chooses Max Surplus
        surplus_indiv = np.sum(np.maximum(wtp_matrix - indiv_prices, 0), axis=1)
        surplus_bundle = bundle_sum_values - bundle_price
        
        # Vectorized Choice
        buy_bundle = (surplus_bundle >= surplus_indiv) & (surplus_bundle >= 0)
        buy_indiv = (~buy_bundle) & (surplus_indiv > 0)
        
        # Revenue Calculation
        rev_bundle = np.sum(buy_bundle) * bundle_price
        
        # For indiv revenue, we must check which items they bought
        items_bought_mask = (wtp_matrix >= indiv_prices) & buy_indiv[:, None]
        rev_indiv = np.sum(items_bought_mask * indiv_prices)

        return -(rev_bundle + rev_indiv) # Minimize negative revenue

    # Set Bounds
    bounds = []
    for i in range(n_prods):
        max_w = np.max(wtp_matrix[:, i])
        bounds.append((0, max_w * 1.5)) # Allow anchor prices higher than WTP
    bounds.append((0, np.max(bundle_sum_values)))

    res = differential_evolution(objective, bounds, strategy='best1bin', maxiter=50, popsize=15, tol=0.01, seed=42)
    return res.x, -res.fun

def get_customer_breakdown(df, products, optimal_prices):
    """Generates the customer-wise decision table."""
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)
    
    indiv_prices = optimal_prices[:n_prods]
    bundle_price = optimal_prices[n_prods]
    
    rows = []
    for i in range(len(df)):
        s_indiv = np.sum(np.maximum(wtp_matrix[i] - indiv_prices, 0))
        s_bundle = bundle_sum_values[i] - bundle_price
        
        decision = "None"
        revenue = 0
        surplus = 0
        items = "-"
        
        if s_bundle >= s_indiv and s_bundle >= 0:
            decision = "Bundle"
            revenue = bundle_price
            surplus = s_bundle
            items = "All Items"
        elif s_indiv > 0:
            decision = "Individual"
            surplus = s_indiv
            bought_indices = np.where(wtp_matrix[i] >= indiv_prices)[0]
            items = ", ".join([products[k] for k in bought_indices])
            revenue = np.sum(indiv_prices[bought_indices])
            
        rows.append({
            "Customer ID": i + 1,
            "Decision": decision,
            "Items Bought": items.replace("Samsung_", "").replace("_", " "), # Clean up names for display
            "Revenue": revenue,
            "Consumer Surplus": surplus
        })
    return pd.DataFrame(rows)

def generate_demand_curve(df, products, optimal_prices):
    """Generates demand curve data by varying bundle price while keeping indiv prices fixed."""
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)
    indiv_prices = optimal_prices[:n_prods]
    
    # Sweep bundle price from 0 to Max Bundle Sum
    max_val = np.max(bundle_sum_values)
    price_points = np.linspace(0, max_val, 100)
    demand = []
    
    for bp in price_points:
        surplus_indiv = np.sum(np.maximum(wtp_matrix - indiv_prices, 0), axis=1)
        surplus_bundle = bundle_sum_values - bp
        buy_bundle = (surplus_bundle >= surplus_indiv) & (surplus_bundle >= 0)
        demand.append(np.sum(buy_bundle))
        
    return pd.DataFrame({"Price": price_points, "Demand": demand})

# --- MAIN APP (Modified for immediate execution) ---

def main():
    st.title("Dynamic Pricing Optimization Engine")
    st.markdown("This dashboard runs the Mixed Bundling optimization automatically using the embedded data file.")

    # 0. Data Input - REMOVED FILE UPLOADER
    # The file 'Samsung_Sankalp.csv' is loaded directly in the load_data function.
    
    df = load_data()
    products = df.columns.tolist()
    
    # Optimization runs immediately upon page load (no button needed)
    with st.spinner("Running Differential Evolution Solver... Analyzing Customer WTPs..."):
        # Run Calculations
        baseline_rev = calculate_baseline(df, products)
        opt_prices, max_rev = solve_pricing(df, products)
        customer_df = get_customer_breakdown(df, products, opt_prices)
        
        total_surplus = customer_df['Consumer Surplus'].sum()
        uplift = ((max_rev - baseline_rev) / baseline_rev) * 100
        
        # Calculate Stats for AI Insights
        bundle_price = opt_prices[-1]
        sum_indiv_opt = np.sum(opt_prices[:-1])
        discount = ((sum_indiv_opt - bundle_price) / sum_indiv_opt) * 100
        bundle_adoption = (len(customer_df[customer_df['Decision'] == 'Bundle']) / len(df)) * 100
        
        # --- SECTION 1: METRICS ---
        st.markdown("### 1. Financial Overview")
        # ... (Metrics display code remains the same)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Total Revenue (Optimized)</div>
                <div class="metric-value">₹{max_rev:,.0f}</div>
                <div class="metric-delta positive">▲ {uplift:.1f}% vs Separate</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-box" style="border-left-color: #22c55e;">
                <div class="metric-label">Consumer Surplus</div>
                <div class="metric-value">₹{total_surplus:,.0f}</div>
                <div class="metric-delta" style="color:#64748b;">Value Retained by Users</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-box" style="border-left-color: #f59e0b;">
                <div class="metric-label">Bundle Adoption</div>
                <div class="metric-value">{bundle_adoption:.0f}%</div>
                <div class="metric-delta" style="color:#64748b;">Conversion Rate</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("---")

        # --- SECTION 2: SPLIT VIEW (AI & Customers) ---
        c_left, c_right = st.columns([1, 2])
        
        with c_left:
            st.subheader("2. AI Strategic Insights")
            
            # Dynamic Text Generation based on stats
            strategy_text = "Volume Driver" if discount > 15 else "Premium Extraction"
            marketing_focus = "Value-for-Money" if discount > 15 else "Exclusivity & Convenience"
            
            st.markdown(f"""
            <div class="insight-card">
                <div class="recommendation">
                    <strong>🎯 Pricing Strategy: {strategy_text}</strong><br>
                    The solver suggests a <strong>{discount:.1f}% discount</strong> on the bundle. 
                    Individual prices are set high to act as anchors, making the bundle price of 
                    <strong>₹{bundle_price:,.0f}</strong> the rational choice for most buyers.
                </div>
                <div class="recommendation" style="border-left-color: #ec4899;">
                    <strong>📢 Marketing Angle: {marketing_focus}</strong><br>
                    Focus marketing on the "Total Ecosystem Savings". 
                    Highlight that buying the bundle saves <strong>₹{(sum_indiv_opt - bundle_price):,.0f}</strong> 
                    compared to individual items.
                </div>
                <div class="recommendation" style="border-left-color: #f59e0b;">
                    <strong>📉 Competitor Analysis</strong><br>
                    Your optimal bundle price effectively prices each item at 
                    <strong>₹{(bundle_price/len(products)):,.0f}</strong> avg. 
                    Use this unit metric to undercut single-product competitors.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_right:
            st.subheader("Customer Purchase Decisions")
            st.dataframe(
                customer_df,
                column_config={
                    "Customer ID": st.column_config.NumberColumn(format="#%d"),
                    "Revenue": st.column_config.NumberColumn(format="₹%d"),
                    "Consumer Surplus": st.column_config.ProgressColumn(
                        format="₹%d",
                        min_value=0,
                        max_value=int(customer_df['Consumer Surplus'].max()),
                    ),
                    "Decision": st.column_config.TextColumn(),
                },
                use_container_width=True,
                height=350,
                hide_index=True
            )

        st.write("---")

        # --- SECTION 3: PRICING MIXES ---
        st.subheader("3. Optimal Pricing Mix")
        st.markdown("The solver calculated these price points to maximize total revenue:")
        
        cols = st.columns(len(products) + 1)
        
        # Individual Prices
        for i, prod in enumerate(products):
            p_opt = opt_prices[i]
            clean_name = prod.replace("Samsung_", "").replace("_", " ") # Clean up names for display
            with cols[i]:
                st.markdown(f"""
                <div class="price-card">
                    <div class="price-title">{clean_name}</div>
                    <div class="price-tag">₹{p_opt:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Bundle Price
        with cols[-1]:
            st.markdown(f"""
            <div class="price-card bundle-highlight">
                <div class="price-title">ALL-IN BUNDLE</div>
                <div class="price-tag">₹{bundle_price:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("---")

        # --- SECTION 4: DEMAND CURVE ---
        st.subheader("4. Bundle Demand Sensitivity")
        
        demand_data = generate_demand_curve(df, products, opt_prices)
        
        fig = px.line(
            demand_data, x="Price", y="Demand",
            title="Projected Bundle Sales at Different Price Points",
            labels={"Price": "Bundle Price (₹)", "Demand": "Number of Buyers"}
        )
        
        # Add vertical line for optimal price
        fig.add_vline(x=bundle_price, line_dash="dash", line_color="green", annotation_text="Optimal Price")
        fig.update_layout(height=400, hovermode="x unified")
        fig.update_traces(line_color='#3b82f6', fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)')
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
