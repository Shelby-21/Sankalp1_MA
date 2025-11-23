import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from io import StringIO
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Samsung Pricing Strategy Engine",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 1. DEFAULT DATA LOADING (FALLBACK ONLY) ---
@st.cache_data
def load_default_data():
    """A small, simple fallback dataset in case no file is uploaded."""
    csv_content = "Samsung_Smartphone,Samsung_Smart_TV_43in,Samsung_Smart_Watch,Samsung_Washing_Machine,Samsung_AC_1.5_Tonne\n50000,40000,10000,30000,50000\n60000,35000,15000,32000,48000\n45000,55000,20000,40000,55000\n70000,45000,25000,50000,60000"
    return pd.read_csv(StringIO(csv_content))

# --- DEMAND FUNCTION LOGIC ---

def get_demand_data(wtp_values):
    """Calculates the Price-Quantity relationship (Demand Curve) from WTP data."""
    prices = np.sort(np.unique(wtp_values))[::-1]
    quantities = [np.sum(wtp_values >= p) for p in prices]
    
    max_wtp = wtp_values.max() if len(wtp_values) > 0 else 0
    prices = np.append(prices, max_wtp * 1.05)
    quantities = np.append(quantities, 0)
    
    prices = np.insert(prices, 0, 0)
    quantities = np.insert(quantities, 0, len(wtp_values))

    return prices, quantities

# --- 2. OPTIMIZATION LOGIC (Cached) ---

def calculate_baseline_separate(df, products):
    """Calculates optimal revenue if products are sold purely separately."""
    total_rev = 0
    prices = {}
    for prod in products:
        wtp = df[prod].values
        best_p = 0
        best_r = 0
        candidates = np.unique(wtp)
        for p in candidates:
            rev = p * np.sum(wtp >= p)
            if rev > best_r:
                best_r = rev
                best_p = p
        prices[prod] = best_p
        total_rev += best_r
    return total_rev, prices

@st.cache_data(show_spinner=False)
def run_evolutionary_optimization(df, products):
    """Runs Differential Evolution to find optimal Mixed Bundling prices."""
    
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_products = len(products)

    def revenue_objective(prices):
        indiv_prices = np.array(prices[:n_products])
        bundle_price = prices[n_products]

        # 1. Surplus if buying individually
        surplus_matrix = np.maximum(wtp_matrix - indiv_prices, 0)
        surplus_indiv = np.sum(surplus_matrix, axis=1)
        
        # 2. Revenue if buying individually (only for items bought)
        buy_flags = (wtp_matrix >= indiv_prices)
        revenue_indiv = np.sum(buy_flags * indiv_prices, axis=1)

        # 3. Surplus if buying bundle
        surplus_bundle = bundle_sum_values - bundle_price

        # 4. Decision
        buy_bundle_mask = (surplus_bundle >= surplus_indiv) & (surplus_bundle >= 0)
        buy_indiv_mask = (~buy_bundle_mask) & (surplus_indiv > 0)

        total_rev = np.sum(buy_bundle_mask * bundle_price) + np.sum(revenue_indiv[buy_indiv_mask])
        return -total_rev # Minimize negative revenue

    # Bounds: 0 to 1.5x Max WTP (to allow high anchors)
    bounds = []
    for i in range(n_products):
        bounds.append((0, np.max(wtp_matrix[:, i]) * 1.5))
    bounds.append((0, np.max(bundle_sum_values))) # Bundle Price

    # Optimization
    result = differential_evolution(
        revenue_objective, 
        bounds, 
        strategy='best1bin', 
        maxiter=100, 
        popsize=15, 
        tol=0.01, 
        seed=42
    )

    return result.x, -result.fun

def generate_customer_details(df, products, opt_prices):
    """Generates detailed table of customer decisions based on optimized prices."""
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_products = len(products)
    
    indiv_prices = opt_prices[:n_products]
    bundle_price = opt_prices[n_products]

    results = []

    for i in range(len(df)):
        row_wtp = wtp_matrix[i]
        surplus_items = np.maximum(row_wtp - indiv_prices, 0)
        surplus_indiv = np.sum(surplus_items)
        
        items_bought = []
        cost_indiv = 0
        for j, p in enumerate(products):
            if row_wtp[j] >= indiv_prices[j]:
                items_bought.append(p)
                cost_indiv += indiv_prices[j]
        
        surplus_bundle = bundle_sum_values[i] - bundle_price
        
        decision = ""
        revenue = 0
        final_surplus = 0
        items_str = ""

        if surplus_bundle >= surplus_indiv and surplus_bundle >= 0:
            decision = "Bundle"
            revenue = bundle_price
            final_surplus = surplus_bundle
            items_str = "Full Ecosystem"
        elif surplus_indiv > 0:
            decision = "Individual"
            revenue = cost_indiv
            final_surplus = surplus_indiv
            items_str = ", ".join([p.replace("Samsung_", "") for p in items_bought])
        else:
            decision = "None"
            revenue = 0
            final_surplus = 0
            items_str = "-"
            
        results.append({
            "Customer ID": i+1,
            "Decision": decision,
            "Items": items_str,
            "Revenue": revenue,
            "Surplus": final_surplus
        })
        
    return pd.DataFrame(results)

# --- MAIN APP UI ---

def main():
    st.title("💎 Samsung Pricing Strategy Optimization")
    st.markdown("This dashboard uses **Evolutionary Algorithms** (Differential Evolution) to find the optimal Mixed Bundling strategy.")

    # Sidebar / File Upload
    with st.sidebar:
        st.header("Data Input")
        uploaded_file = st.file_uploader("Upload WTP Data (CSV)", type=['csv'])
        
    # --- DATA LOADING ---
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_default_data()
        st.warning("Using a small default dataset. Upload your WTP data (CSV) in the sidebar for real-world results.")


    # Identify products (assuming all columns are products)
    products = df.columns.tolist()
    
    # --- COMPUTATION ---
    with st.spinner('Running Evolutionary Solver... finding the optimal price anchors...'):
        # 1. Baseline
        sep_rev, sep_prices = calculate_baseline_separate(df, products)
        
        # 2. Optimization
        opt_params, mixed_rev = run_evolutionary_optimization(df, products)
        
        # 3. Process Results
        customer_df = generate_customer_details(df, products, opt_params)
        total_surplus = customer_df['Surplus'].sum()
        
        # Derived Stats
        uplift_pct = ((mixed_rev - sep_rev) / sep_rev) * 100
        bundle_price_opt = opt_params[len(products)]
        indiv_sum_opt = np.sum(opt_params[:len(products)])
        discount_pct = ((indiv_sum_opt - bundle_price_opt) / indiv_sum_opt) * 100

    # --- SECTION 1: TOP METRICS (Using st.metric) ---
    st.markdown("### 📊 Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Total Revenue",
            value=f"₹{mixed_rev:,.0f}",
            delta=f"{uplift_pct:.1f}% vs Separate Pricing",
            delta_color="normal"
        )
        
    with col2:
        st.metric(
            label="Total Consumer Surplus",
            value=f"₹{total_surplus:,.0f}",
            delta=None # Delta not meaningful for surplus in this context
        )

    st.write("---") 

    # --- NEW SECTION: DEMAND CURVE ---
    st.subheader("📈 Product Demand Curve Analysis")
    st.markdown("Select a product to visualize its **Will-To-Pay (WTP)** distribution, which forms the product's demand curve.")
    
    selected_product = st.selectbox("Select Product for Demand Graph", products)
    
    # Calculate demand data for the selected product
    wtp_values = df[selected_product].values
    prices, quantities = get_demand_data(wtp_values)
    
    # Find optimal price and quantity from separate pricing (for visualization anchor)
    sep_price = sep_prices.get(selected_product, 0)
    sep_quantity = np.sum(wtp_values >= sep_price)
    
    # Find the optimized individual price (Mixed Bundling Anchor)
    opt_index = products.index(selected_product)
    opt_anchor_price = opt_params[opt_index]
    opt_anchor_quantity = np.sum(wtp_values >= opt_anchor_price)

    # Create Plotly figure
    fig = go.Figure()
    
    # 1. Demand Curve (Step Plot)
    fig.add_trace(go.Scatter(
        x=prices, 
        y=quantities,
        mode='lines',
        name='Demand Curve (WTP)',
        line=dict(shape='hv', color='#1d4ed8', width=2),
        hovertemplate='Price: ₹%{x:,.0f}<br>Demand: %{y:.0f} units<extra></extra>'
    ))
    
    # 2. Optimal Separate Price Point (Max Revenue baseline)
    fig.add_trace(go.Scatter(
        x=[sep_price], y=[sep_quantity],
        mode='markers',
        name='Max Revenue (Separate)',
        marker=dict(color='#22c55e', size=10, symbol='star'),
        hovertemplate=f'Separate Price: ₹{sep_price:,.0f}<br>Quantity: {sep_quantity:.0f} units<extra></extra>'
    ))
    
    # 3. Optimized Anchor Price Point (Mixed Bundling)
    fig.add_trace(go.Scatter(
        x=[opt_anchor_price], y=[opt_anchor_quantity],
        mode='markers',
        name='Optimal Anchor Price (Bundling)',
        marker=dict(color='#ef4444', size=10, symbol='circle-open', line=dict(width=2)),
        hovertemplate=f'Anchor Price: ₹{opt_anchor_price:,.0f}<br>Quantity: {opt_anchor_quantity:.0f} units<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Demand Curve for {selected_product.replace("Samsung_", "").replace("_", " ")}',
        xaxis_title='Price (₹)',
        yaxis_title='Quantity Demanded (Customers)',
        hovermode="x unified",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("The **Demand Curve** shows how many customers are willing to buy the product at a given price. The **Optimal Anchor Price** is strategically set by the optimizer to maximize revenue, even if it's not the revenue-maximizing price for that item sold purely individually.")
    
    st.write("---") 
    
    # --- SECTION 2 & 3: SPLIT VIEW ---
    col_left, col_right = st.columns([1, 2])

    # LEFT: AI RECOMMENDATIONS (Using st.info/st.success/st.warning)
    with col_left:
        st.subheader("🤖 AI Recommendations")
        
        anchor_insight = "high" if discount_pct > 15 else "moderate"
        
        st.info(f"""
        **1. The "Anchor Price" Strategy** Individual prices have been set **{anchor_insight}** to act as psychological anchors. The calculated Bundle Price of **₹{bundle_price_opt:,.0f}** offers a **{discount_pct:.1f}% discount** compared to sum-of-parts, maximizing conversion.
        """)
        
        st.warning(f"""
        **2. Cross-Sell Opportunity** Analysis shows **{len(customer_df[customer_df['Decision']=='Bundle'])}** customers chose the full bundle. For the **{len(customer_df[customer_df['Decision']=='Individual'])}** customers buying individually, market a "Mini-Bundle" of the top 2 rejected items to capture mid-tier surplus.
        """)
        
        st.success(f"""
        **3. Competitor Benchmarking** Your optimal bundle price effectively prices each item at **₹{(bundle_price_opt/len(products)):,.0f}** on average. Highlight this "effective unit price" in marketing to undercut single-product competitors.
        """)

    # RIGHT: CUSTOMER TABLE
    with col_right:
        st.subheader("👥 Customer Decisions")
        
        # Styling the dataframe
        st.dataframe(
            customer_df,
            column_config={
                "Customer ID": st.column_config.NumberColumn(format="#%d"),
                "Revenue": st.column_config.NumberColumn(format="₹%d"),
                "Surplus": st.column_config.ProgressColumn(
                    format="₹%d",
                    min_value=0,
                    max_value=int(customer_df['Surplus'].max()),
                ),
                "Decision": st.column_config.TextColumn(),
            },
            use_container_width=True,
            height=350,
            hide_index=True
        )

    # --- SECTION 4: PRICING CONFIGURATION (Simplified Display) ---
    st.write("---")
    st.subheader("🏷️ Optimal Pricing Configuration")
    
    cols = st.columns(len(products) + 1)
    
    # Individual Price Display
    for i, prod in enumerate(products):
        price = opt_params[i]
        clean_name = prod.replace("Samsung_", "").replace("_", " ")
        with cols[i]:
            st.caption(f"**{clean_name}**")
            st.metric(label="Anchor Price", value=f"₹{price:,.0f}")

    # Bundle Card
    with cols[-1]:
        st.caption("**Full Bundle**")
        st.metric(label="Bundle Price", value=f"₹{bundle_price_opt:,.0f}")
        st.markdown(f"*(Sum: ₹{indiv_sum_opt:,.0f})*")


if __name__ == "__main__":
    main()