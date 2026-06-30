import streamlit as st
import pickle
import numpy as np

# Page Setup
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

# Load the trained model
@st.cache_resource
def load_model():
    return pickle.load(open("model_bundle.pkl", "rb"))

bundle       = load_model()
model        = bundle["model"]
encoders     = bundle["encoders"]
feature_cols = bundle["feature_cols"]

def encode(col, value):
    le = encoders.get(col)
    if le is None:
        return 0
    try:
        return int(le.transform([value])[0])
    except ValueError:
        return 0

# Basic Styling
st.markdown("""
<style>
body { background-color: #f9f9f9; }
.title { font-size: 2rem; font-weight: 700; color: #1a1a1a; margin-bottom: 4px; }
.subtitle { color: #555; font-size: 1rem; margin-bottom: 24px; }
.result-box { background: #e8f5e9; border: 2px solid #4caf50; border-radius: 12px;
              padding: 24px; text-align: center; margin-top: 20px; }
.result-label { font-size: 1rem; color: #388e3c; font-weight: 600; margin-bottom: 6px; }
.result-price { font-size: 2.4rem; font-weight: 800; color: #2e7d32; }
.result-note  { font-size: 0.85rem; color: #666; margin-top: 8px; }
.section-title { font-size: 1rem; font-weight: 700; color: #333;
                 border-left: 4px solid #4caf50; padding-left: 10px;
                 margin: 24px 0 12px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🚗 Used Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Fill in the car details below and click the button to get a price estimate.</div>', unsafe_allow_html=True)

st.divider()

# Data
MAKES = ["Maruti Suzuki", "Hyundai", "Honda", "Ford", "Toyota", "Mahindra", "Tata",
         "BMW", "Audi", "Mercedes-Benz", "Kia", "MG", "Jeep", "Land Rover", "Nissan",
         "Chevrolet", "Fiat", "Volkswagen", "Datsun", "Mitsubishi", "Porsche",
         "Jaguar", "Lexus", "MINI", "Isuzu"]

MODELS = {
    "Maruti Suzuki": ["Swift", "Baleno", "Alto", "WagonR", "Dzire", "Vitara Brezza", "Ertiga", "Ciaz", "S-Cross", "Celerio", "Ignis", "XL6", "Grand Vitara", "Fronx", "Jimny"],
    "Hyundai":       ["i20", "Creta", "Verna", "Tucson", "i10", "Venue", "Santro", "Alcazar", "Exter", "Aura", "Kona Electric"],
    "Honda":         ["City", "Amaze", "WR-V", "Jazz", "CR-V", "Elevate", "HR-V"],
    "Ford":          ["EcoSport", "Endeavour", "Figo", "Aspire", "Freestyle", "Mustang"],
    "Toyota":        ["Innova Crysta", "Fortuner", "Glanza", "Urban Cruiser", "Camry", "Vellfire", "Hilux", "Hyryder"],
    "Mahindra":      ["Scorpio", "XUV500", "XUV700", "Thar", "Bolero", "KUV100", "Marazzo", "XUV300", "BE 6"],
    "Tata":          ["Nexon", "Harrier", "Safari", "Tiago", "Tigor", "Altroz", "Punch", "Curvv", "Sierra"],
    "BMW":           ["3 Series", "5 Series", "7 Series", "X1", "X3", "X5", "X7", "M3", "M5", "2 Series"],
    "Audi":          ["A4", "A6", "A8", "Q3", "Q5", "Q7", "Q8", "TT", "R8", "e-tron"],
    "Mercedes-Benz": ["C-Class", "E-Class", "S-Class", "GLA", "GLC", "GLE", "GLS", "A-Class", "CLA", "AMG GT"],
    "Kia":           ["Seltos", "Sonet", "Carnival", "EV6", "Carens"],
    "MG":            ["Hector", "Astor", "Gloster", "ZS EV", "Comet EV"],
    "Jeep":          ["Compass", "Wrangler", "Grand Cherokee", "Meridian"],
    "Land Rover":    ["Range Rover", "Range Rover Sport", "Discovery", "Defender", "Freelander"],
    "Nissan":        ["Magnite", "Kicks", "Terrano", "GT-R"],
    "Chevrolet":     ["Beat", "Cruze", "Spark", "Tavera", "Trailblazer"],
    "Fiat":          ["Punto", "Linea", "Abarth 595"],
    "Volkswagen":    ["Polo", "Vento", "Tiguan", "Taigun", "Virtus", "Passat"],
    "Datsun":        ["redi-GO", "GO", "GO+"],
    "Mitsubishi":    ["Outlander", "Pajero Sport", "Eclipse Cross"],
    "Porsche":       ["Cayenne", "Macan", "911", "Panamera", "Taycan"],
    "Jaguar":        ["XE", "XF", "XJ", "F-Pace", "E-Pace", "I-Pace", "F-Type"],
    "Lexus":         ["ES", "NX", "RX", "LX", "UX", "LC"],
    "MINI":          ["Cooper", "Countryman", "Clubman", "Convertible"],
    "Isuzu":         ["D-Max", "MU-X", "MU-7"],
}

OWNER_LBLS   = ["First Owner", "Second Owner", "Third Owner", "Fourth Owner", "4+ Owners", "Unregistered"]
OWNERS       = ["First",       "Second",       "Third",       "Fourth",       "4 or More",  "UnRegistered Car"]
SELLER_LBLS  = ["Individual",  "Corporate",    "Commercial"]
SELLER_TYPES = ["Individual",  "Corporate",    "Commercial Registration"]
COLORS       = ["White", "Silver", "Grey", "Black", "Blue", "Red", "Brown", "Beige",
                "Maroon", "Gold", "Bronze", "Green", "Orange", "Yellow", "Pink", "Purple", "Others"]
LOCATIONS    = sorted(["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata",
                        "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Gurgaon",
                        "Noida", "Surat", "Indore", "Bhopal", "Nagpur", "Patna", "Goa",
                        "Faridabad", "Agra", "Amritsar", "Aurangabad", "Ghaziabad",
                        "Mysore", "Navi Mumbai", "Nashik", "Vadodara", "Varanasi",
                        "Guwahati", "Ranchi", "Coimbatore", "Ludhiana", "Jalandhar",
                        "Meerut", "Raipur"])

# Form
st.markdown('<div class="section-title">Basic Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    make       = st.selectbox("Car Brand", MAKES)
with col2:
    model_name = st.selectbox("Car Model", MODELS.get(make, ["Other"]))

col3, col4 = st.columns(2)
with col3:
    year      = st.selectbox("Year of Manufacture", list(range(2022, 1999, -1)), index=3)
with col4:
    kilometer = st.number_input("KM Driven", min_value=0, max_value=2_000_000, value=50_000, step=1000)

st.markdown('<div class="section-title">Car Type</div>', unsafe_allow_html=True)

col5, col6, col7 = st.columns(3)
with col5:
    fuel_type    = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "Hybrid", "LPG"])
with col6:
    transmission = st.selectbox("Gear Type", ["Manual", "Automatic"])
with col7:
    drivetrain   = st.selectbox("Drive Type", ["FWD", "RWD", "AWD"])

st.markdown('<div class="section-title">Ownership & Seller</div>', unsafe_allow_html=True)

col8, col9, col10 = st.columns(3)
with col8:
    owner_label  = st.selectbox("Who owned this car before?", OWNER_LBLS)
    owner        = OWNERS[OWNER_LBLS.index(owner_label)]
with col9:
    seller_label = st.selectbox("Who is selling?", SELLER_LBLS)
    seller_type  = SELLER_TYPES[SELLER_LBLS.index(seller_label)]
with col10:
    color        = st.selectbox("Car Color", COLORS)

st.markdown('<div class="section-title">Engine Details</div>', unsafe_allow_html=True)

col11, col12, col13 = st.columns(3)
with col11:
    engine     = st.number_input("Engine Size (CC)", min_value=600,  max_value=7000,  value=1197, step=1)
with col12:
    max_power  = st.number_input("Power (bhp)",      min_value=30.0, max_value=800.0, value=82.0,  step=0.1)
with col13:
    max_torque = st.number_input("Torque (Nm)",      min_value=50.0, max_value=900.0, value=113.0, step=1.0)

st.markdown('<div class="section-title">Size & Capacity</div>', unsafe_allow_html=True)

col14, col15, col16 = st.columns(3)
with col14:
    length  = st.number_input("Length (mm)", min_value=2000, max_value=6000, value=3990, step=1)
with col15:
    width   = st.number_input("Width (mm)",  min_value=1300, max_value=2500, value=1695, step=1)
with col16:
    height  = st.number_input("Height (mm)", min_value=1000, max_value=2500, value=1520, step=1)

col17, col18 = st.columns(2)
with col17:
    seating   = st.selectbox("Number of Seats", [2, 4, 5, 6, 7, 8], index=2)
with col18:
    fuel_tank = st.number_input("Fuel Tank Size (Litres)", min_value=10, max_value=120, value=37, step=1)

st.markdown('<div class="section-title">Location</div>', unsafe_allow_html=True)

location = st.selectbox("City where the car is being sold", LOCATIONS, index=LOCATIONS.index("Delhi"))

st.divider()

# Predict Button
if st.button("🔍 Estimate Price", use_container_width=True):

    car_age = 2025 - int(year)

    row = {
        "Make":               encode("Make", make),
        "Kilometer":          float(kilometer),
        "Fuel Type":          encode("Fuel Type", fuel_type),
        "Transmission":       encode("Transmission", transmission),
        "Location":           encode("Location", location),
        "Color":              encode("Color", color),
        "Owner":              encode("Owner", owner),
        "Seller Type":        encode("Seller Type", seller_type),
        "Engine":             float(engine),
        "Max Power":          float(max_power),
        "Max Torque":         float(max_torque),
        "Drivetrain":         encode("Drivetrain", drivetrain),
        "Length":             float(length),
        "Width":              float(width),
        "Height":             float(height),
        "Seating Capacity":   float(seating),
        "Fuel Tank Capacity": float(fuel_tank),
        "Car_Age":            float(car_age),
    }

    X     = np.array([[row[col] for col in feature_cols]])
    price = float(model.predict(X)[0])

    # Format price in Indian style
    if price >= 10_000_000:
        formatted = f"₹ {price/10_000_000:.2f} Crore"
    elif price >= 100_000:
        formatted = f"₹ {price/100_000:.2f} Lakh"
    else:
        formatted = f"₹ {price:,.0f}"

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Estimated Selling Price</div>
        <div class="result-price">{formatted}</div>
        <div class="result-note">{year} {make} {model_name} &nbsp;|&nbsp; {int(kilometer):,} km driven &nbsp;|&nbsp; {fuel_type} &nbsp;|&nbsp; {transmission}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Summary of what you entered")
    st.write(f"- **Brand:** {make}  |  **Model:** {model_name}")
    st.write(f"- **Year:** {year}  →  Car is {car_age} years old")
    st.write(f"- **KM Driven:** {int(kilometer):,} km")
    st.write(f"- **Fuel:** {fuel_type}  |  **Gear:** {transmission}")
    st.write(f"- **Owner:** {owner_label}  |  **Seller:** {seller_label}")
    st.write(f"- **Engine:** {engine} CC  |  **Power:** {max_power} bhp  |  **Torque:** {max_torque} Nm")
    st.write(f"- **City:** {location}")
