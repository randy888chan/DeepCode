import streamlit as st
import data
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="AI Tech Highlights",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for cyberpunk theme
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Audiowide&family=Roboto+Mono:wght@300;400;700&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #18182a 0%, #111a2f 50%, #0a0a0a 100%);
        color: #ffffff;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main title styling */
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #39ff14, #00fff7, #ff00c8, #f5ff00);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        text-shadow: 0 0 30px rgba(57, 255, 20, 0.5);
        margin-bottom: 2rem;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Audiowide', cursive;
        font-size: 2rem;
        color: #00fff7;
        text-shadow: 0 0 20px rgba(0, 255, 247, 0.7);
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #00fff7;
        padding-bottom: 0.5rem;
    }
    
    /* News cards */
    .news-card {
        background: linear-gradient(135deg, rgba(57, 255, 20, 0.1), rgba(0, 255, 247, 0.1));
        border: 2px solid #39ff14;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 25px rgba(57, 255, 20, 0.3);
        transition: all 0.3s ease;
    }
    
    .news-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 35px rgba(57, 255, 20, 0.5);
        border-color: #00fff7;
    }
    
    .news-title {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.3rem;
        font-weight: 700;
        color: #39ff14;
        margin-bottom: 0.5rem;
    }
    
    .news-content {
        font-family: 'Roboto Mono', monospace;
        color: #ffffff;
        line-height: 1.6;
    }
    
    /* Featured tech panel */
    .featured-panel {
        background: linear-gradient(135deg, rgba(255, 0, 200, 0.2), rgba(245, 255, 0, 0.2));
        border: 3px solid #ff00c8;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 0 40px rgba(255, 0, 200, 0.4);
        text-align: center;
    }
    
    .featured-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #ff00c8;
        text-shadow: 0 0 25px rgba(255, 0, 200, 0.8);
        margin-bottom: 1rem;
    }
    
    .featured-description {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.1rem;
        color: #ffffff;
        line-height: 1.8;
    }
    
    .featured-features {
        font-family: 'Orbitron', monospace;
        font-size: 1rem;
        color: #39ff14;
        margin-top: 1.5rem;
        text-align: left;
        display: inline-block;
    }
    
    .feature-item {
        margin: 0.5rem 0;
        padding: 0.3rem 0;
        border-left: 3px solid #39ff14;
        padding-left: 1rem;
    }
    
    /* Interactive demo section */
    .demo-container {
        background: linear-gradient(135deg, rgba(0, 255, 247, 0.1), rgba(57, 255, 20, 0.1));
        border: 2px solid #00fff7;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 0 30px rgba(0, 255, 247, 0.3);
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(45deg, #ff00c8, #00fff7);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        font-size: 1.1rem;
        text-transform: uppercase;
        box-shadow: 0 0 20px rgba(255, 0, 200, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 30px rgba(255, 0, 200, 0.7);
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(90deg, #18182a, #111a2f);
        border-top: 2px solid #39ff14;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
    }
    
    .footer-links {
        font-family: 'Roboto Mono', monospace;
        color: #00fff7;
        text-decoration: none;
        margin: 0 1rem;
        transition: all 0.3s ease;
    }
    
    .footer-links:hover {
        color: #39ff14;
        text-shadow: 0 0 15px rgba(57, 255, 20, 0.8);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(0, 0, 0, 0.7);
        border: 2px solid #00fff7;
        border-radius: 10px;
        color: #ffffff;
        font-family: 'Roboto Mono', monospace;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #39ff14;
        box-shadow: 0 0 15px rgba(57, 255, 20, 0.5);
    }
    
    /* Animated border effect */
    @keyframes neon-border {
        0% { border-color: #39ff14; }
        25% { border-color: #00fff7; }
        50% { border-color: #ff00c8; }
        75% { border-color: #f5ff00; }
        100% { border-color: #39ff14; }
    }
    
    .animated-border {
        animation: neon-border 4s linear infinite;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Load custom CSS
    load_css()
    
    # Header section with logo and title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Try to load logo if it exists
        logo_path = "assets/logo.png"
        if os.path.exists(logo_path):
            try:
                logo = Image.open(logo_path)
                st.image(logo, width=200)
            except:
                st.markdown("🚀", unsafe_allow_html=True)
        else:
            st.markdown("🚀", unsafe_allow_html=True)
        
        st.markdown('<h1 class="main-title">AI TECH HIGHLIGHTS</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-family: Roboto Mono; color: #00fff7; font-size: 1.2rem;">Discover the Future of Artificial Intelligence</p>', unsafe_allow_html=True)
    
    # Latest AI News Section
    st.markdown('<h2 class="section-header">⚡ LATEST AI NEWS</h2>', unsafe_allow_html=True)
    
    news_items = data.get_news_data()
    for news in news_items:
        st.markdown(f'''
        <div class="news-card">
            <div class="news-title">{news["title"]}</div>
            <div class="news-content">{news["content"]}</div>
            <div style="margin-top: 1rem; font-size: 0.9rem; color: {news.get("accent_color", "#00fff7")};">
                📅 {news.get("date", "Recent")} | 🏷️ {news.get("category", "AI News")}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Featured Technology Section
    st.markdown('<h2 class="section-header">🔥 FEATURED TECHNOLOGY</h2>', unsafe_allow_html=True)
    
    featured_tech = data.get_featured_tech()
    
    # Format features list as HTML
    features_html = ""
    if "features" in featured_tech and featured_tech["features"]:
        features_html = "<div class='featured-features'>"
        for feature in featured_tech["features"]:
            features_html += f'<div class="feature-item">▶ {feature}</div>'
        features_html += "</div>"
    
    st.markdown(f'''
    <div class="featured-panel animated-border">
        <div class="featured-title">{featured_tech["icon"]} {featured_tech["title"]}</div>
        <div class="featured-description">{featured_tech["description"]}</div>
        {features_html}
    </div>
    ''', unsafe_allow_html=True)
    
    # Interactive AI Demo Section
    st.markdown('<h2 class="section-header">🤖 TRY AI DEMO</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="demo-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            user_input = st.text_input(
                "Enter your AI query:",
                placeholder="Ask me about the latest AI trends...",
                key="ai_demo_input"
            )
        
        with col2:
            if st.button("🚀 ANALYZE", key="demo_button"):
                if user_input:
                    # Simple demo response
                    demo_responses = data.get_demo_responses()
                    import random
                    response = random.choice(demo_responses)
                    st.markdown(f'''
                    <div style="background: rgba(57, 255, 20, 0.1); border: 1px solid #39ff14; 
                                border-radius: 10px; padding: 1rem; margin-top: 1rem;">
                        <strong style="color: #39ff14;">AI Response:</strong><br>
                        <span style="color: #ffffff; font-family: Roboto Mono;">{response["response"]}</span>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.warning("Please enter a query first!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with social links
    st.markdown('<h2 class="section-header">🌐 CONNECT</h2>', unsafe_allow_html=True)
    
    social_links = data.get_social_links()
    footer_links_html = ""
    for link in social_links:
        footer_links_html += f'<a href="{link["url"]}" class="footer-links" style="color: {link["color"]};">{link["icon"]} {link["name"]}</a>'
    
    st.markdown(f'''
    <div class="footer">
        <div style="font-family: Orbitron; font-size: 1.5rem; color: #39ff14; margin-bottom: 1rem;">
            🌐 CONNECT WITH THE FUTURE
        </div>
        <div>
            {footer_links_html}
        </div>
        <div style="margin-top: 1rem; font-family: Roboto Mono; color: #666; font-size: 0.9rem;">
            © 2024 AI Tech Highlights | Powered by Cyberpunk Innovation
        </div>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()