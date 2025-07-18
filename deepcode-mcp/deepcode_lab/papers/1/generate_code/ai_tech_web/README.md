# 🚀 Mini AI Technology Highlights Webpage

A lightweight, visually striking Streamlit web app for showcasing and promoting the latest AI technologies and news. Features a cyberpunk-inspired style with neon colors, dark backgrounds, and futuristic fonts for a high-tech, energetic look.

## 🎨 Features

- **Cyberpunk Theme**: Dark backgrounds with neon accents (lime, cyan, magenta, yellow)
- **Latest AI News**: Curated news highlights with styled cards
- **Featured Technology**: Rotating showcase of cutting-edge AI technologies
- **Interactive AI Demo**: Try AI responses with cyberpunk-styled interface
- **Responsive Design**: Works on desktop and mobile devices
- **Custom Assets**: Generated cyberpunk logo and background

## 📁 Project Structure

```
ai_tech_web/
├── app.py              # Main Streamlit application
├── data.py             # Data source with news, tech info, and responses
├── requirements.txt    # Python dependencies
├── assets/
│   ├── logo.png        # Cyberpunk-style logo (400x150)
│   └── bg.jpg          # Cyberpunk background (1920x1080)
├── create_assets.py    # Asset generation script
└── README.md           # This file
```

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   cd ai_tech_web
   pip install -r requirements.txt
   ```

2. **Generate Assets** (if needed):
   ```bash
   python create_assets.py
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

4. **Open in Browser**:
   Navigate to `http://localhost:8501`

## 🎯 Key Components

### app.py
- Main Streamlit application with cyberpunk styling
- Custom CSS for neon effects and dark theme
- Responsive layout with header, news, featured tech, demo, and footer sections
- Interactive AI demo with styled input/output

### data.py
- Comprehensive data module with functions for:
  - `get_news_data()`: Latest AI news with categories and colors
  - `get_featured_tech()`: Rotating featured technologies
  - `get_demo_responses()`: AI demo conversation examples
  - `get_social_links()`: Social media and contact links
  - `get_color_scheme()`: Cyberpunk color palette

### Assets
- **logo.png**: Custom cyberpunk logo with neon text effects
- **bg.jpg**: Cyberpunk background with grid patterns and geometric designs

## 🎨 Style Guidelines

- **Colors**: 
  - Primary: #39ff14 (lime), #00fff7 (cyan)
  - Accents: #ff00c8 (magenta), #f5ff00 (yellow)
  - Backgrounds: #18182a, #111a2f, black gradients
- **Fonts**: Orbitron, Audiowide, Roboto Mono (via Google Fonts)
- **Effects**: Glowing text, shadow effects, animated borders

## 🔧 Customization

### Adding New News Items
Edit the `NEWS_DATA` list in `data.py`:
```python
{
    "title": "Your News Title",
    "content": "News content...",
    "date": "2024-01-15",
    "category": "AI Research",
    "accent_color": "#39ff14"
}
```

### Adding New Technologies
Edit the `FEATURED_TECHNOLOGIES` list in `data.py`:
```python
{
    "title": "Your Technology",
    "description": "Technology description...",
    "icon": "🤖",
    "accent_color": "#00fff7",
    "features": ["Feature 1", "Feature 2"]
}
```

### Modifying Colors
Update the `CYBERPUNK_COLORS` dictionary in `data.py`:
```python
CYBERPUNK_COLORS = {
    "primary": "#39ff14",    # Lime green
    "secondary": "#00fff7",  # Cyan
    # ... add more colors
}
```

## 📦 Dependencies

- **streamlit**: Web app framework
- **Pillow**: Image processing (for asset generation)
- **random**: Random selection utilities
- **datetime**: Date/time handling

## 🌟 Optional Extensions

- Add animated neon borders with CSS keyframes
- Implement background music or audio cues
- Add dark/light theme toggle
- Connect to live AI APIs for real demos
- Add more interactive elements

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy directly from repository

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**🎮 Enjoy your cyberpunk AI showcase!** 🚀✨