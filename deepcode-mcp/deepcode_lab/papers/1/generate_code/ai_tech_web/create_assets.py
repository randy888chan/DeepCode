#!/usr/bin/env python3
"""
Asset Generator for AI Tech Cyberpunk Web App
Creates placeholder logo and background images with cyberpunk styling
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_cyberpunk_logo(width=400, height=150, filename="assets/logo.png"):
    """Create a cyberpunk-style logo with neon text effect"""
    
    # Create image with dark background
    img = Image.new('RGBA', (width, height), (24, 24, 42, 0))  # Dark transparent background
    draw = ImageDraw.Draw(img)
    
    # Cyberpunk colors
    neon_cyan = (0, 255, 247)
    neon_pink = (255, 0, 200)
    neon_lime = (57, 255, 20)
    
    # Create gradient background effect
    for y in range(height):
        alpha = int(255 * (1 - y / height) * 0.3)
        color = (*neon_cyan, alpha)
        draw.line([(0, y), (width, y)], fill=color)
    
    # Draw main text "AI TECH"
    try:
        # Try to use a system font, fallback to default
        font_size = 48
        font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Main text
    text = "AI TECH"
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw glow effect (multiple layers)
    for offset in range(5, 0, -1):
        glow_color = (*neon_cyan, 50)
        for dx in range(-offset, offset + 1):
            for dy in range(-offset, offset + 1):
                draw.text((x + dx, y + dy), text, font=font, fill=glow_color)
    
    # Draw main text
    draw.text((x, y), text, font=font, fill=neon_cyan)
    
    # Add accent lines
    line_color = neon_pink
    draw.line([(50, height//2 - 30), (width-50, height//2 - 30)], fill=line_color, width=2)
    draw.line([(50, height//2 + 30), (width-50, height//2 + 30)], fill=line_color, width=2)
    
    # Add corner accents
    corner_size = 20
    # Top left
    draw.line([(0, 0), (corner_size, 0)], fill=neon_lime, width=3)
    draw.line([(0, 0), (0, corner_size)], fill=neon_lime, width=3)
    # Top right
    draw.line([(width-corner_size, 0), (width, 0)], fill=neon_lime, width=3)
    draw.line([(width, 0), (width, corner_size)], fill=neon_lime, width=3)
    # Bottom left
    draw.line([(0, height-corner_size), (0, height)], fill=neon_lime, width=3)
    draw.line([(0, height), (corner_size, height)], fill=neon_lime, width=3)
    # Bottom right
    draw.line([(width, height-corner_size), (width, height)], fill=neon_lime, width=3)
    draw.line([(width-corner_size, height), (width, height)], fill=neon_lime, width=3)
    
    return img

def create_cyberpunk_background(width=1920, height=1080, filename="assets/bg.jpg"):
    """Create a cyberpunk-style background image"""
    
    # Create image with dark gradient background
    img = Image.new('RGB', (width, height), (17, 17, 35))
    draw = ImageDraw.Draw(img)
    
    # Cyberpunk colors
    colors = [
        (0, 255, 247, 30),    # Cyan
        (255, 0, 200, 30),    # Pink
        (57, 255, 20, 30),    # Lime
        (245, 255, 0, 30),    # Yellow
    ]
    
    # Create gradient background
    for y in range(height):
        # Create a gradient from dark blue to black
        r = int(17 * (1 - y / height))
        g = int(26 * (1 - y / height))
        b = int(47 * (1 - y / height))
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # Add geometric patterns
    import random
    random.seed(42)  # For consistent results
    
    # Add grid lines
    grid_color = (0, 255, 247, 20)
    grid_spacing = 100
    
    for x in range(0, width, grid_spacing):
        alpha = random.randint(10, 40)
        color = (0, 255, 247)
        # Vertical lines with varying opacity
        for y in range(height):
            if random.random() < 0.3:  # Sparse lines
                draw.point((x, y), fill=color)
    
    for y in range(0, height, grid_spacing):
        alpha = random.randint(10, 40)
        color = (255, 0, 200)
        # Horizontal lines with varying opacity
        for x in range(width):
            if random.random() < 0.3:  # Sparse lines
                draw.point((x, y), fill=color)
    
    # Add some geometric shapes
    for _ in range(20):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = x1 + random.randint(50, 200)
        y2 = y1 + random.randint(50, 200)
        
        color = random.choice([(0, 255, 247), (255, 0, 200), (57, 255, 20)])
        
        # Draw rectangles with low opacity
        draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
    
    # Add circuit-like patterns
    for _ in range(50):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(5, 20)
        color = random.choice([(0, 255, 247), (255, 0, 200), (57, 255, 20)])
        
        # Draw small crosses
        draw.line([(x-size, y), (x+size, y)], fill=color, width=1)
        draw.line([(x, y-size), (x, y+size)], fill=color, width=1)
    
    return img

def main():
    """Generate all assets for the cyberpunk web app"""
    
    # Create assets directory if it doesn't exist
    os.makedirs("assets", exist_ok=True)
    
    print("🎨 Generating cyberpunk assets...")
    
    # Generate logo
    print("📱 Creating logo...")
    logo = create_cyberpunk_logo()
    logo.save("assets/logo.png")
    print("✅ Logo saved to assets/logo.png")
    
    # Generate background
    print("🌃 Creating background...")
    bg = create_cyberpunk_background()
    bg.save("assets/bg.jpg", "JPEG", quality=85)
    print("✅ Background saved to assets/bg.jpg")
    
    print("🚀 All assets generated successfully!")
    print("\nGenerated files:")
    print("- assets/logo.png (400x150 cyberpunk logo)")
    print("- assets/bg.jpg (1920x1080 cyberpunk background)")
    
    # Display file sizes
    try:
        logo_size = os.path.getsize("assets/logo.png")
        bg_size = os.path.getsize("assets/bg.jpg")
        print(f"\nFile sizes:")
        print(f"- logo.png: {logo_size:,} bytes")
        print(f"- bg.jpg: {bg_size:,} bytes")
    except:
        pass

if __name__ == "__main__":
    main()