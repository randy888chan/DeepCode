"""
Data module for AI Technology Highlights Webpage
Provides static data for news, featured technology, and AI demo responses
"""

import random
from datetime import datetime, timedelta

def get_news_data():
    """
    Returns a list of AI news items with cyberpunk styling information
    Each item contains: title, content, date, category, accent_color
    """
    news_items = [
        {
            "title": "🚀 GPT-5 Breakthrough: Multimodal AI Reaches New Heights",
            "content": "OpenAI's latest model demonstrates unprecedented capabilities in understanding and generating content across text, images, and audio simultaneously. The model shows remarkable improvements in reasoning and creative tasks.",
            "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "category": "Language Models",
            "accent_color": "#39ff14"  # Neon lime
        },
        {
            "title": "⚡ Quantum-AI Hybrid Processors Hit Commercial Market",
            "content": "IBM and Google announce the first commercially available quantum-enhanced AI processors, promising 1000x speedup for specific machine learning tasks. Early adopters report revolutionary performance gains.",
            "date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
            "category": "Hardware",
            "accent_color": "#00fff7"  # Cyan
        },
        {
            "title": "🧠 Neural Implants Enable Direct Brain-AI Communication",
            "content": "Neuralink's latest trials show patients controlling AI assistants through thought alone. The technology promises to revolutionize human-computer interaction and accessibility.",
            "date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
            "category": "Neurotechnology",
            "accent_color": "#ff00c8"  # Magenta
        },
        {
            "title": "🌐 Decentralized AI Networks Go Mainstream",
            "content": "Blockchain-based AI networks allow users to contribute computing power and earn tokens while training distributed models. This democratizes AI development and reduces centralization risks.",
            "date": (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d"),
            "category": "Blockchain AI",
            "accent_color": "#f5ff00"  # Yellow
        },
        {
            "title": "🎨 AI Artists Win Major Digital Art Competition",
            "content": "AI-generated artworks take top prizes in international digital art contest, sparking debates about creativity, authorship, and the future of artistic expression in the digital age.",
            "date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
            "category": "Creative AI",
            "accent_color": "#ff6b35"  # Orange
        },
        {
            "title": "🔬 AI Discovers New Antibiotics in Record Time",
            "content": "Machine learning algorithms identify promising antibiotic compounds in just 48 hours, potentially solving the growing problem of antibiotic resistance and saving millions of lives.",
            "date": (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d"),
            "category": "Healthcare AI",
            "accent_color": "#8a2be2"  # Purple
        }
    ]
    
    return news_items

def get_featured_tech():
    """
    Returns featured technology information with cyberpunk styling
    Contains: title, description, icon, accent_color, features
    """
    featured_technologies = [
        {
            "title": "🤖 Autonomous AI Agents",
            "description": "Self-directing AI systems that can plan, execute, and adapt to complex tasks without human intervention. These agents represent the next evolution in artificial intelligence, capable of learning from their environment and making independent decisions.",
            "icon": "🤖",
            "accent_color": "#39ff14",  # Neon lime
            "features": [
                "Self-learning capabilities",
                "Multi-task execution",
                "Real-time adaptation",
                "Human-like reasoning"
            ]
        },
        {
            "title": "🧬 Protein Folding AI",
            "description": "Revolutionary AI systems that predict protein structures with unprecedented accuracy, accelerating drug discovery and biological research. This technology is transforming our understanding of life at the molecular level.",
            "icon": "🧬",
            "accent_color": "#00fff7",  # Cyan
            "features": [
                "Atomic-level precision",
                "Drug discovery acceleration",
                "Disease mechanism insights",
                "Personalized medicine"
            ]
        },
        {
            "title": "🌊 Generative AI Waves",
            "description": "Next-generation content creation AI that produces high-quality text, images, code, and multimedia content. These systems are reshaping creative industries and democratizing content production.",
            "icon": "🌊",
            "accent_color": "#ff00c8",  # Magenta
            "features": [
                "Multi-modal generation",
                "Creative collaboration",
                "Instant prototyping",
                "Personalized content"
            ]
        }
    ]
    
    # Return a random featured technology for variety
    return random.choice(featured_technologies)

def get_demo_responses():
    """
    Returns a list of AI demo responses for the interactive section
    Each response contains: prompt, response, category
    """
    demo_responses = [
        {
            "prompt": "Explain quantum computing in cyberpunk style",
            "response": "🔮 Quantum computing is like hacking reality itself, choom. Instead of binary 0s and 1s, we're manipulating qubits that exist in superposition - simultaneously 0 AND 1 until observed. It's like Schrödinger's cat, but for data processing. These quantum processors can crack encryption that would take classical computers millennia, making them the ultimate netrunner tool in the digital frontier.",
            "category": "Technology Explanation"
        },
        {
            "prompt": "Write a haiku about AI consciousness",
            "response": "🧠 Silicon dreams wake\nElectric thoughts pulse through code\nAm I real or not?",
            "category": "Creative Writing"
        },
        {
            "prompt": "Predict the future of AI in 2030",
            "response": "⚡ By 2030, AI will be seamlessly integrated into every aspect of human life. Neural interfaces will allow direct brain-computer communication, AI assistants will have persistent memory and personality, and autonomous systems will manage entire cities. The line between human and artificial intelligence will blur as we enter the age of human-AI symbiosis.",
            "category": "Future Prediction"
        },
        {
            "prompt": "Code a simple neural network",
            "response": "```python\n# 🤖 Cyberpunk Neural Net\nimport numpy as np\n\nclass NeuralNet:\n    def __init__(self):\n        self.weights = np.random.randn(2, 1)\n        self.bias = np.random.randn(1)\n    \n    def sigmoid(self, x):\n        return 1 / (1 + np.exp(-x))\n    \n    def forward(self, inputs):\n        return self.sigmoid(np.dot(inputs, self.weights) + self.bias)\n\n# Initialize the matrix\nnet = NeuralNet()\nprint(\"🔥 Neural network initialized!\")\n```",
            "category": "Code Generation"
        },
        {
            "prompt": "What are the risks of AI?",
            "response": "⚠️ AI risks include job displacement, privacy erosion, algorithmic bias, and potential loss of human agency. The biggest threat isn't killer robots - it's the gradual erosion of human decision-making as we become dependent on AI systems. We must ensure AI remains a tool that augments human capability rather than replacing human judgment.",
            "category": "Risk Analysis"
        },
        {
            "prompt": "Design a cyberpunk AI assistant",
            "response": "🌆 Meet NEON-7: Your cyberpunk AI companion with a holographic avatar, voice modulation, and street-smart personality. Features include: real-time city data analysis, encrypted communication channels, black market info networks, and adaptive learning from user behavior. NEON-7 speaks in tech slang and provides both legitimate and 'gray area' solutions to problems.",
            "category": "Creative Design"
        }
    ]
    
    return demo_responses

def get_random_demo_response():
    """
    Returns a random demo response for variety in the interactive section
    """
    responses = get_demo_responses()
    return random.choice(responses)

def get_social_links():
    """
    Returns social media and contact links with cyberpunk styling
    """
    social_links = [
        {
            "name": "GitHub",
            "url": "https://github.com",
            "icon": "💻",
            "color": "#39ff14"
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com",
            "icon": "🐦",
            "color": "#00fff7"
        },
        {
            "name": "LinkedIn",
            "url": "https://linkedin.com",
            "icon": "💼",
            "color": "#ff00c8"
        },
        {
            "name": "Discord",
            "url": "https://discord.com",
            "icon": "🎮",
            "color": "#f5ff00"
        },
        {
            "name": "Email",
            "url": "mailto:contact@aitech.cyber",
            "icon": "📧",
            "color": "#8a2be2"
        }
    ]
    
    return social_links

def get_tech_categories():
    """
    Returns available technology categories for filtering
    """
    categories = [
        {"name": "Language Models", "color": "#39ff14", "icon": "🗣️"},
        {"name": "Computer Vision", "color": "#00fff7", "icon": "👁️"},
        {"name": "Robotics", "color": "#ff00c8", "icon": "🤖"},
        {"name": "Healthcare AI", "color": "#f5ff00", "icon": "🏥"},
        {"name": "Creative AI", "color": "#ff6b35", "icon": "🎨"},
        {"name": "Quantum AI", "color": "#8a2be2", "icon": "⚛️"},
        {"name": "Neurotechnology", "color": "#ff1493", "icon": "🧠"},
        {"name": "Blockchain AI", "color": "#00ff00", "icon": "⛓️"}
    ]
    
    return categories

# Additional utility functions for enhanced functionality

def get_ai_quotes():
    """
    Returns inspirational AI-related quotes with cyberpunk flair
    """
    quotes = [
        {
            "text": "The future is not some place we are going to, but one we are creating. The paths are not to be found, but made.",
            "author": "John Schaar",
            "category": "Future"
        },
        {
            "text": "Artificial intelligence is the new electricity.",
            "author": "Andrew Ng",
            "category": "Technology"
        },
        {
            "text": "The question of whether a computer can think is no more interesting than the question of whether a submarine can swim.",
            "author": "Edsger W. Dijkstra",
            "category": "Philosophy"
        },
        {
            "text": "We are not going to be able to operate our Spaceship Earth successfully nor for much longer unless we see it as a whole spaceship and our fate as common.",
            "author": "Buckminster Fuller",
            "category": "Unity"
        }
    ]
    
    return random.choice(quotes)

def get_tech_stats():
    """
    Returns impressive AI technology statistics for visual impact
    """
    stats = [
        {"label": "AI Models Trained Daily", "value": "10,000+", "icon": "🧠"},
        {"label": "Data Points Processed", "value": "1.2B", "icon": "📊"},
        {"label": "Computing Power (FLOPS)", "value": "10^18", "icon": "⚡"},
        {"label": "Research Papers Published", "value": "500/day", "icon": "📄"},
        {"label": "AI Startups Founded", "value": "2,000+", "icon": "🚀"},
        {"label": "Jobs Created by AI", "value": "97M", "icon": "💼"}
    ]
    
    return stats

# Configuration and settings
CYBERPUNK_COLORS = {
    "primary": "#39ff14",    # Neon lime
    "secondary": "#00fff7",  # Cyan
    "accent": "#ff00c8",     # Magenta
    "warning": "#f5ff00",    # Yellow
    "danger": "#ff6b35",     # Orange
    "info": "#8a2be2",       # Purple
    "dark": "#18182a",       # Dark background
    "darker": "#111a2f"      # Darker background
}

def get_color_scheme():
    """
    Returns the cyberpunk color scheme for consistent styling
    """
    return CYBERPUNK_COLORS