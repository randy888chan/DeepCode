#!/usr/bin/env python3
"""
Professional CLI Interface Module
专业CLI界面模块 - 包含logo、颜色定义和界面组件
"""

import os
import time
import sys
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import filedialog

class Colors:
    """ANSI color codes for terminal styling"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Gradient colors
    PURPLE = '\033[35m'
    MAGENTA = '\033[95m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'

class CLIInterface:
    """Professional CLI interface with modern styling"""
    
    def __init__(self):
        self.uploaded_file = None
        self.is_running = True
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_logo(self):
        """Print a beautiful ASCII logo with gradient colors"""
        logo = f"""
{Colors.CYAN}╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║  {Colors.MAGENTA}██████╗  █████╗ ██████╗ ███████╗██████╗     ████████╗ ██████╗                {Colors.CYAN}║
║  {Colors.PURPLE}██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗    ╚══██╔══╝██╔═══██╗               {Colors.CYAN}║
║  {Colors.BLUE}██████╔╝███████║██████╔╝█████╗  ██████╔╝       ██║   ██║   ██║               {Colors.CYAN}║
║  {Colors.OKBLUE}██╔═══╝ ██╔══██║██╔═══╝ ██╔══╝  ██╔══██╗       ██║   ██║   ██║               {Colors.CYAN}║
║  {Colors.OKCYAN}██║     ██║  ██║██║     ███████╗██║  ██║       ██║   ╚██████╔╝               {Colors.CYAN}║
║  {Colors.GREEN}╚═╝     ╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝       ╚═╝    ╚═════╝                {Colors.CYAN}║
║                                                                               ║
║          {Colors.YELLOW} ██████╗ ██████╗ ██████╗ ███████╗    ███████╗███╗   ██╗ ██████╗ ██╗███╗   ██╗███████╗{Colors.CYAN}║
║          {Colors.WARNING}██╔════╝██╔═══██╗██╔══██╗██╔════╝    ██╔════╝████╗  ██║██╔════╝ ██║████╗  ██║██╔════╝{Colors.CYAN}║
║          {Colors.OKGREEN}██║     ██║   ██║██║  ██║█████╗      █████╗  ██╔██╗ ██║██║  ███╗██║██╔██╗ ██║█████╗  {Colors.CYAN}║
║          {Colors.OKBLUE}██║     ██║   ██║██║  ██║██╔══╝      ██╔══╝  ██║╚██╗██║██║   ██║██║██║╚██╗██║██╔══╝  {Colors.CYAN}║
║          {Colors.PURPLE}╚██████╗╚██████╔╝██████╔╝███████╗    ███████╗██║ ╚████║╚██████╔╝██║██║ ╚████║███████╗{Colors.CYAN}║
║          {Colors.MAGENTA} ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝    ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝╚══════╝{Colors.CYAN}║
║                                                                               ║
║  {Colors.BOLD}{Colors.YELLOW}🚀 AI-Powered Research Paper to Code Generation Engine 🚀{Colors.ENDC}{Colors.CYAN}                 ║
║                                                                               ║
║  {Colors.GREEN}✨ Features:{Colors.ENDC}                                                              {Colors.CYAN}║
║     {Colors.OKCYAN}• Intelligent PDF Analysis & Code Extraction                              {Colors.CYAN}║
║     {Colors.OKCYAN}• Advanced Document Processing with Docling                              {Colors.CYAN}║
║     {Colors.OKCYAN}• Multi-format Support (PDF, DOCX, PPTX, HTML)                          {Colors.CYAN}║
║     {Colors.OKCYAN}• Smart File Upload Interface                                           {Colors.CYAN}║
║     {Colors.OKCYAN}• Automated GitHub Repository Management                                 {Colors.CYAN}║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
        print(logo)
        
    def print_welcome_banner(self):
        """Print welcome banner with version info"""
        banner = f"""
{Colors.BOLD}{Colors.CYAN}╔═══════════════════════════════════════════════════════════════════════════════╗
║                              WELCOME TO PAPER-TO-CODE                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  {Colors.YELLOW}Version: 2.0.0 | Build: Professional Edition                                 {Colors.CYAN}║
║  {Colors.GREEN}Status: Ready | Engine: Initialized                                          {Colors.CYAN}║
║  {Colors.PURPLE}Author: AI Research Team | License: MIT                                      {Colors.CYAN}║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
        print(banner)
        
    def print_separator(self, char="═", length=79, color=Colors.CYAN):
        """Print a styled separator line"""
        print(f"{color}{char * length}{Colors.ENDC}")
        
    def print_status(self, message: str, status_type: str = "info"):
        """Print status message with appropriate styling"""
        status_styles = {
            "success": f"{Colors.OKGREEN}✅",
            "error": f"{Colors.FAIL}❌",
            "warning": f"{Colors.WARNING}⚠️ ",
            "info": f"{Colors.OKBLUE}ℹ️ ",
            "processing": f"{Colors.YELLOW}⏳",
            "upload": f"{Colors.PURPLE}📁",
            "download": f"{Colors.CYAN}📥",
            "analysis": f"{Colors.MAGENTA}🔍"
        }
        
        icon = status_styles.get(status_type, status_styles["info"])
        print(f"{icon} {Colors.BOLD}{message}{Colors.ENDC}")
        
    def create_menu(self):
        """Create an interactive menu"""
        menu = f"""
{Colors.BOLD}{Colors.CYAN}╔═══════════════════════════════════════════════════════════════════════════════╗
║                                MAIN MENU                                      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  {Colors.OKGREEN}🌐 [U] Process URL       {Colors.CYAN}│  {Colors.PURPLE}📁 [F] Upload File    {Colors.CYAN}│  {Colors.FAIL}❌ [Q] Quit{Colors.CYAN}         ║
║                                                                               ║
║  {Colors.YELLOW}📝 Enter a research paper URL (arXiv, IEEE, ACM, etc.)                      {Colors.CYAN}║
║  {Colors.YELLOW}   or upload a PDF/DOC file for intelligent analysis                        {Colors.CYAN}║
║                                                                               ║
║  {Colors.OKCYAN}💡 Tip: Press 'F' to open file browser or 'U' to enter URL manually        {Colors.CYAN}║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
        print(menu)
        
    def get_user_input(self):
        """Get user input with styled prompt"""
        print(f"\n{Colors.BOLD}{Colors.OKCYAN}➤ Your choice: {Colors.ENDC}", end="")
        return input().strip().lower()
        
    def upload_file_gui(self) -> Optional[str]:
        """Modern file upload interface using tkinter"""
        def select_file():
            # Create a hidden root window
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            root.attributes('-topmost', True)  # Bring to front
            
            # Set window icon and title
            root.title("Paper-to-Code Engine - File Selector")
            
            # Configure file dialog
            file_types = [
                ("Research Papers", "*.pdf"),
                ("Word Documents", "*.docx;*.doc"),
                ("PowerPoint Files", "*.pptx;*.ppt"),
                ("HTML Files", "*.html;*.htm"),
                ("Text Files", "*.txt;*.md"),
                ("All Supported", "*.pdf;*.docx;*.doc;*.pptx;*.ppt;*.html;*.htm;*.txt;*.md"),
                ("All Files", "*.*")
            ]
            
            file_path = filedialog.askopenfilename(
                title="🚀 Select Research Paper File - Paper-to-Code Engine",
                filetypes=file_types,
                initialdir=os.getcwd()
            )
            
            root.destroy()
            return file_path
        
        self.print_status("Opening file browser dialog...", "upload")
        file_path = select_file()
        
        if file_path:
            # Validate file
            if not os.path.exists(file_path):
                self.print_status("File not found!", "error")
                return None
                
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            file_ext = Path(file_path).suffix.lower()
            
            # Display file info with beautiful formatting
            file_name = Path(file_path).name
            directory = str(Path(file_path).parent)
            
            # Truncate long paths for display
            if len(file_name) > 50:
                display_name = file_name[:47] + "..."
            else:
                display_name = file_name
                
            if len(directory) > 49:
                display_dir = "..." + directory[-46:]
            else:
                display_dir = directory
            
            print(f"""
{Colors.OKGREEN}╔═══════════════════════════════════════════════════════════════════════════════╗
║                               FILE SELECTED                                   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  {Colors.BOLD}📄 File Name:{Colors.ENDC} {Colors.CYAN}{display_name:<50}{Colors.OKGREEN}║
║  {Colors.BOLD}📁 Directory:{Colors.ENDC} {Colors.YELLOW}{display_dir:<49}{Colors.OKGREEN}║
║  {Colors.BOLD}📊 File Size:{Colors.ENDC} {Colors.PURPLE}{file_size:.2f} MB{Colors.OKGREEN}                                      ║
║  {Colors.BOLD}🔖 File Type:{Colors.ENDC} {Colors.MAGENTA}{file_ext.upper():<50}{Colors.OKGREEN}║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
""")
            
            self.print_status(f"File successfully selected: {file_name}", "success")
            return file_path
        else:
            self.print_status("No file selected", "warning")
            return None
            
    def get_url_input(self) -> str:
        """Get URL input with validation and examples"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔═══════════════════════════════════════════════════════════════════════════════╗")
        print(f"║                              URL INPUT                                        ║")
        print(f"╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}")
        
        print(f"\n{Colors.YELLOW}📝 Supported URL Examples:{Colors.ENDC}")
        print(f"   {Colors.CYAN}• arXiv: https://arxiv.org/pdf/2403.00813")
        print(f"   {Colors.CYAN}• arXiv: @https://arxiv.org/pdf/2403.00813")
        print(f"   {Colors.CYAN}• IEEE:  https://ieeexplore.ieee.org/document/...")
        print(f"   {Colors.CYAN}• ACM:   https://dl.acm.org/doi/...")
        print(f"   {Colors.CYAN}• Direct PDF: https://example.com/paper.pdf{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}{Colors.OKCYAN}🌐 Enter paper URL: {Colors.ENDC}", end="")
        url = input().strip()
        
        if url:
            # Basic URL validation
            if any(domain in url.lower() for domain in ['arxiv.org', 'ieee', 'acm.org', '.pdf', 'researchgate']):
                self.print_status(f"URL received: {url}", "success")
                return url
            else:
                self.print_status("URL appears valid, proceeding...", "info")
                return url
        else:
            self.print_status("No URL provided", "warning")
            return ""
            
    def show_progress_bar(self, message: str, duration: float = 2.0):
        """Show a progress animation with enhanced styling"""
        print(f"\n{Colors.YELLOW}{message}{Colors.ENDC}")
        
        # Progress bar animation with different styles
        bar_length = 50
        for i in range(bar_length + 1):
            percent = (i / bar_length) * 100
            filled = "█" * i
            empty = "░" * (bar_length - i)
            
            # Color gradient effect
            if percent < 33:
                color = Colors.FAIL
            elif percent < 66:
                color = Colors.WARNING
            else:
                color = Colors.OKGREEN
                
            print(f"\r{color}[{filled}{empty}] {percent:6.1f}%{Colors.ENDC}", end="", flush=True)
            time.sleep(duration / bar_length)
        
        print(f"\n{Colors.OKGREEN}✅ {message} completed!{Colors.ENDC}\n")
        
    def show_spinner(self, message: str, duration: float = 1.0):
        """Show a spinner animation"""
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        end_time = time.time() + duration
        
        while time.time() < end_time:
            for char in spinner_chars:
                print(f"\r{Colors.CYAN}{char} {Colors.BOLD}{message}{Colors.ENDC}", end="", flush=True)
                time.sleep(0.1)
                if time.time() >= end_time:
                    break
                    
        print(f"\r{Colors.OKGREEN}✅ {Colors.BOLD}{message} - Done!{Colors.ENDC}")
        
    def print_results_header(self):
        """Print results section header"""
        header = f"""
{Colors.OKGREEN}╔═══════════════════════════════════════════════════════════════════════════════╗
║                             PROCESSING RESULTS                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
        print(header)
        
    def print_error_box(self, title: str, error_msg: str):
        """Print error message in a styled box"""
        print(f"""
{Colors.FAIL}╔═══════════════════════════════════════════════════════════════════════════════╗
║                                  ERROR                                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  {Colors.BOLD}Title: {title:<66}{Colors.FAIL}║
║  {Colors.BOLD}Error: {error_msg:<66}{Colors.FAIL}║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
""")
        
    def print_goodbye(self):
        """Print goodbye message"""
        goodbye = f"""
{Colors.BOLD}{Colors.YELLOW}╔═══════════════════════════════════════════════════════════════════════════════╗
║                                GOODBYE!                                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  {Colors.CYAN}Thank you for using Paper-to-Code Engine!                                  {Colors.YELLOW}║
║  {Colors.GREEN}🌟 Star us on GitHub: https://github.com/your-repo                        {Colors.YELLOW}║
║  {Colors.PURPLE}📧 Contact: support@paper-to-code.com                                     {Colors.YELLOW}║
║  {Colors.MAGENTA}🐛 Report issues: https://github.com/your-repo/issues                    {Colors.YELLOW}║
║                                                                               ║
║  {Colors.OKGREEN}✨ Happy coding! See you next time! ✨                                   {Colors.YELLOW}║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
        print(goodbye)
        
    def ask_continue(self) -> bool:
        """Ask user if they want to continue"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Press Enter to continue or 'q' to quit: {Colors.ENDC}", end="")
        choice = input().strip().lower()
        return choice not in ['q', 'quit', 'exit'] 