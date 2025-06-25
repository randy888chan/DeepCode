#!/usr/bin/env python3
"""
Enhanced CLI Interface Module for Paper to Code
增强版CLI界面模块 - 专为Paper to Code设计
"""

import os
import time
import sys
import platform
from pathlib import Path
from typing import Optional
import threading

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
    """Enhanced CLI interface with modern styling for Paper to Code"""
    
    def __init__(self):
        self.uploaded_file = None
        self.is_running = True
        self.processing_history = []
        
        # Check tkinter availability for file dialogs
        self.tkinter_available = True
        try:
            import tkinter as tk
            # Test if tkinter can create a window
            test_root = tk.Tk()
            test_root.withdraw()
            test_root.destroy()
        except Exception:
            self.tkinter_available = False
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_logo(self):
        """Print enhanced ASCII logo for Paper to Code CLI"""
        logo = f"""
{Colors.CYAN}╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║  {Colors.BOLD}{Colors.MAGENTA}██████╗  █████╗ ██████╗ ███████╗██████╗     ████████╗ ██████╗ {Colors.CYAN}               ║
║  {Colors.BOLD}{Colors.PURPLE}██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗    ╚══██╔══╝██╔═══██╗{Colors.CYAN}               ║
║  {Colors.BOLD}{Colors.BLUE}██████╔╝███████║██████╔╝█████╗  ██████╔╝       ██║   ██║   ██║{Colors.CYAN}               ║
║  {Colors.BOLD}{Colors.OKBLUE}██╔═══╝ ██╔══██║██╔═══╝ ██╔══╝  ██╔══██╗       ██║   ██║   ██║{Colors.CYAN}               ║
║  {Colors.BOLD}{Colors.OKCYAN}██║     ██║  ██║██║     ███████╗██║  ██║       ██║   ╚██████╔╝{Colors.CYAN}               ║
║  {Colors.BOLD}{Colors.GREEN}╚═╝     ╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝       ╚═╝    ╚═════╝ {Colors.CYAN}               ║
║                                                                               ║
║  {Colors.BOLD}{Colors.YELLOW} ██████╗ ██████╗ ██████╗ ███████╗    ██████╗██╗     ██╗{Colors.CYAN}                    ║
║  {Colors.BOLD}{Colors.YELLOW}██╔════╝██╔═══██╗██╔══██╗██╔════╝   ██╔════╝██║     ██║{Colors.CYAN}                    ║
║  {Colors.BOLD}{Colors.YELLOW}██║     ██║   ██║██║  ██║█████╗     ██║     ██║     ██║{Colors.CYAN}                    ║
║  {Colors.BOLD}{Colors.YELLOW}██║     ██║   ██║██║  ██║██╔══╝     ██║     ██║     ██║{Colors.CYAN}                    ║
║  {Colors.BOLD}{Colors.YELLOW}╚██████╗╚██████╔╝██████╔╝███████╗   ╚██████╗███████╗██║{Colors.CYAN}                    ║
║  {Colors.BOLD}{Colors.YELLOW} ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝    ╚═════╝╚══════╝╚═╝{Colors.CYAN}                    ║
║                                                                               ║
║  {Colors.BOLD}{Colors.GREEN}🤖 AI-POWERED RESEARCH PAPER REPRODUCTION ENGINE 🚀                    {Colors.CYAN}║
║  {Colors.BOLD}{Colors.GREEN}⚡ INTELLIGENT • AUTOMATED • CUTTING-EDGE ⚡                        {Colors.CYAN}║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
        print(logo)
        
    def print_welcome_banner(self):
        """Print enhanced welcome banner"""
        banner = f"""
{Colors.BOLD}{Colors.CYAN}╔═══════════════════════════════════════════════════════════════════════════════╗
║                          WELCOME TO PAPER-TO-CODE CLI                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  {Colors.YELLOW}Version: 2.0.0 CLI Edition | Build: Professional Command Line             {Colors.CYAN}║
║  {Colors.GREEN}Status: Ready | Engine: Neural Processing Initialized                      {Colors.CYAN}║
║  {Colors.PURPLE}Author: AI Research Team | License: MIT                                    {Colors.CYAN}║
║                                                                               ║
║  {Colors.BOLD}{Colors.OKCYAN}💎 CORE CAPABILITIES:{Colors.ENDC}                                                      {Colors.CYAN}║
║    {Colors.BOLD}{Colors.OKCYAN}▶ Neural PDF Analysis & Code Extraction                               {Colors.CYAN}║
║    {Colors.BOLD}{Colors.OKCYAN}▶ Advanced Multi-Agent Research Pipeline                             {Colors.CYAN}║
║    {Colors.BOLD}{Colors.OKCYAN}▶ Automated Repository Management & Code Generation                  {Colors.CYAN}║
║    {Colors.BOLD}{Colors.OKCYAN}▶ Smart File Processing (PDF•DOCX•PPTX•HTML•TXT)                    {Colors.CYAN}║
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
            "analysis": f"{Colors.MAGENTA}🔍",
            "implementation": f"{Colors.GREEN}⚙️ ",
            "complete": f"{Colors.OKGREEN}🎉"
        }
        
        icon = status_styles.get(status_type, status_styles["info"])
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{Colors.BOLD}{timestamp}{Colors.ENDC}] {icon} {Colors.BOLD}{message}{Colors.ENDC}")
        
    def create_menu(self):
        """Create enhanced interactive menu"""
        menu = f"""
{Colors.BOLD}{Colors.CYAN}╔═══════════════════════════════════════════════════════════════════════════════╗
║                                MAIN MENU                                      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  {Colors.OKGREEN}🌐 [U] Process URL       {Colors.CYAN}│  {Colors.PURPLE}📁 [F] Upload File    {Colors.CYAN}│  {Colors.FAIL}❌ [Q] Quit{Colors.CYAN}         ║
║                                                                               ║
║  {Colors.YELLOW}📝 URL Processing:{Colors.CYAN}                                                         ║
║  {Colors.YELLOW}   ▶ Enter research paper URL (arXiv, IEEE, ACM, etc.)                    {Colors.CYAN}║
║  {Colors.YELLOW}   ▶ Supports direct PDF links and academic paper pages                   {Colors.CYAN}║
║                                                                               ║
║  {Colors.PURPLE}📁 File Processing:{Colors.CYAN}                                                        ║
║  {Colors.PURPLE}   ▶ Upload PDF, DOCX, PPTX, HTML, or TXT files                          {Colors.CYAN}║
║  {Colors.PURPLE}   ▶ Intelligent file format detection and processing                     {Colors.CYAN}║
║                                                                               ║
║  {Colors.OKCYAN}🔄 Processing Pipeline:{Colors.CYAN}                                                    ║
║  {Colors.OKCYAN}   ▶ Document analysis → Reference extraction → Code generation           {Colors.CYAN}║
║  {Colors.OKCYAN}   ▶ Multi-agent research pipeline with progress tracking                 {Colors.CYAN}║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
        print(menu)
        
    def get_user_input(self):
        """Get user input with styled prompt"""
        print(f"\n{Colors.BOLD}{Colors.OKCYAN}➤ Your choice: {Colors.ENDC}", end="")
        return input().strip().lower()
        
    def upload_file_gui(self) -> Optional[str]:
        """Enhanced file upload interface with better error handling"""
        if not self.tkinter_available:
            self.print_status("GUI file dialog not available - using manual input", "warning")
            return self._get_manual_file_path()
            
        def select_file():
            try:
                import tkinter as tk
                from tkinter import filedialog
                
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                
                file_types = [
                    ("Research Papers", "*.pdf;*.docx;*.doc"),
                    ("PDF Files", "*.pdf"),
                    ("Word Documents", "*.docx;*.doc"),
                    ("PowerPoint Files", "*.pptx;*.ppt"),
                    ("HTML Files", "*.html;*.htm"),
                    ("Text Files", "*.txt;*.md"),
                    ("All Files", "*.*")
                ]
                
                if platform.system() == "Darwin":
                    file_types = [
                        ("Research Papers", ".pdf .docx .doc"),
                        ("PDF Files", ".pdf"),
                        ("Word Documents", ".docx .doc"),
                        ("PowerPoint Files", ".pptx .ppt"),
                        ("HTML Files", ".html .htm"),
                        ("Text Files", ".txt .md"),
                        ("All Files", ".*")
                    ]
                
                file_path = filedialog.askopenfilename(
                    title="Select Research Paper File - Paper to Code CLI",
                    filetypes=file_types,
                    initialdir=os.getcwd()
                )
                
                root.destroy()
                return file_path
                
            except Exception as e:
                self.print_status(f"File dialog error: {str(e)}", "error")
                return self._get_manual_file_path()
        
        self.print_status("Opening file browser dialog...", "upload")
        file_path = select_file()
        
        if file_path:
            self.print_status(f"File selected: {os.path.basename(file_path)}", "success")
            return file_path
        else:
            self.print_status("No file selected", "warning")
            return None
            
    def _get_manual_file_path(self) -> Optional[str]:
        """Get file path through manual input with validation"""
        self.print_separator("─", 79, Colors.YELLOW)
        print(f"{Colors.BOLD}{Colors.YELLOW}📁 Manual File Path Input{Colors.ENDC}")
        print(f"{Colors.CYAN}Please enter the full path to your research paper file:{Colors.ENDC}")
        print(f"{Colors.CYAN}Supported formats: PDF, DOCX, PPTX, HTML, TXT, MD{Colors.ENDC}")
        self.print_separator("─", 79, Colors.YELLOW)
        
        while True:
            print(f"\n{Colors.BOLD}{Colors.OKCYAN}📂 File path: {Colors.ENDC}", end="")
            file_path = input().strip()
            
            if not file_path:
                self.print_status("Empty path entered. Please try again or press Ctrl+C to cancel.", "warning")
                continue
                
            file_path = os.path.expanduser(file_path)
            file_path = os.path.abspath(file_path)
            
            if not os.path.exists(file_path):
                self.print_status(f"File not found: {file_path}", "error")
                retry = input(f"{Colors.YELLOW}Try again? (y/n): {Colors.ENDC}").strip().lower()
                if retry != 'y':
                    return None
                continue
                
            if not os.path.isfile(file_path):
                self.print_status(f"Path is not a file: {file_path}", "error")
                continue
                
            supported_extensions = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.html', '.htm', '.txt', '.md'}
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext not in supported_extensions:
                self.print_status(f"Unsupported file format: {file_ext}", "warning")
                proceed = input(f"{Colors.YELLOW}Process anyway? (y/n): {Colors.ENDC}").strip().lower()
                if proceed != 'y':
                    continue
            
            self.print_status(f"File validated: {os.path.basename(file_path)}", "success")
            return file_path
        
    def get_url_input(self) -> str:
        """Enhanced URL input with validation"""
        self.print_separator("─", 79, Colors.GREEN)
        print(f"{Colors.BOLD}{Colors.GREEN}🌐 URL Input Interface{Colors.ENDC}")
        print(f"{Colors.CYAN}Enter a research paper URL from supported platforms:{Colors.ENDC}")
        print(f"{Colors.CYAN}• arXiv (arxiv.org)        • IEEE Xplore (ieeexplore.ieee.org){Colors.ENDC}")
        print(f"{Colors.CYAN}• ACM Digital Library      • SpringerLink • Nature • Science{Colors.ENDC}")
        print(f"{Colors.CYAN}• Direct PDF links         • Academic publisher websites{Colors.ENDC}")
        self.print_separator("─", 79, Colors.GREEN)
        
        while True:
            print(f"\n{Colors.BOLD}{Colors.OKCYAN}🔗 URL: {Colors.ENDC}", end="")
            url = input().strip()
            
            if not url:
                self.print_status("Empty URL entered. Please try again or press Ctrl+C to cancel.", "warning")
                continue
                
            if not url.startswith(('http://', 'https://')):
                self.print_status("URL must start with http:// or https://", "error")
                retry = input(f"{Colors.YELLOW}Try again? (y/n): {Colors.ENDC}").strip().lower()
                if retry != 'y':
                    return ""
                continue
                
            academic_domains = [
                'arxiv.org', 'ieeexplore.ieee.org', 'dl.acm.org',
                'link.springer.com', 'nature.com', 'science.org',
                'scholar.google.com', 'researchgate.net', 'semanticscholar.org'
            ]
            
            is_academic = any(domain in url.lower() for domain in academic_domains)
            if not is_academic and not url.lower().endswith('.pdf'):
                self.print_status("URL doesn't appear to be from a known academic platform", "warning")
                proceed = input(f"{Colors.YELLOW}Process anyway? (y/n): {Colors.ENDC}").strip().lower()
                if proceed != 'y':
                    continue
                    
            self.print_status(f"URL validated: {url}", "success")
            return url
            
    def show_progress_bar(self, message: str, duration: float = 2.0):
        """Show animated progress bar"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{message}{Colors.ENDC}")
        
        bar_length = 50
        for i in range(bar_length + 1):
            percent = (i / bar_length) * 100
            filled = "█" * i
            empty = "░" * (bar_length - i)
            
            print(f"\r{Colors.OKGREEN}[{filled}{empty}] {percent:3.0f}%{Colors.ENDC}", end="", flush=True)
            time.sleep(duration / bar_length)
        
        print(f"\n{Colors.OKGREEN}✓ {message} completed{Colors.ENDC}")
        
    def show_spinner(self, message: str, duration: float = 1.0):
        """Show spinner animation"""
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        end_time = time.time() + duration
        
        print(f"{Colors.BOLD}{Colors.CYAN}{message}... {Colors.ENDC}", end="", flush=True)
        
        i = 0
        while time.time() < end_time:
            print(f"\r{Colors.BOLD}{Colors.CYAN}{message}... {Colors.YELLOW}{spinner_chars[i % len(spinner_chars)]}{Colors.ENDC}", end="", flush=True)
            time.sleep(0.1)
            i += 1
            
        print(f"\r{Colors.BOLD}{Colors.CYAN}{message}... {Colors.OKGREEN}✓{Colors.ENDC}")
        
    def display_processing_stages(self, current_stage: int = 0):
        """Display processing pipeline stages with current progress"""
        stages = [
            ("🚀", "Initialize", "Setting up AI engine"),
            ("📊", "Analyze", "Analyzing paper content"),
            ("📥", "Download", "Processing document"),
            ("🔍", "References", "Analyzing references"),
            ("📋", "Plan", "Generating code plan"),
            ("📦", "Repos", "Downloading repositories"),
            ("🗂️", "Index", "Building code index"),
            ("⚙️", "Implement", "Implementing code")
        ]
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}📋 PROCESSING PIPELINE STATUS{Colors.ENDC}")
        self.print_separator("─", 79, Colors.CYAN)
        
        for i, (icon, name, desc) in enumerate(stages):
            if i < current_stage:
                status = f"{Colors.OKGREEN}✓ COMPLETED{Colors.ENDC}"
            elif i == current_stage:
                status = f"{Colors.YELLOW}⏳ IN PROGRESS{Colors.ENDC}"
            else:
                status = f"{Colors.CYAN}⏸️  PENDING{Colors.ENDC}"
                
            print(f"{icon} {Colors.BOLD}{name:<12}{Colors.ENDC} │ {desc:<25} │ {status}")
            
        self.print_separator("─", 79, Colors.CYAN)
        
    def print_results_header(self):
        """Print results section header"""
        header = f"""
{Colors.BOLD}{Colors.OKGREEN}╔═══════════════════════════════════════════════════════════════════════════════╗
║                              PROCESSING RESULTS                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
        print(header)
        
    def print_error_box(self, title: str, error_msg: str):
        """Print formatted error box"""
        print(f"\n{Colors.FAIL}╔══════════════════════════════════════════════════════════════╗")
        print(f"║ {Colors.BOLD}ERROR: {title:<50}{Colors.FAIL} ║")
        print(f"╠══════════════════════════════════════════════════════════════╣")
        
        words = error_msg.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + word) <= 54:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
            
        for line in lines:
            print(f"║ {line:<56} ║")
            
        print(f"╚══════════════════════════════════════════════════════════════╝{Colors.ENDC}")
        
    def cleanup_cache(self):
        """清理Python缓存文件 / Clean up Python cache files"""
        try:
            self.print_status("Cleaning up cache files...", "info")
            # 清理__pycache__目录
            os.system('find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null')
            # 清理.pyc文件
            os.system('find . -name "*.pyc" -delete 2>/dev/null')
            self.print_status("Cache cleanup completed", "success")
        except Exception as e:
            self.print_status(f"Cache cleanup failed: {e}", "warning")

    def print_goodbye(self):
        """Print goodbye message"""
        # 清理缓存文件
        self.cleanup_cache()
        
        goodbye = f"""
{Colors.BOLD}{Colors.CYAN}╔═══════════════════════════════════════════════════════════════════════════════╗
║                                GOODBYE                                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  {Colors.OKGREEN}🎉 Thank you for using Paper to Code CLI!                                 {Colors.CYAN}║
║                                                                               ║
║  {Colors.YELLOW}🧬 Your research papers have been transformed into working code            {Colors.CYAN}║
║  {Colors.PURPLE}⚡ Keep pushing the boundaries of AI-powered research automation          {Colors.CYAN}║
║                                                                               ║
║  {Colors.OKCYAN}💡 Questions? Feedback? Visit our documentation or GitHub repository      {Colors.CYAN}║
║  {Colors.GREEN}🧹 Cache files cleaned up for optimal performance                         {Colors.CYAN}║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
        print(goodbye)
        
    def ask_continue(self) -> bool:
        """Ask if user wants to continue with another paper"""
        self.print_separator("─", 79, Colors.YELLOW)
        print(f"\n{Colors.BOLD}{Colors.YELLOW}🔄 Process another paper?{Colors.ENDC}")
        choice = input(f"{Colors.OKCYAN}Continue? (y/n): {Colors.ENDC}").strip().lower()
        return choice in ['y', 'yes', '1', 'true']
        
    def add_to_history(self, input_source: str, result: dict):
        """Add processing result to history"""
        entry = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'input_source': input_source,
            'status': result.get('status', 'unknown'),
            'result': result
        }
        self.processing_history.append(entry)
        
    def show_history(self):
        """Display processing history"""
        if not self.processing_history:
            self.print_status("No processing history available", "info")
            return
            
        print(f"\n{Colors.BOLD}{Colors.CYAN}📚 PROCESSING HISTORY{Colors.ENDC}")
        self.print_separator("─", 79, Colors.CYAN)
        
        for i, entry in enumerate(self.processing_history, 1):
            status_icon = "✅" if entry['status'] == 'success' else "❌"
            source = entry['input_source']
            if len(source) > 50:
                source = source[:47] + "..."
                
            print(f"{i}. {status_icon} {entry['timestamp']} | {source}")
            
        self.print_separator("─", 79, Colors.CYAN) 