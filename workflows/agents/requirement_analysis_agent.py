"""
User Requirement Analysis Agent

Responsible for analyzing user initial requirements, generating guiding questions, 
and summarizing detailed requirement documents based on user responses.
This Agent seamlessly integrates with existing chat workflows to provide more precise requirement understanding.
"""

import json
import logging
from typing import Dict, Any, List, Optional

from mcp_agent.agents.agent import Agent
from utils.llm_utils import get_preferred_llm_class


class RequirementAnalysisAgent:
    """
    User Requirement Analysis Agent

    Core Functions:
    1. Generate 5-8 guiding questions based on user initial requirements
    2. Collect user responses and analyze requirement completeness
    3. Generate detailed requirement documents for subsequent workflows
    4. Support skipping questions to directly enter implementation process

    Design Philosophy:ß
    - Intelligent question generation covering functionality, technology, performance, UI, deployment dimensions
    - Flexible user interaction supporting partial answers or complete skipping
    - Structured requirement output for easy understanding by code generation agents
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize requirement analysis agent
        Args:
            logger: Logger instance
        """
        self.logger = logger or self._create_default_logger()
        self.mcp_agent = None
        self.llm = None

    def _create_default_logger(self) -> logging.Logger:
        """Create default logger"""
        logger = logging.getLogger(f"{__name__}.RequirementAnalysisAgent")
        logger.setLevel(logging.INFO)
        return logger

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

    async def initialize(self):
        """Initialize MCP Agent connection and LLM"""
        try:
            self.mcp_agent = Agent(
                name="RequirementAnalysisAgent",
                instruction="""You are a professional requirement analysis expert, skilled at guiding users to provide more detailed project requirements through precise questions.

Your core capabilities:
1. **Intelligent Question Generation**: Based on user initial descriptions, generate 5-8 key questions covering functional requirements, technology selection, performance requirements, user interface, deployment environment, etc.
2. **Requirement Understanding Analysis**: Deep analysis of user's real intentions and implicit requirements
3. **Structured Requirement Output**: Integrate scattered requirement information into clear technical specification documents

Question Generation Principles:
- Questions should be specific and clear, avoiding overly broad scope
- Cover key decision points for technical implementation
- Consider project feasibility and complexity
- Help users think about important details they might have missed

Requirement Summary Principles:
- Maintain user's original intent unchanged
- Supplement key information for technical implementation
- Provide clear functional module division
- Give reasonable technical architecture suggestions""",
                server_names=[],  # No MCP servers needed, only use LLM
            )

            # Initialize agent context
            await self.mcp_agent.__aenter__()

            # Attach LLM
            self.llm = await self.mcp_agent.attach_llm(get_preferred_llm_class())

            self.logger.info("RequirementAnalysisAgent initialized successfully")

        except Exception as e:
            self.logger.error(f"RequirementAnalysisAgent initialization failed: {e}")
            raise

    async def cleanup(self):
        """Clean up resources"""
        if self.mcp_agent:
            try:
                await self.mcp_agent.__aexit__(None, None, None)
            except Exception as e:
                self.logger.warning(f"Error during resource cleanup: {e}")

    async def generate_guiding_questions(self, user_input: str) -> List[Dict[str, str]]:
        """
        Generate guiding questions based on user initial requirements
        
        Args:
            user_input: User's initial requirement description
            
        Returns:
            List[Dict]: Question list, each question contains category, question, importance and other fields
        """
        try:
            self.logger.info("Starting to generate AI precise guiding questions")
            
            # Build more precise prompt
            prompt = f"""Based on user's project requirements, generate precise guiding questions to help refine requirements.

User Requirements: {user_input}

Please analyze user requirements and generate 5-6 targeted questions covering the following key dimensions:
1. Core functionality refinement
2. Technical architecture selection
3. User interaction experience
4. Performance and scalability
5. Deployment and operations

Return format (pure JSON array, no other text):
[
  {{
    "category": "Functional Requirements",
    "question": "Specific question content",
    "importance": "High",
    "hint": "Question hint"
  }}
]

Requirements: Questions should be specific and practical, avoiding general discussions."""

            from mcp_agent.workflows.llm.augmented_llm import RequestParams
            
            params = RequestParams(
                max_tokens=3000,
                temperature=0.5  # Lower temperature for more stable JSON output
            )

            self.logger.info(f"Calling LLM to generate precise questions, input length: {len(user_input)}")
            
            result = await self.llm.generate_str(
                message=prompt,
                request_params=params
            )
            
            self.logger.info(f"LLM returned result length: {len(result) if result else 0}")
            
            if not result or not result.strip():
                self.logger.error("LLM returned empty result")
                raise ValueError("LLM returned empty result")
            
            self.logger.info(f"LLM returned result: {result[:500]}...")

            # Clean result and extract JSON part
            result_cleaned = result.strip()
            
            # Try to find JSON array
            import re
            json_pattern = r'\[\s*\{.*?\}\s*\]'
            json_match = re.search(json_pattern, result_cleaned, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                self.logger.info(f"Extracted JSON: {json_str[:200]}...")
            else:
                # If complete JSON not found, try direct parsing
                json_str = result_cleaned
            
            # Parse JSON result
            try:
                questions = json.loads(json_str)
                if isinstance(questions, list) and len(questions) > 0:
                    self.logger.info(f"✅ Successfully generated {len(questions)} AI precise guiding questions")
                    return questions
                else:
                    raise ValueError("Returned result is not a valid question list")
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing failed: {e}")
                self.logger.error(f"Original result: {result}")
                
                # Try more lenient JSON extraction
                lines = result.split('\n')
                json_lines = []
                in_json = False
                
                for line in lines:
                    if '[' in line:
                        in_json = True
                    if in_json:
                        json_lines.append(line)
                    if ']' in line and in_json:
                        break
                
                if json_lines:
                    try:
                        json_attempt = '\n'.join(json_lines)
                        questions = json.loads(json_attempt)
                        if isinstance(questions, list) and len(questions) > 0:
                            self.logger.info(f"✅ Generated {len(questions)} questions through lenient parsing")
                            return questions
                    except Exception:
                        pass
                
                # Last fallback: generate smart default questions based on user input
                self.logger.warning("JSON parsing completely failed, generating smart default questions")
                return self._generate_smart_default_questions(user_input)

        except Exception as e:
            self.logger.error(f"Failed to generate guiding questions: {e}")
            # Generate smart default questions based on user input
            return self._generate_smart_default_questions(user_input)

    async def summarize_detailed_requirements(self, 
                                            initial_input: str, 
                                            answers: Dict[str, str]) -> str:
        """
        Generate detailed requirement document based on initial input and user answers
        
        Args:
            initial_input: User's initial requirement description
            answers: User's answer dictionary {question_id: answer}
            
        Returns:
            str: Detailed requirement document
        """
        try:
            self.logger.info("Starting to generate AI detailed requirement summary")
            
            # Build answer content
            answers_text = ""
            if answers:
                for question_id, answer in answers.items():
                    if answer and answer.strip():
                        answers_text += f"• {answer}\n"
            
            if not answers_text:
                answers_text = "User chose to skip questions, generating based on initial requirements"
            
            prompt = f"""Based on user requirements and responses, generate detailed project requirement document.

Initial Requirements: {initial_input}

Additional Information:
{answers_text}

Please generate complete requirement document including:

## Project Overview
Project goals and core value

## Functional Requirements  
Detailed feature list and characteristics

## Technical Architecture
- Recommended technology stack
- System architecture design
- Data storage solution

## Performance & Scalability
- Performance metric requirements
- Scalability considerations

## User Experience
- Interface design requirements
- Interaction workflow

## Deployment & Operations
- Deployment solution
- Monitoring and logging
- Security considerations

## Implementation Plan
- Development phases
- Priority ranking
- Risk points

Requirements: Specific and executable, convenient for development implementation."""

            from mcp_agent.workflows.llm.augmented_llm import RequestParams
            
            params = RequestParams(
                max_tokens=4000,
                temperature=0.3
            )

            self.logger.info(f"Calling LLM to generate requirement summary, initial requirement length: {len(initial_input)}")
            
            result = await self.llm.generate_str(
                message=prompt,
                request_params=params
            )

            if not result or not result.strip():
                self.logger.error("LLM returned empty requirement summary")
                raise ValueError("LLM returned empty requirement summary")

            self.logger.info(f"✅ Requirement summary generation completed, length: {len(result)}")
            return result.strip()

        except Exception as e:
            self.logger.error(f"Requirement summary failed: {e}")
            # Return basic requirement document
            return f"""## Project Overview
Based on user requirements: {initial_input}

## Functional Requirements
{initial_input}

## Technical Requirements
- Select appropriate technology stack based on project requirements
- Adopt modular architecture design
- Consider scalability and maintainability

## Implementation Suggestions
- Develop in phases, prioritize core functionality implementation
- Focus on code quality and documentation completeness
- Thorough testing to ensure system stability

Note: Due to technical issues, this is a simplified requirement document. Manual supplementation of detailed information is recommended."""

    def _generate_smart_default_questions(self, user_input: str) -> List[Dict[str, str]]:
        """Generate smart default questions based on user input"""
        # Analyze keywords in user input to generate targeted questions
        user_lower = user_input.lower()
        
        questions = []
        
        # Generate related questions based on user input content type
        if any(keyword in user_lower for keyword in ['web', 'website', 'application', 'app', 'site']):
            questions.extend([
                {
                    "category": "Functional Requirements",
                    "question": "Besides core functionality, what auxiliary features are needed? Such as user management, data export, notification system, etc.",
                    "importance": "High",
                    "hint": "Consider the complete user workflow"
                },
                {
                    "category": "User Interface",
                    "question": "Any special requirements for the user interface? Need mobile support? Any specific design style preferences?",
                    "importance": "Medium",
                    "hint": "Consider target user groups and usage scenarios"
                }
            ])
        
        if any(keyword in user_lower for keyword in ['data', 'analysis', 'machine learning', 'ml', 'analytics']):
            questions.extend([
                {
                    "category": "Data Management",
                    "question": "What are the data sources? What scale of data needs to be processed? Any real-time data requirements?",
                    "importance": "High",
                    "hint": "Affects architecture design and technology selection"
                },
                {
                    "category": "Performance Requirements",
                    "question": "What are the processing speed requirements? Need concurrent processing support?",
                    "importance": "Medium",
                    "hint": "Affects algorithm selection and system architecture"
                }
            ])
        
        # General questions
        questions.extend([
            {
                "category": "Technology Selection",
                "question": "Any preferences or constraints on technology stack? What technologies is the team more familiar with?",
                "importance": "High",
                "hint": "Consider team technical capabilities and project maintenance"
            },
            {
                "category": "Deployment Environment",
                "question": "What environment is planned for deployment? Cloud servers, local servers, or containerized deployment?",
                "importance": "Medium",
                "hint": "Affects deployment strategy and operations plan"
            }
        ])
        
        # Ensure at least 5 questions
        if len(questions) < 5:
            questions.append({
                "category": "Project Scale",
                "question": "What is the expected user scale? What are the system scalability requirements?",
                "importance": "Medium",
                "hint": "Affects architecture design and technology selection"
            })
        
        return questions[:6]  # Return at most 6 questions

    def _get_default_questions(self) -> List[Dict[str, str]]:
        """Get default guiding question list"""
        return [
            {
                "category": "Functional Requirements",
                "question": "Besides core functionality, what auxiliary features are needed? Such as user management, data export, notification system, etc.",
                "importance": "High",
                "hint": "Consider the complete user workflow"
            },
            {
                "category": "Technology Selection",
                "question": "Any preferences or constraints on technology stack? Such as must use specific programming languages or frameworks",
                "importance": "High",
                "hint": "Consider team technical capabilities and project maintenance"
            },
            {
                "category": "User Interface",
                "question": "Any special requirements for user interface? Such as responsive design, specific interaction methods, etc.",
                "importance": "Medium",
                "hint": "Consider target user groups and usage scenarios"
            },
            {
                "category": "Performance Requirements",
                "question": "What are the expected user scale and performance requirements? How many concurrent users need to be supported?",
                "importance": "Medium",
                "hint": "Affects architecture design and technology selection"
            },
            {
                "category": "Data Management",
                "question": "What types of data need to be stored? What are the data security requirements?",
                "importance": "High",
                "hint": "Involves database selection and security design"
            },
            {
                "category": "Deployment Environment",
                "question": "What environment is planned for deployment? Cloud servers, local servers, or containerized deployment?",
                "importance": "Medium",
                "hint": "Affects deployment strategy and operations plan"
            }
        ]
