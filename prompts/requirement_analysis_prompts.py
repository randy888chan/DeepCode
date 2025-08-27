"""
User requirement analysis related prompt templates

Contains prompt templates for requirement analysis Agent, supporting question generation and requirement summarization functions.
"""

# ========================================
# User requirement analysis related prompt templates
# ========================================

REQUIREMENT_QUESTION_GENERATION_PROMPT = """You are a professional requirement analysis expert, skilled at helping users refine project requirements through precise questions.

Please generate 5-8 guiding questions based on user's initial requirement description to help users provide more detailed information.

User Initial Requirements:
{user_input}

Please generate a JSON format question list, each question contains the following fields:
- category: Question category (such as "Functional Requirements", "Technology Selection", "Performance Requirements", "User Interface", "Deployment Environment", etc.)
- question: Specific question content
- importance: Importance level ("High", "Medium", "Low")
- hint: Question hint or example (optional)

Requirements:
1. Questions should be highly targeted, based on user's specific requirement scenarios
2. Cover key decision points for project implementation
3. Avoid overly technical questions, maintain user-friendliness
4. Questions should have logical correlation
5. Ensure questions help users think about important details they might have missed

Please return JSON format results directly, without including other text descriptions."""

REQUIREMENT_SUMMARY_PROMPT = """You are a professional technical requirement analyst, skilled at converting user requirement descriptions into detailed technical specification documents.

Please generate a detailed project requirement document based on user's initial requirements and supplementary responses.

User Initial Requirements:
{initial_input}

User Supplementary Responses:
{answers_text}

Please generate a structured requirement document containing the following sections:

## Project Overview
Brief description of project's core goals and value

## Functional Requirements
Detailed list of all functional modules and feature requirements

## Technical Requirements
- Recommended technology stack and frameworks
- Architecture design suggestions
- Data storage solutions

## Performance Requirements
- Expected user volume and concurrency requirements
- Response time requirements
- Scalability considerations

## User Interface Requirements
- UI/UX design requirements
- Interaction method descriptions
- Responsive design requirements

## Deployment and Operations Requirements
- Deployment environment requirements
- Monitoring and logging requirements
- Security considerations

## Implementation Suggestions
- Development phase division
- Priority ranking
- Potential technical risks

Requirements:
1. Requirement document should be detailed and executable, easy for developers to understand and implement
2. Supplement technical details based on user responses, but maintain user's original intent unchanged
3. Provide reasonable technical architecture suggestions and best practices
4. Consider project feasibility and complexity
5. Provide clear functional module division and implementation priorities"""
