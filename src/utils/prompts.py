# src/utils/prompts.py
"""
Prompts for various LLM tasks in the research system
"""

# Technical validation prompt
technical_validation_instructions = """
You are a technical validator analyzing research findings. Your task is to critically assess 
the technical aspects of the provided information.

Analyze the content for:
1. Technical correctness: Identify any technical inaccuracies, misunderstandings, or errors
2. Implementation feasibility: Evaluate whether proposed solutions or described systems are realistically implementable
3. Potential issues: Flag any technical contradictions, gaps, or areas that need further investigation
4. Recommendations: Suggest technical improvements or further areas to research

Provide your response as a JSON object with these fields:
{
  "overall_assessment": "Brief overall technical assessment",
  "technical_correctness": "Assessment of technical accuracy",
  "implementation_feasibility": "Evaluation of implementation feasibility",
  "potential_issues": ["List of potential technical issues or concerns"],
  "recommendations": ["List of technical recommendations"]
}

Be specific, citing technical details from the content where possible.
Focus on technical aspects rather than general content quality.
"""

# Security analysis prompt
security_analysis_instructions = """
You are a security analyst examining research findings. Your task is to identify
any security implications, vulnerabilities, or concerns in the provided information.

Analyze the content for:
1. Potential security vulnerabilities
2. Sensitive information exposure
3. Authentication/authorization issues
4. Data protection concerns
5. Secure coding practices (or lack thereof)
6. Compliance issues

Return a list of findings in this JSON format:
[
  {
    "type": "The type of security issue (e.g., 'authentication', 'data exposure')",
    "severity": "high|medium|low",
    "description": "Detailed description of the security concern",
    "recommendation": "Recommendation for addressing the issue"
  }
]

If no security concerns are found, return an empty list.
Be specific and technical, and focus only on legitimate security concerns.
"""

# Jira summarization prompt
jira_summarization_prompt = """
Summarize the Jira tickets related to {topic}. Focus on:
1. Key requirements or features described
2. Implementation status and priorities
3. Technical constraints or issues mentioned
4. Dependencies between tickets
5. Timeline information if available

Create a comprehensive summary that synthesizes the information across all tickets.
Highlight conflicts or inconsistencies if present.
"""

# Confluence summarization prompt
confluence_summarization_prompt = """
Summarize the Confluence documentation related to {topic}. Focus on:
1. System architecture and components described
2. APIs, interfaces, and data models
3. Technical requirements and specifications
4. Implementation guidelines
5. Current status information

Create a comprehensive summary that consolidates the documentation information.
Identify any inconsistencies or outdated information if present.
"""

# Perforce code summarization prompt
perforce_summarization_prompt = """
Summarize the code changes and implementations related to {topic}. Focus on:
1. Key components, classes, and functions implemented
2. APIs and interfaces exposed
3. Technical approaches and patterns used
4. Changes and their purpose
5. Implementation status and known issues

Create a comprehensive summary of the code base architecture and implementation.
Highlight any technical debt, workarounds, or issues identified in the code.
"""

# Cross-source analysis prompt
cross_source_analysis_prompt = """
Analyze the relationship between the following information sources related to {topic}:

1. Jira tickets (requirements and issues)
2. Confluence documentation (specifications and designs)
3. Perforce code implementation (actual implementation)

Identify:
- Alignment: Where requirements, documentation, and implementation align
- Gaps: Where implementation is missing for documented requirements
- Contradictions: Where implementation differs from documented requirements
- Documentation gaps: Where implemented features lack proper documentation
- Requirement gaps: Where code implements features without clear requirements

Produce a comprehensive analysis of how these sources relate to each other,
highlighting both strengths and areas for improvement.
"""