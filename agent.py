import re
import json
import streamlit as st
from typing_extensions import TypedDict
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END, START
from langchain_groq.chat_models import ChatGroq
from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import datetime

load_dotenv()

tavily_client = TavilyClient()

##  GraphState with new features
class GraphState(TypedDict):
    user_request: str
    requirements_docs_content: str
    requirements_docs_summary: str
    testcases_format: str
    testcases: str
    answer: str
    tavily_search_results: str
    industry_best_practices: str
    _testcases: str
    validation_results: str
    coverage_analysis: str
    risk_assessment: str
    test_metrics: Dict[str, Any]
    requirements_traceability: str
    defect_prediction: str
    test_prioritization: str
    execution_plan: str
    final_report: str

## Requirements Analysis and Risk Assessment
def analyze_requirements_risk_node_function(state: GraphState) -> GraphState:
    requirements_content = state.get("requirements_docs_content", "")
    
    if "llm" not in st.session_state:
        raise RuntimeError("LLM not initialized. Please call initialize_app first.")

    prompt = (
        "You are a senior QA analyst specializing in risk assessment and requirements analysis.\n"
        "Analyze the given requirements document and provide:\n"
        "1. Complexity Analysis (High/Medium/Low for each feature)\n"
        "2. Risk Assessment (identify high-risk areas that need thorough testing)\n"
        "3. Dependencies and Integration Points\n"
        "4. Data Flow Analysis\n"
        "5. Security and Performance Considerations\n\n"
        f"Requirements Document: {requirements_content}\n\n"
        "Provide your analysis in structured format with clear sections."
    )
    
    try:
        response = st.session_state.llm.invoke(prompt)
        state['risk_assessment'] = response.content
    except Exception as e:
        state['risk_assessment'] = f"Error in risk assessment: {str(e)}"
        
    return state

##  Summary Generation with Traceability
def generate_summary_node_function(state: GraphState) -> GraphState:
    requirements_docs_content = state.get("requirements_docs_content", "")
    if "llm" not in st.session_state:
        raise RuntimeError("LLM not initialized. Please call initialize_app first.")

    prompt = (
        "You are an expert in requirements analysis and test planning.\n"
        "Study the given 'Requirements Documents Content' and provide:\n"
        "1. A comprehensive 5-line summary\n"
        "2. Key functional requirements (numbered list)\n"
        "3. Non-functional requirements (performance, security, usability)\n"
        "4. Business rules and constraints\n"
        "5. User personas and scenarios\n"
        "6. Integration requirements\n\n"
        f"Requirements Documents Content: {requirements_docs_content}\n\n"
        "Format your response with clear headings for each section."
    )
    
    try:
        response = st.session_state.llm.invoke(prompt)
        state['requirements_docs_summary'] = response.content
        
        # Generate requirements traceability matrix
        traceability_prompt = (
            "Based on the requirements analysis, create a requirements traceability structure.\n"
            "List each requirement with a unique ID (REQ-001, REQ-002, etc.) and categorize them as:\n"
            "- Functional (F)\n"
            "- Non-Functional (NF)\n"
            "- Business Rule (BR)\n"
            "- Integration (INT)\n\n"
            f"Requirements: {response.content}\n\n"
            "Provide in format: REQ-ID | Category | Description | Priority (High/Medium/Low)"
        )
        
        traceability_response = st.session_state.llm.invoke(traceability_prompt)
        state['requirements_traceability'] = traceability_response.content
        state['answer'] = response.content
        
    except Exception as e:
        state['requirements_docs_summary'] = f"Error generating summary: {str(e)}"
        state['requirements_traceability'] = "Traceability matrix could not be generated."
        state['answer'] = f"Error: {str(e)}"
        
    return state

##  Best Practices Search with ML Insights
def search_best_practices_node_function(state: GraphState) -> GraphState:
    summary = state.get("requirements_docs_summary", "")
    user_request = state.get("user_request", "")
    testcases_format = state.get("testcases_format", "")
    risk_assessment = state.get("risk_assessment", "")
    
    search_queries = [
        f"Best practices {testcases_format} testing automation 2024",
        f"Test coverage strategies for {summary[:100]}",
        f"Quality assurance metrics and KPIs software testing",
        f"Test case optimization and prioritization techniques"
    ]
    
    all_results = ""
    
    try:
        for query in search_queries:
            search_results = tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=2
            )
            
            for result in search_results.get("results", []):
                all_results += f"- {result.get('title', 'No title')}\n"
                all_results += f"  {result.get('content', 'No content')[:200]}...\n\n"
        
        state["tavily_search_results"] = all_results
        
        #  best practices extraction with ML insights
        prompt = (
            "You are a senior QA architect with expertise in modern testing methodologies.\n"
            "Based on the search results and risk assessment, provide comprehensive best practices including:\n"
            "1. Testing Strategy Recommendations\n"
            "2. Automation Framework Guidelines\n"
            "3. Test Data Management\n"
            "4. CI/CD Integration Points\n"
            "5. Quality Metrics and KPIs\n"
            "6. Risk Mitigation Strategies\n"
            "7. Performance Testing Considerations\n"
            "8. Security Testing Requirements\n\n"
            f"SEARCH RESULTS:\n{all_results}\n\n"
            f"REQUIREMENTS SUMMARY:\n{summary}\n\n"
            f"RISK ASSESSMENT:\n{risk_assessment}\n\n"
            f"TEST FORMAT: {testcases_format}\n\n"
            "Provide actionable recommendations with modern testing approaches."
        )
        
        best_practices_response = st.session_state.llm.invoke(prompt)
        state["industry_best_practices"] = best_practices_response.content
        
    except Exception as e:
        state["tavily_search_results"] = f"Error performing search: {str(e)}"
        state["industry_best_practices"] = "Could not retrieve industry best practices."
    
    return state

##  router with more options
def _route_user_request(state: GraphState) -> str:
    preselected_format = state.get("testcases_format", "auto")
    
    if preselected_format != "auto":
        return preselected_format

    user_request = state["user_request"]
    tool_selection = {
        "gherkin_format": "Generate BDD test cases in Gherkin format with Given-When-Then structure",
        "selenium_format": "Generate automated UI test cases in Selenium WebDriver format",
        "api_format": "Generate API test cases for REST/GraphQL endpoints",
        "performance_format": "Generate performance and load test scenarios",
        "security_format": "Generate security test cases including authentication and authorization"
    }

    SYS_PROMPT = """Act as an intelligent test format router. Analyze the user's request and determine 
                    the most appropriate test format based on:
                    - Keywords in the request (UI, API, performance, security, behavior, etc.)
                    - Context from requirements
                    - Testing objectives
                    Output only the exact key from the tool selection dictionary."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT),
        ("human", """User request: {user_request}
                    Tool options: {tool_selection}
                    Output the most relevant format key only.""")
    ])

    inputs = {"user_request": user_request, "tool_selection": tool_selection}
    tool = (prompt | st.session_state.llm | StrOutputParser()).invoke(inputs)
    tool = re.sub(r"[\\'\"`]", "", tool.strip()).lower()

    # Map to simplified routing
    if "gherkin" in tool:
        format_type = "gherkin"
    elif "selenium" in tool:
        format_type = "selenium"
    elif "api" in tool:
        format_type = "api"
    elif "performance" in tool:
        format_type = "performance"
    elif "security" in tool:
        format_type = "security"
    else:
        format_type = "gherkin"  # default
        
    state["testcases_format"] = format_type
    return format_type

##  test case generation with multiple formats
def generate_testcases(user_request, requirements_content, traceability, llm, format_type, best_practices="", risk_assessment=""):
    base_prompt = (
        "You are a senior test architect specializing in comprehensive test case design.\n"
        "Generate detailed, production-ready test cases with proper formatting.\n\n"
        "IMPORTANT FORMATTING RULES:\n"
        "- Each field should be on a separate line\n"
        "- Use double line breaks between test cases\n"
        "- Bold field names followed by a colon and the value\n"
        "- For test steps, use numbered lists with proper line breaks\n"
        "- For expected results, use bullet points with proper line breaks\n\n"
        "FORMAT STRUCTURE FOR EACH TEST CASE:\n"
        "**Test Case ID:** [ID]\n"
        "**Test Case Name:** [Name]\n"
        "**Requirements Traceability:** [REQ-IDs]\n"
        "**Test Objective:** [Objective]\n"
        "**Preconditions:** [Preconditions]\n"
        "**Test Steps:**\n"
        "1. [Step 1]\n"
        "2. [Step 2]\n"
        "3. [Step 3]\n"
        "**Expected Results:**\n"
        "* [Result 1]\n"
        "* [Result 2]\n"
        "**Priority:** [High/Medium/Low]\n"
        "**Test Data Requirements:** [Data needed]\n"
        "**Environment Requirements:** [Environment details]\n\n"
        "---\n\n"
        f"User Request: {user_request}\n"
        f"Requirements: {requirements_content}\n"
        f"Requirements Traceability: {traceability}\n"
    )
    
    if best_practices:
        base_prompt += f"Industry Best Practices: {best_practices}\n"
    if risk_assessment:
        base_prompt += f"Risk Assessment: {risk_assessment}\n"
    
    format_specific_prompts = {
        "gherkin": base_prompt + "\nGenerate 5-7 BDD scenarios in Gherkin format with comprehensive Given-When-Then steps. Include background scenarios and scenario outlines with examples.",
        
        "selenium": base_prompt + "\nGenerate 5-7 Selenium WebDriver test cases with detailed locator strategies, wait conditions, and assertion methods. Include page object model references.",
        
        "api": base_prompt + "\nGenerate 5-7 API test cases covering CRUD operations, authentication, error handling, and data validation. Include request/response examples and status codes.",
        
        "performance": base_prompt + "\nGenerate 5-7 performance test scenarios covering load testing, stress testing, volume testing, and endurance testing. Include performance thresholds and metrics.",
        
        "security": base_prompt + "\nGenerate 5-7 security test cases covering authentication, authorization, input validation, session management, and data protection. Include OWASP guidelines."
    }
    
    prompt = format_specific_prompts.get(format_type, format_specific_prompts["gherkin"])
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating test cases: {str(e)}"

## Test Case Validation Node
def validate_testcases_node_function(state: GraphState) -> GraphState:
    testcases = state.get("testcases", "")
    requirements_traceability = state.get("requirements_traceability", "")
    testcases_format = state.get("testcases_format", "")
    
    if "llm" not in st.session_state:
        raise RuntimeError("LLM not initialized.")
    
    validation_prompt = (
        "You are a QA validation expert. Analyze the generated test cases and provide a comprehensive validation report:\n\n"
        "VALIDATION CRITERIA:\n"
        "1. Completeness: Do test cases cover all requirements?\n"
        "2. Correctness: Are test steps logical and executable?\n"
        "3. Traceability: Can each test case be traced back to requirements?\n"
        "4. Coverage: Are all scenarios (positive, negative, edge cases) covered?\n"
        "5. Clarity: Are test cases clear and unambiguous?\n"
        "6. Maintainability: Are test cases easy to update and maintain?\n"
        "7. Format Compliance: Do test cases follow the specified format standards?\n\n"
        f"TEST CASES TO VALIDATE:\n{testcases}\n\n"
        f"REQUIREMENTS TRACEABILITY:\n{requirements_traceability}\n\n"
        f"FORMAT: {testcases_format}\n\n"
        "Provide validation results with:\n"
        "- Overall Quality Score (1-10)\n"
        "- Issues Found (High/Medium/Low priority)\n"
        "- Recommendations for improvement\n"
        "- Missing test scenarios\n"
        "- Traceability gaps"
    )
    
    try:
        validation_response = st.session_state.llm.invoke(validation_prompt)
        state["validation_results"] = validation_response.content
    except Exception as e:
        state["validation_results"] = f"Validation error: {str(e)}"
    
    return state

## Coverage Analysis Node
def analyze_coverage_node_function(state: GraphState) -> GraphState:
    testcases = state.get("testcases", "")
    requirements_summary = state.get("requirements_docs_summary", "")
    requirements_traceability = state.get("requirements_traceability", "")
    
    coverage_prompt = (
        "You are a test coverage analyst. Analyze the test coverage and provide detailed metrics:\n\n"
        "COVERAGE ANALYSIS AREAS:\n"
        "1. Requirements Coverage: % of requirements covered by test cases\n"
        "2. Functional Coverage: Coverage of all functional areas\n"
        "3. Scenario Coverage: Positive, negative, and edge case coverage\n"
        "4. Data Coverage: Different data combinations and boundary values\n"
        "5. User Journey Coverage: End-to-end user scenarios\n"
        "6. Integration Coverage: System integration points\n\n"
        f"TEST CASES:\n{testcases}\n\n"
        f"REQUIREMENTS:\n{requirements_summary}\n\n"
        f"TRACEABILITY MATRIX:\n{requirements_traceability}\n\n"
        "Provide coverage analysis with:\n"
        "- Coverage percentages for each area\n"
        "- Coverage gaps and missing scenarios\n"
        "- Recommendations to improve coverage\n"
        "- Priority areas needing additional tests\n"
        "- Coverage matrix mapping tests to requirements"
    )
    
    try:
        coverage_response = st.session_state.llm.invoke(coverage_prompt)
        state["coverage_analysis"] = coverage_response.content
    except Exception as e:
        state["coverage_analysis"] = f"Coverage analysis error: {str(e)}"
    
    return state

## Test Prioritization and Execution Planning
def prioritize_and_plan_node_function(state: GraphState) -> GraphState:
    testcases = state.get("testcases", "")
    risk_assessment = state.get("risk_assessment", "")
    validation_results = state.get("validation_results", "")
    
    prioritization_prompt = (
        "You are a test execution planning expert. Create a comprehensive test execution plan:\n\n"
        "PRIORITIZATION FACTORS:\n"
        "1. Business Impact and Risk Level\n"
        "2. Requirements Criticality\n"
        "3. Complexity and Dependencies\n"
        "4. Historical Defect Patterns\n"
        "5. Customer Usage Patterns\n\n"
        f"TEST CASES:\n{testcases}\n\n"
        f"RISK ASSESSMENT:\n{risk_assessment}\n\n"
        f"VALIDATION RESULTS:\n{validation_results}\n\n"
        "Generate:\n"
        "1. Test Case Prioritization (P0/P1/P2/P3)\n"
        "2. Execution Sequence and Dependencies\n"
        "3. Resource Allocation Recommendations\n"
        "4. Timeline Estimation\n"
        "5. Parallel Execution Opportunities\n"
        "6. Environment and Data Setup Requirements\n"
        "7. Entry and Exit Criteria\n"
        "8. Risk Mitigation Strategies"
    )
    
    try:
        prioritization_response = st.session_state.llm.invoke(prioritization_prompt)
        state["test_prioritization"] = prioritization_response.content
        
        # Generate execution plan
        execution_prompt = (
            f"Based on the prioritization analysis, create a detailed execution plan:\n"
            f"{prioritization_response.content}\n\n"
            "Generate a structured execution plan with:\n"
            "- Phase 1: Smoke/Sanity Tests (P0)\n"
            "- Phase 2: Core Functionality Tests (P1)\n"
            "- Phase 3: Extended Testing (P2)\n"
            "- Phase 4: Nice-to-have Tests (P3)\n"
            "Include estimated hours, dependencies, and resource requirements for each phase."
        )
        
        execution_response = st.session_state.llm.invoke(execution_prompt)
        state["execution_plan"] = execution_response.content
        
    except Exception as e:
        state["test_prioritization"] = f"Prioritization error: {str(e)}"
        state["execution_plan"] = f"Execution planning error: {str(e)}"
    
    return state

## Defect Prediction using AI
def predict_defects_node_function(state: GraphState) -> GraphState:
    requirements_summary = state.get("requirements_docs_summary", "")
    risk_assessment = state.get("risk_assessment", "")
    testcases = state.get("testcases", "")
    
    prediction_prompt = (
        "You are an AI-powered defect prediction specialist with access to historical defect patterns.\n"
        "Analyze the requirements and test cases to predict potential defect-prone areas:\n\n"
        "PREDICTION FACTORS:\n"
        "1. Code Complexity Indicators\n"
        "2. Integration Points and Dependencies\n"
        "3. Data Flow Complexity\n"
        "4. User Interface Complexity\n"
        "5. Business Logic Complexity\n"
        "6. Historical Defect Patterns in Similar Features\n\n"
        f"REQUIREMENTS:\n{requirements_summary}\n\n"
        f"RISK ASSESSMENT:\n{risk_assessment}\n\n"
        f"TEST CASES:\n{testcases}\n\n"
        "Provide defect predictions with:\n"
        "1. High-risk areas likely to have defects\n"
        "2. Predicted defect types (functional, UI, integration, performance)\n"
        "3. Severity and impact predictions\n"
        "4. Recommended additional testing focus areas\n"
        "5. Defect prevention strategies\n"
        "6. Monitoring and early detection recommendations"
    )
    
    try:
        prediction_response = st.session_state.llm.invoke(prediction_prompt)
        state["defect_prediction"] = prediction_response.content
    except Exception as e:
        state["defect_prediction"] = f"Defect prediction error: {str(e)}"
    
    return state

## Generate Comprehensive Final Report
def generate_final_report_node_function(state: GraphState) -> GraphState:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate test metrics
    testcases_content = state.get("testcases", "")
    test_count = len([line for line in testcases_content.split('\n') if 'Test Case' in line or 'Scenario:' in line])
    
    metrics = {
        "total_test_cases": test_count,
        "generation_timestamp": timestamp,
        "format_used": state.get("testcases_format", "unknown"),
        "requirements_analyzed": "Yes" if state.get("requirements_docs_content") else "No",
        "validation_performed": "Yes" if state.get("validation_results") else "No",
        "coverage_analyzed": "Yes" if state.get("coverage_analysis") else "No"
    }
    
    state["test_metrics"] = metrics
    
    # Generate comprehensive report
    report_sections = []
    
    if state.get("requirements_docs_summary"):
        report_sections.append(f"## Requirements Summary\n{state['requirements_docs_summary']}\n")
    
    if state.get("risk_assessment"):
        report_sections.append(f"## Risk Assessment\n{state['risk_assessment']}\n")
    
    if state.get("testcases"):
        report_sections.append(f"## Generated Test Cases\n{state['testcases']}\n")
    
    if state.get("validation_results"):
        report_sections.append(f"## Validation Results\n{state['validation_results']}\n")
    
    if state.get("coverage_analysis"):
        report_sections.append(f"## Coverage Analysis\n{state['coverage_analysis']}\n")
    
    if state.get("defect_prediction"):
        report_sections.append(f"## Defect Predictions\n{state['defect_prediction']}\n")
    
    if state.get("test_prioritization"):
        report_sections.append(f"## Test Prioritization\n{state['test_prioritization']}\n")
    
    if state.get("execution_plan"):
        report_sections.append(f"## Execution Plan\n{state['execution_plan']}\n")
    
    # Add metrics section
    metrics_section = f"""## Test Generation Metrics
- **Total Test Cases Generated:** {metrics['total_test_cases']}
- **Test Format:** {metrics['format_used'].upper()}
- **Generation Timestamp:** {metrics['generation_timestamp']}
- **Requirements Analyzed:** {metrics['requirements_analyzed']}
- **Validation Performed:** {metrics['validation_performed']}
- **Coverage Analysis:** {metrics['coverage_analyzed']}
"""
    
    final_report = f"""# Test Case Generation Report
Generated by  Test Case Generation Agent

{metrics_section}

{"".join(report_sections)}

## Recommendations
Based on the comprehensive analysis, follow the execution plan and prioritization recommendations. 
Pay special attention to the predicted defect-prone areas and ensure thorough testing coverage 
for high-risk components.

---
*Report generated on {timestamp}*
"""
    
    state["final_report"] = final_report
    state["answer"] = final_report
    return state

## Individual test case generation nodes for different formats
def generate_gherkin_testcases_node_function(state: GraphState) -> GraphState:
    user_request = state["user_request"]
    requirements_docs_content = state.get("requirements_docs_content", "")
    requirements_traceability = state.get("requirements_traceability", "")
    best_practices = state.get("industry_best_practices", "")
    risk_assessment = state.get("risk_assessment", "")
    
    response = generate_testcases(
        user_request, requirements_docs_content, requirements_traceability,
        st.session_state.llm, "gherkin", best_practices, risk_assessment
    )

    state["testcases_format"] = "gherkin"
    state['testcases'] = response
    return state

def generate_selenium_testcases_node_function(state: GraphState) -> GraphState:
    user_request = state["user_request"]
    requirements_docs_content = state.get("requirements_docs_content", "")
    requirements_traceability = state.get("requirements_traceability", "")
    best_practices = state.get("industry_best_practices", "")
    risk_assessment = state.get("risk_assessment", "")
    
    response = generate_testcases(
        user_request, requirements_docs_content, requirements_traceability,
        st.session_state.llm, "selenium", best_practices, risk_assessment
    )
    
    state["testcases_format"] = "selenium"
    state['testcases'] = response
    return state

def generate_api_testcases_node_function(state: GraphState) -> GraphState:
    user_request = state["user_request"]
    requirements_docs_content = state.get("requirements_docs_content", "")
    requirements_traceability = state.get("requirements_traceability", "")
    best_practices = state.get("industry_best_practices", "")
    risk_assessment = state.get("risk_assessment", "")
    
    response = generate_testcases(
        user_request, requirements_docs_content, requirements_traceability,
        st.session_state.llm, "api", best_practices, risk_assessment
    )
    
    state["testcases_format"] = "api"
    state['testcases'] = response
    return state

def generate_performance_testcases_node_function(state: GraphState) -> GraphState:
    user_request = state["user_request"]
    requirements_docs_content = state.get("requirements_docs_content", "")
    requirements_traceability = state.get("requirements_traceability", "")
    best_practices = state.get("industry_best_practices", "")
    risk_assessment = state.get("risk_assessment", "")
    
    state["testcases_format"] = "performance"
    response = generate_testcases(
        user_request, requirements_docs_content, requirements_traceability,
        st.session_state.llm, "performance", best_practices, risk_assessment
    )
    
    state['testcases'] = response
    return state

def generate_security_testcases_node_function(state: GraphState) -> GraphState:
    user_request = state["user_request"]
    requirements_docs_content = state.get("requirements_docs_content", "")
    requirements_traceability = state.get("requirements_traceability", "")
    best_practices = state.get("industry_best_practices", "")
    risk_assessment = state.get("risk_assessment", "")
    
    response = generate_testcases(
        user_request, requirements_docs_content, requirements_traceability,
        st.session_state.llm, "security", best_practices, risk_assessment
    )
    
    state["testcases_format"] = "security"
    state['testcases'] = response
    return state

##  workflow with comprehensive pipeline
def build_workflow():
    workflow = StateGraph(GraphState)
    
    # Add all nodes
    workflow.add_node("_summary_node", generate_summary_node_function)
    workflow.add_node("risk_analysis_node", analyze_requirements_risk_node_function)
    workflow.add_node("_best_practices_node", search_best_practices_node_function)
    workflow.add_node("gherkin_node", generate_gherkin_testcases_node_function)
    workflow.add_node("selenium_node", generate_selenium_testcases_node_function)
    workflow.add_node("api_node", generate_api_testcases_node_function)
    workflow.add_node("performance_node", generate_performance_testcases_node_function)
    workflow.add_node("security_node", generate_security_testcases_node_function)
    workflow.add_node("validation_node", validate_testcases_node_function)
    workflow.add_node("coverage_analysis_node", analyze_coverage_node_function)
    workflow.add_node("defect_prediction_node", predict_defects_node_function)
    workflow.add_node("prioritization_node", prioritize_and_plan_node_function)
    workflow.add_node("final_report_node", generate_final_report_node_function)
    
    # Set workflow structure
    workflow.set_entry_point("_summary_node")
    workflow.add_edge("_summary_node", "risk_analysis_node")
    workflow.add_edge("risk_analysis_node", "_best_practices_node")
    
    # Conditional routing based on test format
    workflow.add_conditional_edges(
        "_best_practices_node",
        _route_user_request,
        {
            "gherkin": "gherkin_node",
            "selenium": "selenium_node",
            "api": "api_node",
            "performance": "performance_node",
            "security": "security_node"
        }
    )
    
    # All test generation paths lead to validation
    for node in ["gherkin_node", "selenium_node", "api_node", "performance_node", "security_node"]:
        workflow.add_edge(node, "validation_node")
    
    # Continue with analysis pipeline
    workflow.add_edge("validation_node", "coverage_analysis_node")
    workflow.add_edge("coverage_analysis_node", "defect_prediction_node")
    workflow.add_edge("defect_prediction_node", "prioritization_node")
    workflow.add_edge("prioritization_node", "final_report_node")
    workflow.add_edge("final_report_node", END)
    
    return workflow

## Initialize  app
def initialize_app(model_name: str):
    """Initialize the  app with comprehensive testing capabilities."""
    if "selected_model" in st.session_state and st.session_state.selected_model == model_name:
        return build_workflow().compile()

    st.session_state.llm = ChatGroq(model=model_name, temperature=0.0)
    st.session_state.selected_model = model_name
    print(f"Using  model: {model_name}")
    return build_workflow().compile()